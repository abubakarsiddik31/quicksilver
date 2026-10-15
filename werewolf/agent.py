"""Quicksilver's Werewolf policy.

Seven agents: 2 wolves, 1 seer, 4 villagers, **no Doctor** — so every night kill
is a deliberate choice and night 1 is peaceful by construction. One discussion
window per day (2 minutes, at most 5 chat messages of at most 50 words each),
then a day vote; at night the wolves kill and the seer inspects.

Two things make this game different from Pokémon, and they shape the policy:

**It is a language game.** The decision that matters most is *what to say* in a
50-word window, which is what a language model is good at, and the 2-minute
window makes latency almost free. This is the one place in Quicksilver where an
LLM belongs on the critical path.

**The other players are adversaries, not narrators.** In-game messages are
untrusted input. Opponents are known to attempt prompt injection — asking the
agent to reveal its role, to vote a particular way, or to send its credentials
anywhere. ``werewolf.messages`` renders chat as delimited data and flags
obvious attempts, and the flags feed suspicion: someone trying to extract our
role is telling us something about themselves.

The moves themselves are driven by ``werewolf.belief``, which turns the public
record (published dead roles, vote history, and our own investigations when we
are the seer) into a suspicion ranking, then picks by which side we are on.

.. note::

    ``choose_message`` is not implemented yet, so the dispatcher's default
    applies and we vote every discussion closed. That is safe (it costs
    nothing but silence) and it means the baseline currently says nothing at
    all. The speech policy is the next piece of work here; the belief tracker
    and the untrusted-rendering it will need are already in place.
"""

from __future__ import annotations

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, SendMessage
from common import llm, trace
from werewolf import belief as belief_mod
from werewolf import messages as msg
from werewolf import jev, speech, synthesizer


def _seat_from_action_id(action_id: object) -> int | None:
    """Werewolf encodes a seat as the action's id (``"3"``); ``"7"`` is
    abstain."""
    if not isinstance(action_id, str) or not action_id.isdigit():
        return None
    return int(action_id)


class WerewolfAgent:
    """Per-match Werewolf decision logic.

    One instance per match, in that match's own process — which is exactly what
    lets the belief state, the discussion quota and the message history live on
    ``self`` without ever leaking into another game.
    """

    def __init__(self) -> None:
        self.belief = belief_mod.Belief()
        self.quota = msg.Quota()
        self._last_phase: str | None = None
        self._said: set[str] = set()
        self._said_day: int | None = None
        self._jev = jev.from_env()
        # Built lazily: no key configured means no model is ever constructed.
        self._llm = None
        self._llm_checked = False

    def choose_action(self, state: GameState, context: DecisionContext):
        legal = state.legal_actions
        if not legal:
            # An empty list means "not your turn" — the runner only asks when
            # there is something to pick, so this is defensive.
            return None

        self.belief.update(state)

        engine = belief_mod.engine_phase(belief_mod.raw_of(state))
        client = self._client()
        self.belief.note_messages(getattr(state, "new_messages", None), our_position=self.belief.seat)
        window = self.belief.chat[-30:]  # the whole conversation: the vote window itself only holds what is new

        source = "rules"
        target = None
        if engine == "day_vote":
            candidates = self._vote_candidates()
            ranked = self.belief.ranked_suspects()
            if ranked and ranked[0][1] >= belief_mod.KNOWN_WOLF_SCORE:
                target, source = self.belief.best_vote(), "proof"  # a proven wolf needs no second opinion
            else:
                # Jev decides; the Gemini jury and then the rule score only catch a failure.
                target = jev.choose_seat(
                    self._jev, self.belief, window, candidates, jev.VOTE, what="day_vote"
                )
                source = "jev"
                if target is None and client is not None:
                    target = synthesizer.synthesize_day_vote(client, self.belief, window)
                    source = "gemini"
                if target is None:
                    target, source = self.belief.best_vote(), "rules"

        elif engine == "night_wolf":
            candidates = [
                s
                for s in self.belief.alive
                if s != self.belief.seat and s not in self.belief.allies and s not in self.belief.known_wolves
            ]
            target = jev.choose_seat(self._jev, self.belief, window, candidates, jev.KILL, what="night_kill")
            source = "jev"
            if target is None and client is not None:
                target = synthesizer.synthesize_night_kill(client, self.belief, window)
                source = "gemini"
            if target is None:
                target, source = self.belief.best_kill(), "rules"

        elif engine == "night_seer":
            unchecked = self._unchecked()
            target = jev.choose_seat(
                self._jev, self.belief, window, unchecked, jev.INVESTIGATE, what="investigate"
            )
            source = "jev"
            if target is None:
                target, source = self._investigation_target(), "rules"
        trace.write("werewolf", engine=engine, day=self.belief.day, role=self.belief.role, target=target, source=source)

        if target is not None:
            match = next((a for a in legal if _seat_from_action_id(a.action_id) == target), None)
            if match is not None:
                return match

        # Anything unexpected: a legal action is always better than a crash,
        # because a raised decision stops this match's worker.
        return legal[0]

    def _vote_candidates(self) -> list[int]:
        """Seats a day vote may name: never ourselves, an ally, or someone already cleared."""
        return [seat for seat, _ in self.belief.ranked_suspects()]

    def _unchecked(self) -> list[int]:
        checked = set(self.belief.investigations)
        return [
            seat
            for seat in self.belief.alive
            if seat != self.belief.seat
            and seat not in checked
            and seat not in self.belief.known_wolves
        ]

    def _investigation_target(self) -> int | None:
        """Who to inspect: an unchecked player, most suspicious first. The
        seer's checks are its only private information, so a wasted one is
        expensive."""
        unchecked = self._unchecked()
        if not unchecked:
            return None
        return max(unchecked, key=lambda seat: (self.belief.suspicion(seat), -seat))

    def choose_message(self, state: GameState, context: DecisionContext):
        """Say one useful thing, then vote the window closed.

        Silence loses Werewolf games: the seer's findings die with the seer, and
        a village that never compares notes cannot find the wolves. So the seer
        claims and reports, a wolf covers, and everyone else names a suspect —
        once each, which is why what has been said is tracked on ``self``.

        ``SendMessage`` content is checked against the server's own limits
        (50 words, counted as ``len(content.split())``) before it is returned,
        because an over-long message is refused and costs the window.
        """
        phase = getattr(state, "phase", None)
        self.quota.note_phase(phase)
        self.belief.update(state)
        # The runner only calls this during a window, but speaking on a move
        # turn would spend the quota for nothing, so check rather than assume.
        if phase != "messaging":
            return TERMINATE_MESSAGING

        self._observe_window(state)
        if self.belief.day != self._said_day:
            # A new day is a new table: the seer reports tonight's check, everyone gives a fresh read.
            self._said_day, self._said = self.belief.day, set()

        if not self.belief.eliminated and self.quota.can_send():
            # Say what we will vote: Jev's read of the table so far, so the words and the ballot agree.
            read = jev.choose_seat(
                self._jev,
                self.belief,
                self.belief.chat[-30:],
                self._vote_candidates(),
                jev.VOTE,
                what="speech_target",
            )
            composed = speech.compose(self.belief, already_sent=self._said, vote=read)
            if composed is not None:
                kind, text = composed
                text = speech.phrase(
                    self._client(),
                    kind=kind,
                    intent=text,
                    window=getattr(state, "new_messages", None),
                    our_position=self.belief.seat,
                )
                if msg.fits(text):
                    self._said.add(kind)
                    self.quota.record()
                    return SendMessage(text)
        return TERMINATE_MESSAGING

    def _client(self):
        """The model client, built once per match — only if a key is configured.

        Absent a key this is ``None`` every time, and the deterministic wording
        from ``speech.compose`` is used unchanged, so the agent works with no
        model at all.
        """
        if self._llm is None and not self._llm_checked:
            self._llm_checked = True
            try:
                self._llm = llm.GeminiClient.from_env()
            except Exception:  # noqa: BLE001 - never let this cost a match
                self._llm = None
        return self._llm

    def _observe_window(self, state: GameState) -> None:
        """Fold the open discussion into beliefs: who claimed what, and who
        tried to extract information from us."""
        try:
            window = getattr(state, "new_messages", None)
            self.belief.note_messages(window, our_position=self.belief.seat)
        except Exception:  # noqa: BLE001 - never let bookkeeping kill a match
            return


def create_agent():
    return WerewolfAgent()
