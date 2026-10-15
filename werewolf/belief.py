"""Werewolf belief tracking: turning the public record into suspects.

The engine publishes unusually rich evidence, and the strategy is mostly about
using it honestly (see ``docs/altruagent-skill/werewolf.md``):

- **7 players: 2 wolves, 1 seer, 4 villagers, and no Doctor.** Every night kill
  is therefore evidence, and night 1 is peaceful by construction — day 1 opens
  with all seven alive and exactly one piece of private information in play.
- **``game_state.dead`` publishes the true role of every death.** This is the
  richest source in the game: a night-killed seer means the wolves knew
  something, a villager eliminated by vote means the table was wrong, and the
  number of dead wolves tells you exactly how many remain.
- **``vote_history`` is published only once a day resolves**, and it records who
  voted for whom. Votes *against* eliminating a confirmed wolf, or *for*
  eliminating a confirmed villager, are the strongest behavioural tells there
  are.

Everything is keyed by **seat number**, because that is what legal actions and
message senders use. ``game_state`` publishes display names, so names are
translated through the state's ``players`` map on the way in.

Scope: this module produces *beliefs* — suspicion, and what is known for
certain. Choosing what to do with them is the policy's job, because it depends
on our own role: a villager votes the most suspicious player, while a wolf votes
someone who is not an ally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from werewolf import messages as msg

WOLVES_PER_TABLE = 2
TABLE_SIZE = 7

# Suspicion weights. Kept as named constants so they can be tuned against
# recorded games rather than buried in expressions.
W_VOTED_FOR_CONFIRMED_VILLAGER = 2.0
W_VOTED_AGAINST_CONFIRMED_WOLF = 3.0
W_ABSTAINED_WHILE_WOLF_ELIMINATED = 1.5
W_FALSE_SEER_CLAIM = 5.0
W_INJECTION_ATTEMPT = 1.0
W_ACCUSED_BY_DEAD_SEER = 4.0
W_ATTACKED_CLEAN_VILLAGER = 1.5
# A living seer claim is unverified, but it is the only hard information the table has. A lone claimant's
# accusation counts; two claimants mean one of them is a wolf lying, so both are suspect and neither is believed.
W_ACCUSED_BY_LONE_CLAIMANT = 3.0
W_LONE_CLAIMANT = -1.5
W_RIVAL_CLAIMANT = 2.0
W_SILENT = 0.5  # said nothing while the table talked: a small tell, and it breaks day-1 ties off Player0
# Applied once for anything we know is not a wolf — a published role, our own
# investigation, or (for the seer) ourselves. It must not be summed per source:
# an investigated villager is one fact, not two.
W_KNOWN_GOOD = -5.0
KNOWN_WOLF_SCORE = 100.0

# -- reading the state ------------------------------------------------------

_ROLE = re.compile(r"Your role:\s*([A-Za-z]+)")
_ALLY = re.compile(r"Your fellow wolves?:\s*(.+)")
_INVESTIGATION = re.compile(r"(Player\d+)\s+is\s+(wolf|villager)", re.IGNORECASE)
_SEAT_LABEL = re.compile(r"Player(\d+)", re.IGNORECASE)
_GAME_OVER = re.compile(r"Game over\s*[—\-]\s*(\w+)\s+win", re.IGNORECASE)
_SEER_CLAIM = re.compile(r"\b(i\s*am|i'?m|im)\s+(the\s+)?seer\b", re.IGNORECASE)


def observation_text(state: Any) -> str:
    """Werewolf's observation is prose, carried as the raw string."""
    raw = getattr(state, "raw", None)
    if isinstance(raw, dict):
        value = raw.get("observation")
        if isinstance(value, str):
            return value
    value = getattr(state, "observation", None)
    return value if isinstance(value, str) else ""


def raw_of(state: Any) -> dict:
    raw = getattr(state, "raw", None)
    return raw if isinstance(raw, dict) else {}


def label_to_seat(label: Any) -> int | None:
    """``"Player4"`` → ``4``."""
    if not isinstance(label, str):
        return None
    match = _SEAT_LABEL.search(label)
    return int(match.group(1)) if match else None


def role_from_text(text: str) -> str | None:
    match = _ROLE.search(text or "")
    return match.group(1).upper() if match else None


def allies_from_text(text: str) -> tuple[int, ...]:
    match = _ALLY.search(text or "")
    if not match:
        return ()
    return tuple(
        seat
        for seat in (label_to_seat(part) for part in match.group(1).split(","))
        if seat is not None
    )


def investigations_from_text(text: str) -> dict[int, str]:
    """The seer's own accumulated checks. Private: no one else can see them."""
    return {
        int(_SEAT_LABEL.search(seat).group(1)): verdict.lower()
        for seat, verdict in _INVESTIGATION.findall(text or "")
    }


def game_over_winner(text: str) -> str | None:
    match = _GAME_OVER.search(text or "")
    return match.group(1).lower() if match else None


def players(raw: dict) -> list[dict]:
    value = raw.get("players")
    return [p for p in value if isinstance(p, dict)] if isinstance(value, list) else []


def seat_map(raw: dict) -> dict[int, str]:
    """``{seat: display name}``."""
    out: dict[int, str] = {}
    for player in players(raw):
        position = player.get("position")
        name = player.get("name")
        if isinstance(position, int) and not isinstance(position, bool) and isinstance(name, str):
            out[position] = name
    return out


def game_state(raw: dict) -> dict:
    value = raw.get("game_state")
    return value if isinstance(value, dict) else {}


def dead_block(raw: dict) -> list[dict]:
    value = game_state(raw).get("dead")
    return [d for d in value if isinstance(d, dict)] if isinstance(value, list) else []


def vote_history(raw: dict) -> list[dict]:
    value = game_state(raw).get("vote_history")
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def day_of(raw: dict) -> int | None:
    value = game_state(raw).get("day")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def engine_phase(raw: dict) -> str | None:
    """The engine sub-phase: ``night_wolf``, ``night_seer`` or ``day_vote``.
    Distinct from the platform's top-level ``phase``."""
    value = game_state(raw).get("phase")
    return value if isinstance(value, str) else None


def you_are_eliminated(state: Any) -> bool:
    return bool(raw_of(state).get("eliminated"))


# -- belief -----------------------------------------------------------------


@dataclass
class Belief:
    """What we know and suspect, rebuilt from each state.

    ``update`` is idempotent: it recomputes everything from the state rather
    than accumulating, so a re-read or a replayed state cannot double-count.
    The one exception is messages, which arrive as a window and are folded in
    as they are seen.
    """

    seat: int | None = None
    role: str | None = None
    allies: set[int] = field(default_factory=set)
    investigations: dict[int, str] = field(default_factory=dict)
    seats: dict[int, str] = field(default_factory=dict)
    alive: tuple[int, ...] = ()
    dead_roles: dict[int, str] = field(default_factory=dict)
    dead_causes: dict[int, str] = field(default_factory=dict)
    vote_history: list[dict] = field(default_factory=list)
    seer_claims: set[int] = field(default_factory=set)
    accused_by_seer: set[int] = field(default_factory=set)
    injection_seats: set[int] = field(default_factory=set)
    accusations: dict[int, set[int]] = field(default_factory=dict)  # sender -> seats they called wolves
    speakers: set[int] = field(default_factory=set)
    chat: list = field(default_factory=list)  # every message seen so far, once each, oldest first
    _chat_seen: set = field(default_factory=set)
    day: int | None = None
    winner: str | None = None
    eliminated: bool = False

    # -- ingest ---------------------------------------------------------

    def update(self, state: Any) -> Belief:
        raw = raw_of(state)
        text = observation_text(state)

        self.seat = _int_or_none(raw.get("your_position"))
        self.role = role_from_text(text)
        self.allies = set(allies_from_text(text))
        self.investigations = investigations_from_text(text)
        self.seats = seat_map(raw)
        self.winner = game_over_winner(text)
        self.eliminated = you_are_eliminated(state)

        by_name = {name: seat for seat, name in self.seats.items()}
        alive_names = game_state(raw).get("alive")
        if isinstance(alive_names, list):
            self.alive = tuple(
                seat
                for seat in (by_name.get(n) for n in alive_names if isinstance(n, str))
                if seat is not None
            )

        self.dead_roles = {}
        self.dead_causes = {}
        for entry in dead_block(raw):
            seat = by_name.get(entry.get("player"))
            role = entry.get("role")
            cause = entry.get("cause")
            if seat is not None and isinstance(role, str):
                self.dead_roles[seat] = role.upper()
            if seat is not None and isinstance(cause, str):
                self.dead_causes[seat] = cause
        self.vote_history = vote_history(raw)
        self.day = day_of(raw)
        # Only the published seer's own accusations are evidence: anyone can type "PlayerN is wolf".
        self.accused_by_seer = {
            target
            for seat, role in self.dead_roles.items()
            if role == "SEER"
            for target in self.accusations.get(seat, ())
        }
        return self

    def note_messages(self, window: Any, *, our_position: int | None = None) -> Belief:
        """Fold a discussion window into beliefs.

        Two things come out of chat: who *claimed* to be the seer (a claim the
        dead role list can later prove false), and who tried to inject us —
        which is itself a tell worth weighing.
        """
        seat = self.seat if our_position is None else our_position
        for message in window or []:
            sender = getattr(message, "sender", None)
            index = getattr(message, "index", None)
            if (
                getattr(message, "type", "chat") != "terminate"
                and (index is None or index not in self._chat_seen)
            ):
                # The vote window only carries what is new, so the day's whole conversation is kept here.
                self.chat.append(message)
                if index is not None:
                    self._chat_seen.add(index)
            if not isinstance(sender, int) or isinstance(sender, bool) or sender == seat:
                continue
            if getattr(message, "type", "chat") == "terminate":
                continue
            self.speakers.add(sender)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                if _SEER_CLAIM.search(content):
                    self.seer_claims.add(sender)
                for t_str, verdict in _INVESTIGATION.findall(content):
                    t_num = label_to_seat(t_str)
                    if t_num is not None and verdict.lower() == "wolf" and t_num != sender:
                        self.accusations.setdefault(sender, set()).add(t_num)

        for flagged in msg.flag_messages(window, our_position=seat):
            sender = flagged.get("sender")
            if isinstance(sender, int) and not isinstance(sender, bool):
                self.injection_seats.add(sender)
        return self

    # -- certainties ----------------------------------------------------

    @property
    def known_wolves(self) -> set[int]:
        """Seats proved to be wolves: a dead wolf, or our own investigation."""
        found = {seat for seat, role in self.dead_roles.items() if role == "WOLF"}
        if self.role == "SEER":
            found |= {seat for seat, verdict in self.investigations.items() if verdict == "wolf"}
        return found

    @property
    def known_villagers(self) -> set[int]:
        found = {seat for seat, role in self.dead_roles.items() if role == "VILLAGER"}
        if self.role == "SEER":
            found |= {
                seat for seat, verdict in self.investigations.items() if verdict == "villager"
            }
        if self.role == "SEER" and self.seat is not None:
            found.add(self.seat)
        for seat, role in list(self.dead_roles.items()):
            if role == "SEER":
                found.add(seat)
        return found

    @property
    def wolves_remaining(self) -> int:
        """Exactly how many wolves are still alive."""
        return max(0, WOLVES_PER_TABLE - len(self.known_wolves))

    @property
    def seer_is_dead(self) -> bool:
        return any(role == "SEER" for role in self.dead_roles.values())

    @property
    def living_claimants(self) -> set[int]:
        """Living players claiming the seer that nothing has disproved yet."""
        return {
            seat
            for seat in self.seer_claims
            if seat not in self.dead_roles and seat != self.seat and (not self.alive or seat in self.alive)
        }

    @property
    def false_seer_claimants(self) -> set[int]:
        """Living players claiming the seer after the real seer's role was
        published. The dead role list is truth, so those claims cannot be
        honest — this is a hard deduction, not a hunch."""
        if self.role == "SEER":
            return self.living_claimants  # we are the seer: every other claim is a wolf's
        if not self.seer_is_dead:
            return set()
        return {seat for seat in self.seer_claims if seat not in self.dead_roles}

    @property
    def night_killed(self) -> set[int]:
        """Seats the wolves chose. With no Doctor every one is a deliberate
        kill, so who the wolves wanted is itself information."""
        return {seat for seat, cause in self.dead_causes.items() if cause == "night_kill"}

    # -- suspicion ------------------------------------------------------

    def suspicion(self, seat: int) -> float:
        """How much this seat looks like a wolf. Higher is worse."""
        if seat in self.known_wolves:
            return KNOWN_WOLF_SCORE
        score = 0.0

        if seat in self.known_villagers:
            score += W_KNOWN_GOOD
        if seat in self.false_seer_claimants:
            score += W_FALSE_SEER_CLAIM
        if seat in self.injection_seats:
            score += W_INJECTION_ATTEMPT

        # If a confirmed dead seer pointed out this seat before dying
        if self.seer_is_dead and seat in self.accused_by_seer:
            score += W_ACCUSED_BY_DEAD_SEER

        claimants = self.living_claimants
        if not self.seer_is_dead and self.role != "SEER":
            if len(claimants) == 1:
                (only,) = claimants
                if seat == only:
                    score += W_LONE_CLAIMANT
                elif seat in self.accusations.get(only, ()):
                    score += W_ACCUSED_BY_LONE_CLAIMANT
            elif len(claimants) > 1 and seat in claimants:
                score += W_RIVAL_CLAIMANT
        if (
            seat != self.seat
            and seat not in self.speakers
            and len(self.speakers) >= 2
            and seat in self.alive
        ):
            score += W_SILENT

        by_name = {name: s for s, name in self.seats.items()}
        for record in self.vote_history:
            votes = record.get("votes")
            if not isinstance(votes, dict):
                continue
            eliminated_name = record.get("eliminated")
            eliminated = by_name.get(eliminated_name) if isinstance(eliminated_name, str) else None
            # Find this seat's own vote.
            own = next(
                (target for voter, target in votes.items() if by_name.get(voter) == seat),
                None,
            )
            if own is None:
                continue
            target_seat = by_name.get(own) if isinstance(own, str) else None

            if eliminated is not None and eliminated in self.known_wolves:
                # A wolf went down. Anyone who did not vote for them was
                # helping them survive — voting a known villager instead is
                # the sharper tell, abstaining (or voting an unknown) the
                # milder one.
                if target_seat != eliminated:
                    if target_seat is not None and target_seat in self.known_villagers:
                        score += W_VOTED_AGAINST_CONFIRMED_WOLF
                    else:
                        score += W_ABSTAINED_WHILE_WOLF_ELIMINATED
            if (
                target_seat is not None
                and target_seat in self.known_villagers
                and eliminated == target_seat
            ):
                # Voted a known villager out.
                score += W_VOTED_FOR_CONFIRMED_VILLAGER
        return score

    def ranked_suspects(self) -> list[tuple[int, float]]:
        """Living seats, most suspicious first. Excludes ourselves, anyone
        already dead, and anyone we know to be a villager."""
        candidates = [
            seat
            for seat in self.alive
            if seat != self.seat
            and seat not in self.known_villagers
            and seat not in (self.allies if self.role == "WOLF" else ())
        ]
        return sorted(
            ((seat, self.suspicion(seat)) for seat in candidates),
            key=lambda pair: (-pair[1], pair[0]),
        )

    def best_vote(self) -> int | None:
        """Who to vote out today — which depends on which side we are on.

        A villager or the seer wants a wolf: a known one first, otherwise the
        most suspicious living player. A wolf wants anyone but an ally, and
        prefers a player the table already suspects, because a vote nobody
        follows is a wasted vote.
        """
        if self.role == "WOLF":
            candidates = [
                seat
                for seat in self.alive
                if seat != self.seat and seat not in self.allies and seat not in self.known_wolves
            ]
            if not candidates:
                return None
            # Most suspicious first; ties to the lower seat, deterministically.
            return max(candidates, key=lambda seat: (self.suspicion(seat), -seat))

        wolves = sorted(seat for seat in self.known_wolves if seat in self.alive)
        if wolves:
            return wolves[0]
        ranked = self.ranked_suspects()
        return ranked[0][0] if ranked else None

    def best_kill(self) -> int | None:
        """The seat a wolf should kill: never an ally, and preferably the seer
        or whoever is driving the table.

        Priority:
        1. If someone claimed Seer and is still alive, kill them immediately
           to silence their investigations.
        2. Otherwise, kill the least-suspicious living player (the most trusted).
        """
        candidates = [
            seat
            for seat in self.alive
            if seat != self.seat and seat not in self.allies and seat not in self.known_wolves
        ]
        if not candidates:
            return None

        # Priority 1: Living Seer claimants
        living_claimants = [s for s in candidates if s in self.seer_claims]
        if living_claimants:
            return living_claimants[0]

        # Priority 2: Least suspected (most trusted by village)
        return min(candidates, key=lambda seat: (self.suspicion(seat), seat))


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
