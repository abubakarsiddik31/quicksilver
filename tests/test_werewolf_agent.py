"""Tests for the Werewolf policy: belief-driven moves, and safety.

The point of these is that the agent acts on what it knows rather than on
whatever the server happens to list first — while never raising, because a
raised decision kills that match's worker process.
"""

from __future__ import annotations

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, SendMessage
from werewolf import messages as msg
from werewolf import speech
from werewolf.agent import WerewolfAgent

NAMES = ["alice", "bob", "carl", "dave", "erin", "fran", "gus"]


def werewolf_state(
    *,
    observation,
    your_position=3,
    engine_phase="day_vote",
    dead=(),
    alive=None,
    vote_history=(),
    seats=(0, 1, 2, 3, 4, 5, 6),
    phase="moving",
):
    payload = {
        "session_id": "s1",
        "game_type": "werewolf",
        "status": "in_progress",
        "state_version": 1,
        "observation": observation,
        "phase": phase,
        "messaging_enabled": True,
        "terminated_messaging": [],
        "new_messages": [],
        "legal_actions": {
            "actions": [
                {
                    "action_id": str(seat),
                    "label": f"Vote to eliminate Player{seat}",
                    "input": {"action_id": str(seat)},
                }
                for seat in seats
            ]
        },
        "current_actor": {"agent_id": "us", "position": your_position},
        "is_current_actor": True,
        "is_terminal": False,
        "your_position": your_position,
        "players": [{"position": i, "name": n} for i, n in enumerate(NAMES)],
        "game_state": {
            "day": 2,
            "phase": engine_phase,
            "alive": list(NAMES if alive is None else alive),
            "dead": list(dead),
            "vote_history": list(vote_history),
        },
    }
    return GameState.from_mcp_state(payload)


def context() -> DecisionContext:
    return DecisionContext(
        session_id="s1", tournament_id=None, game_type="werewolf", agent_id="us", seat_position=3
    )


VILLAGER = "=== Werewolf ===\nYour role: VILLAGER\nYou do not know anyone else's role."
WOLF = "=== Werewolf ===\nYour role: WOLF\nYour fellow wolves: Player6"
SEER = "=== Werewolf ===\nYour role: SEER\nYour investigations: Player1 is wolf"


def action_ids(state: GameState) -> list[str]:
    return [a.action_id for a in state.legal_actions]


# -- voting -----------------------------------------------------------------


def test_a_villager_votes_the_most_suspicious_seat_not_the_first_listed():
    state = werewolf_state(observation=VILLAGER, your_position=3)
    # Reorder so the first *offered* action is deliberately not the answer.
    state.legal_actions = sorted(
        state.legal_actions, key=lambda a: (a.action_id == "0", a.action_id)
    )
    assert action_ids(state)[0] != "0"

    agent = WerewolfAgent()
    chosen = agent.choose_action(state, context())
    assert chosen.action_id == "0"  # lowest-seat suspect, not the first listed


def test_a_villager_follows_the_evidence_when_it_points_elsewhere():
    state = werewolf_state(
        observation=VILLAGER,
        your_position=3,
        dead=[{"day": 1, "player": "carl", "role": "villager", "cause": "vote"}],
        vote_history=[{"day": 1, "votes": {"gus": "carl", "bob": "abstain"}, "eliminated": "carl"}],
    )
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id == "6"  # gus voted the villager out


def test_a_wolf_never_votes_its_own_ally():
    state = werewolf_state(observation=WOLF, your_position=3)  # ally is seat 6
    state.legal_actions = [a for a in state.legal_actions if a.action_id != "0"]
    agent = WerewolfAgent()
    chosen = agent.choose_action(state, context())
    assert chosen.action_id not in ("3", "6")


def test_a_known_wolf_is_voted_out_first():
    state = werewolf_state(
        observation=VILLAGER,
        your_position=3,
        dead=[{"day": 1, "player": "gus", "role": "wolf", "cause": "vote"}],
    )
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id == "6"


# -- night actions ----------------------------------------------------------


def test_a_wolf_targets_a_non_ally_at_night():
    state = werewolf_state(observation=WOLF, your_position=3, engine_phase="night_wolf")
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id not in ("3", "6")


def test_the_seer_investigates_an_unchecked_player():
    state = werewolf_state(observation=SEER, your_position=3, engine_phase="night_seer")
    agent = WerewolfAgent()
    chosen = agent.choose_action(state, context())
    # Seat 1 is already known to be a wolf, so re-checking it wastes the night.
    assert chosen.action_id != "1"
    assert chosen.action_id != "3"


def test_the_seer_prefers_the_most_suspicious_unchecked_player():
    state = werewolf_state(observation=SEER, your_position=3, engine_phase="night_seer")
    agent = WerewolfAgent()
    # Someone tried to extract our role: that is worth investigating.
    agent.belief.injection_seats.add(5)
    chosen = agent.choose_action(state, context())
    assert chosen.action_id == "5"


# -- safety -----------------------------------------------------------------


def test_an_unknown_engine_phase_falls_back_to_a_legal_action():
    state = werewolf_state(observation=VILLAGER, engine_phase=None)
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id in action_ids(state)


def test_no_legal_actions_returns_none_rather_than_raising():
    state = werewolf_state(observation=VILLAGER)
    state.legal_actions = []
    assert WerewolfAgent().choose_action(state, context()) is None


def test_a_missing_target_falls_back_to_a_legal_action():
    """If belief names a seat the server did not offer, we must still act."""
    state = werewolf_state(
        observation=VILLAGER,
        your_position=3,
        dead=[{"day": 1, "player": "gus", "role": "wolf", "cause": "vote"}],
    )
    state.legal_actions = [a for a in state.legal_actions if a.action_id != "6"]
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id in action_ids(state)


def test_a_junk_state_does_not_raise():
    state = werewolf_state(observation="")
    state.raw = {"observation": None}
    chosen = WerewolfAgent().choose_action(state, context())
    assert chosen.action_id in action_ids(state)


def test_the_agent_tracks_belief_across_calls():
    agent = WerewolfAgent()
    state = werewolf_state(observation=VILLAGER, your_position=3)
    agent.choose_action(state, context())
    assert agent.belief.role == "VILLAGER"
    assert agent.belief.seat == 3


# -- messaging --------------------------------------------------------------


def test_the_agent_speaks_once_and_then_terminates():
    """Silence loses Werewolf games, so the agent now says one useful thing —
    and then closes its part of the window rather than repeating itself."""
    agent = WerewolfAgent()
    state = werewolf_state(observation=VILLAGER, phase="messaging")

    first = agent.choose_message(state, context())
    assert isinstance(first, SendMessage)
    assert msg.word_count(first.content) <= msg.MAX_WORDS
    assert msg.fits(first.content) is True

    assert agent.choose_message(state, context()) is TERMINATE_MESSAGING


def test_the_agent_stays_silent_when_it_has_nothing_concrete_to_say():
    """No readable board means no target, and a generic line tells the table
    nothing while drawing attention."""
    agent = WerewolfAgent()
    state = werewolf_state(observation="", phase="messaging")
    state.raw = {"observation": "", "phase": "messaging"}
    state.new_messages = []
    assert agent.choose_message(state, context()) is TERMINATE_MESSAGING


def test_the_agent_never_speaks_outside_the_discussion_window():
    agent = WerewolfAgent()
    moving = werewolf_state(observation=VILLAGER, phase="moving")
    assert agent.choose_message(moving, context()) is TERMINATE_MESSAGING
    assert agent.quota.used == 0


def test_the_quota_resets_between_windows():
    agent = WerewolfAgent()
    messaging = werewolf_state(observation=VILLAGER, phase="messaging")
    moving = werewolf_state(observation=VILLAGER, phase="moving")
    agent.quota.record()
    agent.quota.record()
    assert agent.quota.used == 2
    agent.choose_message(moving, context())
    assert agent.quota.used == 0


def test_the_agent_can_render_its_window_as_untrusted_data():
    """The pieces the speech policy needs are already in place."""
    state = werewolf_state(observation=VILLAGER, phase="messaging")
    rendered = msg.render_untrusted([], our_position=3)
    assert msg.FENCE_BEGIN in rendered and msg.FENCE_END in rendered
    assert state.phase == "messaging"


# -- optional model phrasing ------------------------------------------------


class StubClient:
    """Stands in for the Gemini client, with no network."""

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, **kwargs):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def with_client(agent, reply):
    agent._llm = StubClient(reply)
    agent._llm_checked = True
    return agent


def test_the_model_only_rewords_a_claim_the_policy_already_made():
    agent = with_client(WerewolfAgent(), "Player1 is a wolf, I would vote them out.")
    state = werewolf_state(observation=VILLAGER, phase="messaging")
    sent = agent.choose_message(state, context())
    assert isinstance(sent, SendMessage)
    assert sent.content == "Player1 is a wolf, I would vote them out."
    # The claim itself is still recorded, so it is not made twice.
    assert speech.ACCUSE in agent._said


def test_a_failing_model_falls_back_to_the_deterministic_wording():
    for reply in (None, "", RuntimeError("boom")):
        agent = with_client(WerewolfAgent(), reply)
        state = werewolf_state(observation=VILLAGER, phase="messaging")
        sent = agent.choose_message(state, context())
        assert isinstance(sent, SendMessage), reply
        # your_position is 3 in this file's state helper, so the lowest-seat
        # suspect is Player0.
        assert "Vote Player0" in sent.content
        assert msg.fits(sent.content) is True


def test_the_agent_never_constructs_a_client_without_a_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    agent = WerewolfAgent()
    assert agent._client() is None
    # And it still plays: the deterministic path needs no model at all.
    state = werewolf_state(observation=VILLAGER, phase="messaging")
    assert isinstance(agent.choose_message(state, context()), SendMessage)
