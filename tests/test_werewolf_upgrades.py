"""Tests verifying the tactical deduction upgrades in Werewolf belief and speech."""

from __future__ import annotations

from types import SimpleNamespace

from altruagent import Message
from werewolf import belief, speech


def make_state(*, observation, your_position=3, dead=(), vote_history=()):
    raw = {
        "your_position": your_position,
        "players": [{"position": i, "name": f"Player{i}"} for i in range(7)],
        "game_state": {
            "day": 2,
            "phase": "day_vote",
            "alive": [
                f"Player{i}"
                for i in range(7)
                if f"Player{i}" not in [d.get("player") for d in dead]
            ],
            "dead": list(dead),
            "vote_history": list(vote_history),
        },
        "eliminated": False,
        "observation": observation,
    }
    return SimpleNamespace(raw=raw, observation=observation, phase="moving")


def note(sender, content):
    return Message(
        index=0,
        sender=sender,
        recipients=[],
        content=content,
        type="chat",
        sent_at="",
        move_index=0,
    )


def test_dead_seer_accusation_boosts_suspicion():
    """If Player 1 claimed Seer and said 'Player 0 is wolf', and Player 1 is now dead
    with confirmed role SEER, Player 0 must receive a heavy suspicion penalty."""
    b = belief.Belief()
    # Day 1: Player 1 claimed seer and named Player 0 as wolf
    b.note_messages([note(1, "I am the seer. Player0 is wolf.")], our_position=3)

    # Night 1: Player 1 killed by wolves, revealed as SEER
    state = make_state(
        observation="=== Werewolf ===\nYour role: VILLAGER\nYou do not know anyone else's role.",
        your_position=3,
        dead=[{"player": "Player1", "role": "seer", "cause": "night_kill"}],
    )
    b.update(state)

    assert b.seer_is_dead is True
    assert 0 in b.accused_by_seer
    # Player 0 should be top suspect due to W_ACCUSED_BY_DEAD_SEER
    assert b.best_vote() == 0


def test_speech_includes_dead_seer_reasoning():
    """When accusing someone pointed out by a dead seer, speech includes that reason."""
    b = belief.Belief()
    b.seat = 3
    b.role = "VILLAGER"
    b.accused_by_seer = {0}
    b.seats = {i: f"Player{i}" for i in range(7)}

    text = speech.accusation_text(b, vote=0)
    assert "The real seer identified them" in text


def test_a_wolf_typing_player_n_is_wolf_does_not_frame_n_after_the_seer_dies():
    b = belief.Belief()
    b.note_messages([note(1, "Player3 is wolf, trust me")], our_position=4)
    b.update(
        make_state(
            observation="Your role: VILLAGER",
            your_position=4,
            dead=[{"player": "Player6", "role": "seer", "cause": "night_kill"}],
        )
    )
    assert 3 not in b.accused_by_seer


def test_speech_resets_each_day_so_the_seer_reports_every_morning():
    from werewolf.agent import WerewolfAgent

    agent = WerewolfAgent()
    agent._said, agent._said_day = {speech.SEER_CLAIM}, 1
    agent.belief.day = 2
    state = SimpleNamespace(phase="messaging", raw={}, new_messages=[], observation="")
    agent.belief.update = lambda s: agent.belief  # keep the day we set
    agent.choose_message(state, None)
    assert agent._said_day == 2 and speech.SEER_CLAIM not in agent._said


def test_jev_decides_the_vote_whatever_its_confidence_and_only_among_legal_seats():
    from common.jev import Answer
    from werewolf import jev

    class FakeClient:
        def __init__(self, choice):
            self.choice = choice

        def ask(self, state, questions, timeout=None):
            assert "chat_untrusted" in state and set(questions["pick"]["criteria"]) == {"Player1", "Player2"}
            return {"pick": Answer("choice", choice=self.choice, confidence=0.2)}

    b = belief.Belief()
    b.seat, b.role, b.alive = 4, "VILLAGER", (1, 2, 4)
    assert jev.choose_seat(FakeClient("Player2"), b, [], [1, 2], jev.VOTE, what="t") == 2
    assert jev.choose_seat(FakeClient("Player9"), b, [], [1, 2], jev.VOTE, what="t") is None
    assert jev.choose_seat(None, b, [], [1, 2], jev.VOTE, what="t") is None


def test_the_wolf_jury_never_sees_its_ally():
    from werewolf import synthesizer

    seen = {}

    class Client:
        def generate(self, prompt, **kw):
            seen["prompt"], seen["system"] = prompt, kw["system"]
            return "VERDICT: Player1 - ok"

    b = belief.Belief()
    b.seat, b.role, b.allies, b.alive = 4, "WOLF", {0}, tuple(range(7))
    b.seats = {i: f"Player{i}" for i in range(7)}
    assert synthesizer.synthesize_day_vote(Client(), b, []) == 1
    assert "Eligible Top Suspects: Player1" in seen["prompt"] and "Player0" not in seen["prompt"].split("Eligible")[1]
    assert seen["system"] == synthesizer.WOLF_VOTE_SYSTEM
