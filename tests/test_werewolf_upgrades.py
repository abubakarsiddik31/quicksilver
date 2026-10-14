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
