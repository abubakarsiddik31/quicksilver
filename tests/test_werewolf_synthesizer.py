"""Tests verifying Rule-Scored LLM Synthesis in Werewolf."""

from __future__ import annotations

from werewolf import belief as belief_mod
from werewolf import synthesizer


def mock_belief():
    b = belief_mod.Belief()
    b.seat = 3
    b.role = "VILLAGER"
    b.alive = (0, 1, 2, 3, 4, 5, 6)
    b.dead_roles = {0: "WOLF"}
    b.dead_causes = {0: "voted_out"}
    b.seats = {i: f"Player{i}" for i in range(7)}
    b.seer_claims = set()
    b.injection_seats = set()
    return b


def test_format_forensic_scoreboard():
    b = mock_belief()
    summary = synthesizer.format_forensic_scoreboard(b)
    assert "Our Seat: Player3 (Role: VILLAGER)" in summary
    assert "Player0: WOLF" in summary


def test_synthesize_day_vote_extracts_verdict():
    b = mock_belief()
    b.alive = (1, 2, 3, 4)
    # Player 1 and Player 4 are suspects
    b.vote_history = [
        {"votes": {"Player1": "Player2", "Player4": "Player2"}, "eliminated": "Player0"}
    ]

    class MockClient:
        def generate(self, prompt, **kwargs):
            return "VERDICT: Player1 - They actively defended Player 0 in the previous round."

    vote = synthesizer.synthesize_day_vote(MockClient(), b, window=[])
    assert vote == 1


def test_synthesize_day_vote_fails_gracefully_to_none():
    b = mock_belief()
    b.alive = (1, 2, 3)

    class CrashingClient:
        def generate(self, prompt, **kwargs):
            raise TimeoutError("timeout")

    vote = synthesizer.synthesize_day_vote(CrashingClient(), b, window=[])
    assert vote is None


def test_synthesize_night_kill_extracts_target():
    b = mock_belief()
    b.role = "WOLF"
    b.allies = {1}
    b.alive = (1, 2, 3, 4)

    class MockClient:
        def generate(self, prompt, **kwargs):
            return "KILL: Player2 - They are acting as the village leader."

    target = synthesizer.synthesize_night_kill(MockClient(), b, window=[])
    assert target == 2
