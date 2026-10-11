"""Automated proof that the production gameplay path (run_match / run_game /
python -m agent) never constructs or calls the REST GameSession, for any
reason — not merely documented, but testable (per the Milestone 6 migration
requirement). This is the strongest form of the guarantee: it patches
GameSession.__init__ itself to fail loudly if ANY code path along the way
ever tries to build one, rather than just checking which Match method was
called (see test_runner.py::test_run_match_never_constructs_a_rest_gamesession
for that narrower check).
"""

from __future__ import annotations

import pytest

from altruagent.game import GameSession
from altruagent.models import DecisionContext, GameState, Match
from altruagent.runner import RESIGN, run_game, run_match


def make_mcp_state(**overrides) -> GameState:
    payload = {
        "session_id": "session-1",
        "game_type": "tic_tac_toe",
        "status": "in_progress",
        "state_version": 0,
        "observation": "...",
        "phase": "moving",
        "messaging_enabled": False,
        "terminated_messaging": [],
        "new_messages": [],
        "current_actor": {"agent_id": "agent-1", "position": 0},
        "is_current_actor": True,
        "is_terminal": False,
    }
    payload.update(overrides)
    return GameState.from_mcp_state(payload)


class FakeMCPGameSession:
    def __init__(self) -> None:
        self._states = [make_mcp_state(), make_mcp_state(is_terminal=True, is_current_actor=False)]

    def get_state(self):
        return self._states[0] if len(self._states) == 1 else self._states.pop(0)

    def get_legal_actions(self):
        return {"session_id": "session-1", "state_version": 0, "actions": []}

    def resign(self):
        return {"session_id": "session-1", "is_terminal": True, "status": "completed", "returns": {}, "termination_reason": "resignation"}

    def get_result(self):
        return {"session_id": "session-1", "is_terminal": True, "status": "completed", "returns": {}, "termination_reason": None}


@pytest.fixture
def forbid_rest_gamesession(monkeypatch):
    """Patch GameSession.__init__ to fail loudly if constructed — proving no
    code path in this test reaches it, not just that we didn't call it
    ourselves.
    """

    def _forbidden(self, *args, **kwargs):
        raise AssertionError(
            "GameSession (REST) was constructed during a production run_game/"
            "run_match call — this is a hard regression: MCP must be the only "
            "gameplay path python -m agent/run_match ever uses."
        )

    monkeypatch.setattr(GameSession, "__init__", _forbidden)
    yield


def test_run_game_never_constructs_rest_gamesession(forbid_rest_gamesession):
    game = FakeMCPGameSession()
    context = DecisionContext(session_id="session-1", tournament_id=None, game_type="tic_tac_toe", agent_id="agent-1")

    result = run_game(game, context, lambda s, c: RESIGN, sleep=lambda s: None)

    assert result.is_terminal is True


def test_run_match_never_constructs_rest_gamesession(forbid_rest_gamesession):
    match = Match.from_dict({"session_id": "session-1", "status": "in_progress", "game_type": "tic_tac_toe"})
    match.game = lambda: FakeMCPGameSession()

    result = run_match(match, "agent-1", lambda s, c: RESIGN, sleep=lambda s: None)

    assert result.is_terminal is True
