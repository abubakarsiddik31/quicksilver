"""Unit tests for MCPGameSession. call_tool()'s own JSON-RPC/auth/error-
unwrapping mechanics (built on the official `mcp` SDK client) are fully
covered in tests/test_mcp_transport.py — these tests only check that
MCPGameSession's eight methods build the right (tool_name, arguments) pairs
and parse the right response shape, by monkeypatching
`altruagent.mcp_game.call_tool` directly rather than re-exercising the real
async transport here too.
"""

from __future__ import annotations

import pytest

import altruagent.mcp_game as mcp_game_module
from altruagent.client import AltruAgentClient
from altruagent.game import GameSession
from altruagent.mcp_game import MCPGameSession
from altruagent.mcp_transport import MCPToolError

SESSION_ID = "session-1"
GAME_SERVER_URL = "http://game.example.test"


class FakeClient:
    """Stand-in for AltruAgentClient — MCPGameSession never calls anything
    on it directly (call_tool is monkeypatched below), so this only needs
    to exist as an opaque object to pass through.
    """


def make_session(**overrides) -> MCPGameSession:
    return MCPGameSession(FakeClient(), session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)


def patch_call_tool(monkeypatch, fn):
    monkeypatch.setattr(mcp_game_module, "call_tool", fn)


def test_get_state_calls_get_game_state_and_parses_generic_state(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append((mcp_url, name, arguments))
        return {
            "session_id": SESSION_ID,
            "game_type": "tic_tac_toe",
            "status": "in_progress",
            "state_version": 2,
            "observation": "...",
            "phase": "moving",
            "is_current_actor": True,
            "is_terminal": False,
            "current_actor": {"agent_id": "agent-1", "position": 0},
        }

    patch_call_tool(monkeypatch, fake_call_tool)
    session = make_session()
    state = session.get_state()

    assert calls == [(f"{GAME_SERVER_URL}/mcp", "get_game_state", {"session_id": SESSION_ID})]
    assert state.session_id == SESSION_ID
    assert state.state_version == 2
    assert state.is_current_actor is True
    assert state.legal_actions == []  # none embedded in this payload


def test_get_state_parses_embedded_legal_actions(monkeypatch):
    def fake_call_tool(client, mcp_url, name, arguments):
        return {
            "session_id": SESSION_ID,
            "state_version": 5,
            "is_current_actor": True,
            "legal_actions": {
                "session_id": SESSION_ID,
                "state_version": 5,
                "actions": [{"action_id": "3", "label": "Vote Player3", "input": {}}],
            },
        }

    patch_call_tool(monkeypatch, fake_call_tool)
    state = make_session().get_state()

    assert [a.action_id for a in state.legal_actions] == ["3"]
    assert state.state_version == 5


def test_wait_for_update_sends_cursors_and_parses_state(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append((name, arguments))
        return {"session_id": SESSION_ID, "state_version": 9, "updated": True}

    patch_call_tool(monkeypatch, fake_call_tool)
    state = make_session().wait_for_update(
        since_version=8,
        since_message_seq=4,
        since_is_current_actor=False,
        since_phase="moving",
        timeout_seconds=20.0,
    )

    assert calls == [
        (
            "wait_for_update",
            {
                "session_id": SESSION_ID,
                "since_version": 8,
                "since_message_seq": 4,
                "since_is_current_actor": False,
                "since_phase": "moving",
                "timeout_seconds": 20.0,
            },
        )
    ]
    assert state.state_version == 9
    assert state.raw["updated"] is True


def test_wait_for_update_omits_unset_optional_arguments(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append(arguments)
        return {"session_id": SESSION_ID}

    patch_call_tool(monkeypatch, fake_call_tool)
    make_session().wait_for_update(since_version=1)

    assert calls == [{"session_id": SESSION_ID, "since_version": 1}]


def test_mcp_url_appends_mcp_path_to_normalized_game_server_url(monkeypatch):
    seen_urls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        seen_urls.append(mcp_url)
        return {"session_id": SESSION_ID}

    patch_call_tool(monkeypatch, fake_call_tool)
    session = MCPGameSession(FakeClient(), session_id=SESSION_ID, game_server_url="game.example.test")
    session.get_state()

    assert seen_urls == ["https://game.example.test/mcp"]


def test_get_legal_actions_returns_raw_dict_not_gamestate(monkeypatch):
    def fake_call_tool(client, mcp_url, name, arguments):
        assert name == "get_legal_actions"
        assert arguments == {"session_id": SESSION_ID}
        return {"session_id": SESSION_ID, "state_version": 5, "actions": [{"action_id": "0", "label": "a", "input": {}}]}

    patch_call_tool(monkeypatch, fake_call_tool)
    result = make_session().get_legal_actions()

    assert result["state_version"] == 5
    assert result["actions"][0]["action_id"] == "0"


def test_play_action_with_action_id_sends_action_id_and_state_version(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append((name, arguments))
        return {"accepted": True, "session_id": SESSION_ID, "state_version": 6, "status": "in_progress"}

    patch_call_tool(monkeypatch, fake_call_tool)
    result = make_session().play_action(action_id="0", state_version=5)

    assert calls == [("play_action", {"session_id": SESSION_ID, "state_version": 5, "action_id": "0"})]
    assert result["state_version"] == 6


def test_play_action_with_structured_action_sends_action_not_action_id(monkeypatch):
    team = {"type": "submit_team", "team": ["a", "b", "c", "d", "e", "f"]}
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append(arguments)
        return {"accepted": True, "state_version": 1, "status": "in_progress"}

    patch_call_tool(monkeypatch, fake_call_tool)
    make_session().play_action(action=team, state_version=0)

    assert calls == [{"session_id": SESSION_ID, "state_version": 0, "action": team}]
    assert "action_id" not in calls[0]


def test_send_message_never_sends_state_version(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append((name, arguments))
        return {"accepted": True, "phase": "messaging"}

    patch_call_tool(monkeypatch, fake_call_tool)
    make_session().send_message(message_type="chat", content="hi", recipients=[1])

    assert calls == [
        ("send_message", {"session_id": SESSION_ID, "message_type": "chat", "content": "hi", "recipients": [1]})
    ]
    assert "state_version" not in calls[0][1]


def test_resign_calls_resign_tool(monkeypatch):
    def fake_call_tool(client, mcp_url, name, arguments):
        assert name == "resign"
        assert arguments == {"session_id": SESSION_ID}
        return {
            "session_id": SESSION_ID,
            "is_terminal": True,
            "status": "completed",
            "returns": {"Me": -1.0},
            "your_return": -1.0,
            "termination_reason": "resignation",
        }

    patch_call_tool(monkeypatch, fake_call_tool)
    result = make_session().resign()

    assert result["is_terminal"] is True
    assert result["termination_reason"] == "resignation"


def test_get_result_calls_get_result_tool(monkeypatch):
    def fake_call_tool(client, mcp_url, name, arguments):
        assert name == "get_result"
        return {"session_id": SESSION_ID, "is_terminal": True, "status": "completed", "returns": {"Me": 1.0}}

    patch_call_tool(monkeypatch, fake_call_tool)
    result = make_session().get_result()

    assert result["returns"] == {"Me": 1.0}


def test_get_messages_sends_since_cursor(monkeypatch):
    calls = []

    def fake_call_tool(client, mcp_url, name, arguments):
        calls.append((name, arguments))
        return {"messages": [], "phase": "messaging"}

    patch_call_tool(monkeypatch, fake_call_tool)
    make_session().get_messages(since=3)

    assert calls == [("get_messages", {"session_id": SESSION_ID, "since": 3})]


def test_stale_state_error_propagates_as_mcp_tool_error(monkeypatch):
    def fake_call_tool(client, mcp_url, name, arguments):
        raise MCPToolError("stale", status_code=None, error_code="STALE_STATE")

    patch_call_tool(monkeypatch, fake_call_tool)

    with pytest.raises(MCPToolError) as exc_info:
        make_session().play_action(action_id="0", state_version=1)

    assert exc_info.value.error_code == "STALE_STATE"


def test_mcp_game_and_rest_game_from_same_client_are_independent_classes():
    session = make_session()
    real_client = AltruAgentClient(control_url="https://example.test", api_key="k", load_env_file=False)
    rest_session = real_client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    assert isinstance(session, MCPGameSession)
    assert isinstance(rest_session, GameSession)
    assert not hasattr(session, "state")  # MCPGameSession never exposes REST-shaped method names
    assert not hasattr(rest_session, "get_state")  # nor vice versa
