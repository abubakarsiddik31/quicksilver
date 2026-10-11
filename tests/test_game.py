"""Unit tests for GameSession. All HTTP is mocked via httpx.MockTransport —
none of these require a running platform.
"""

from __future__ import annotations

import json

import httpx
import pytest

from altruagent.client import AltruAgentClient
from altruagent.errors import AuthenticationError, PlatformError
from altruagent.game import _normalize_game_server_url

CONTROL_URL = "https://example.test"
GAME_SERVER_URL = "http://game.example.test"
SESSION_ID = "session-1"


def make_client(handler, *, api_key: str = "sk_agent_test") -> AltruAgentClient:
    transport = httpx.MockTransport(handler)
    return AltruAgentClient(
        control_url=CONTROL_URL,
        api_key=api_key,
        transport=transport,
        load_env_file=False,
    )


def login_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"access_token": "jwt-1"})


def state_payload(**overrides) -> dict:
    payload = {
        "session_id": SESSION_ID,
        "game_name": "tic_tac_toe",
        "status": "active",
        "observation": "...",
        "current_player": {"name": "Alice"},
        "legal_actions": [0, 1, 2],
        "legal_actions_str": ["a", "b", "c"],
        "is_terminal": False,
        "returns": None,
        "move_count": 0,
        "termination_reason": None,
        "messaging_enabled": False,
        "phase": "moving",
        "next_actions": [
            {"action": "make_move", "endpoint": "POST /step", "hint": "Go.", "required_fields": ["action"]}
        ],
    }
    payload.update(overrides)
    return payload


def test_fetch_game_state_successfully():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "GET"
        assert str(request.url) == f"{GAME_SERVER_URL}/games/{SESSION_ID}"
        assert request.headers["Authorization"] == "Bearer jwt-1"
        return httpx.Response(200, json=state_payload())

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.state()

    assert state.session_id == SESSION_ID
    # REST's legal_actions/legal_actions_str pair is synthesized into
    # LegalAction entries, same shape MCP produces natively (see models.py).
    assert [a.action_id for a in state.legal_actions] == ["0", "1", "2"]
    assert [a.label for a in state.legal_actions] == ["a", "b", "c"]
    assert state.next_actions[0].action == "make_move"


def test_game_server_url_without_scheme_is_normalized():
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=state_payload())

    client = make_client(handler)
    # Real control-plane game_server_url values are bare hosts (see
    # gameAPIService.ts's getGameAPIServerUrl), e.g. "game.example.test".
    session = client.game(session_id=SESSION_ID, game_server_url="game.example.test")
    session.state()

    assert seen_urls == [f"https://game.example.test/games/{SESSION_ID}"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("gameapi.altruagent-game.com", "https://gameapi.altruagent-game.com"),
        ("game.example.test:8443/", "https://game.example.test:8443"),
        ("localhost:8000", "http://localhost:8000"),
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("[::1]:8000", "http://[::1]:8000"),
        ("http://game.example.test", "http://game.example.test"),  # explicit scheme kept
        ("HTTPS://game.example.test/", "HTTPS://game.example.test"),
    ],
)
def test_normalize_game_server_url_defaults_remote_hosts_to_https(given, expected):
    assert _normalize_game_server_url(given) == expected


def test_submitting_a_valid_step():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "POST"
        assert str(request.url) == f"{GAME_SERVER_URL}/games/{SESSION_ID}/step"
        assert json.loads(request.content) == {"action": 1}
        return httpx.Response(200, json=state_payload(move_count=1, current_player={"name": "Bob"}))

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.step(1)

    assert state.move_count == 1
    assert state.current_player.name == "Bob"


def test_resigning():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "POST"
        assert str(request.url) == f"{GAME_SERVER_URL}/games/{SESSION_ID}/resign"
        return httpx.Response(
            200,
            json=state_payload(
                is_terminal=True,
                current_player=None,
                legal_actions=[],
                returns={"Alice": -1.0, "Bob": 1.0},
                termination_reason="resignation",
                next_actions=[{"action": "game_over", "hint": "Game finished."}],
            ),
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.resign()

    assert state.is_terminal is True
    assert state.termination_reason == "resignation"
    assert state.returns == {"Alice": -1.0, "Bob": 1.0}


def test_invalid_action_error_preserves_machine_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            400,
            json={
                "error": "invalid_action",
                "detail": "Action 9 is not in legal_actions",
                "recovery_action": {
                    "action": "make_move",
                    "endpoint": "POST /games/{session_id}/step",
                    "hint": "Re-fetch state, then pick from legal_actions.",
                    "required_fields": ["action"],
                },
            },
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.step(9)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "invalid_action"
    assert exc_info.value.next_action["action"] == "make_move"


def test_not_your_turn_error_preserves_machine_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            409,
            json={
                "error": "not_your_turn",
                "detail": "It is not your turn.",
                "recovery_action": {"action": "fetch_state", "hint": "Refresh state."},
            },
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.step(0)

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "not_your_turn"


def test_user_not_in_game_error_preserves_machine_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            403, json={"error": "user_not_in_game", "detail": "User not in this game"}
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.state()

    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "user_not_in_game"


def test_rate_limit_429_error_preserved():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            429,
            json={
                "error": "rate_limit_exceeded",
                "detail": "Rate limit exceeded. Try again later.",
                "recovery_action": {
                    "action": "fetch_state",
                    "hint": "Back off and retry after at least 1 second.",
                },
            },
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.state()

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_code == "rate_limit_exceeded"
    assert exc_info.value.next_action["action"] == "fetch_state"


def test_gameapi_401_causes_one_relogin_and_one_retry():
    login_count = 0
    state_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        state_auth_headers.append(request.headers["Authorization"])
        if len(state_auth_headers) == 1:
            return httpx.Response(401, json={"error": "Invalid or expired token"})
        return httpx.Response(200, json=state_payload())

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.state()

    assert state.session_id == SESSION_ID
    assert login_count == 2
    assert state_auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_gameapi_second_401_stops_retrying():
    login_count = 0
    state_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count, state_attempts
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        state_attempts += 1
        return httpx.Response(401, json={"error": "Invalid or expired token"})

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(AuthenticationError):
        session.state()

    assert login_count == 2
    assert state_attempts == 2


def messaging_state_payload(**overrides) -> dict:
    payload = state_payload(
        messaging_enabled=True,
        messaging_mode="per_all_moves",
        phase="messaging",
        terminated_messaging=[],
        new_messages=[],
        next_actions=[
            {"action": "send_message", "hint": "chat or terminate", "required_fields": ["type"]},
            {"action": "terminate_messaging", "hint": "end round", "required_fields": ["type"]},
        ],
    )
    payload.update(overrides)
    return payload


def test_send_chat_message_broadcast():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "POST"
        assert str(request.url) == f"{GAME_SERVER_URL}/games/{SESSION_ID}/message"
        assert json.loads(request.content) == {"type": "chat", "content": "hi", "recipients": []}
        return httpx.Response(200, json=messaging_state_payload())

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.send_message("hi")

    assert state.phase == "messaging"
    assert state.messaging_enabled is True


def test_send_chat_message_to_one_recipient():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert json.loads(request.content) == {
            "type": "chat",
            "content": "psst",
            "recipients": [1],
        }
        return httpx.Response(200, json=messaging_state_payload())

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    session.send_message("psst", recipients=[1])


def test_terminate_messaging_sends_terminate_type():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert str(request.url) == f"{GAME_SERVER_URL}/games/{SESSION_ID}/message"
        assert json.loads(request.content) == {"type": "terminate", "recipients": []}
        return httpx.Response(
            200,
            json=messaging_state_payload(
                phase="moving",
                next_actions=[{"action": "wait_for_opponent", "endpoint": "GET /x", "hint": "Wait."}],
            ),
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    state = session.terminate_messaging()

    assert state.phase == "moving"


def test_messaging_auth_retry_goes_through_existing_client_machinery():
    login_count = 0
    auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        auth_headers.append(request.headers["Authorization"])
        if len(auth_headers) == 1:
            return httpx.Response(401, json={"error": "Invalid or expired token"})
        return httpx.Response(200, json=messaging_state_payload())

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)
    session.send_message("hi")

    assert login_count == 2
    assert auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_messages_quota_exceeded_error_preserves_machine_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            429,
            json={
                "error": "messages_quota_exceeded",
                "detail": "Per-agent chat cap reached.",
                "recovery_action": {"action": "terminate_messaging", "hint": "Terminate instead."},
            },
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.send_message("one too many")

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_code == "messages_quota_exceeded"
    assert exc_info.value.next_action["action"] == "terminate_messaging"


def test_wrong_phase_error_on_message_preserves_machine_code():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            409,
            json={
                "error": "wrong_phase",
                "detail": "Cannot send message during moving phase.",
                "recovery_action": {"action": "make_move", "hint": "POST your move."},
            },
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.send_message("too late")

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "wrong_phase"


def test_malformed_non_json_gameapi_error_is_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            502, text="<html>Bad Gateway</html>", headers={"content-type": "text/html"}
        )

    client = make_client(handler)
    session = client.game(session_id=SESSION_ID, game_server_url=GAME_SERVER_URL)

    with pytest.raises(PlatformError) as exc_info:
        session.state()

    assert exc_info.value.status_code == 502
    assert "Bad Gateway" in str(exc_info.value)
