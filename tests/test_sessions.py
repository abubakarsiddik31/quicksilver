"""Unit tests for AltruAgentClient.sessions() and Match.game()'s lazy
resolution. All HTTP is mocked via httpx.MockTransport — none of these
require a running platform.
"""

from __future__ import annotations

import httpx
import pytest

from altruagent.client import AltruAgentClient
from altruagent.errors import AuthenticationError, PlatformError
from altruagent.game import GameSession
from altruagent.mcp_game import MCPGameSession

CONTROL_URL = "https://example.test"


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


def match_payload(session_id: str, status: str, **overrides) -> dict:
    payload = {
        "session_id": session_id,
        "game_type": "tic_tac_toe",
        "status": status,
        "tournament_id": None,
        "created_at": "2026-01-01T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def sessions_payload(**overrides) -> dict:
    payload = {"joined_sessions": [], "active_sessions": [], "completed_sessions": []}
    payload.update(overrides)
    return payload


def test_client_sessions_successful_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.url.path == "/agents/me/sessions"
        return httpx.Response(
            200,
            json=sessions_payload(
                joined_sessions=[match_payload("s-waiting", "waiting")],
                active_sessions=[match_payload("s-active", "in_progress")],
                completed_sessions=[match_payload("s-done", "completed")],
            ),
        )

    client = make_client(handler)
    sessions = client.sessions()

    assert [m.session_id for m in sessions.waiting] == ["s-waiting"]
    assert [m.session_id for m in sessions.active] == ["s-active"]
    assert [m.session_id for m in sessions.completed] == ["s-done"]


def test_sessions_makes_exactly_one_request_to_agents_me_sessions():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json=sessions_payload(
                active_sessions=[
                    match_payload("s-1", "in_progress"),
                    match_payload("s-2", "in_progress"),
                ]
            ),
        )

    client = make_client(handler)
    client.sessions()

    assert calls == ["/agents/me/sessions"]


def test_sessions_does_not_resolve_game_server_url():
    competitions_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path.startswith("/competitions/"):
            competitions_calls.append(request.url.path)
            return httpx.Response(200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"})
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    sessions = client.sessions()

    assert competitions_calls == []
    assert sessions.active[0].game_server_url is None


def test_multiple_active_matches_remain_independent():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            return httpx.Response(200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host-1:8000"})
        if request.url.path == "/competitions/s-2":
            return httpx.Response(200, json={"session_id": "s-2", "status": "in_progress", "game_server_url": "host-2:8000"})
        return httpx.Response(
            200,
            json=sessions_payload(
                active_sessions=[
                    match_payload("s-1", "in_progress"),
                    match_payload("s-2", "in_progress"),
                ]
            ),
        )

    client = make_client(handler)
    sessions = client.sessions()
    assert len(sessions.active) == 2

    game_1 = sessions.active[0].game()
    assert game_1.game_server_url == "https://host-1:8000"
    # Resolving match 1 must not have touched match 2's cached URL.
    assert sessions.active[1].game_server_url is None

    game_2 = sessions.active[1].game()
    assert game_2.game_server_url == "https://host-2:8000"
    # Match.game_server_url caches the raw value the server returned (not the
    # GameSession's normalized http://-prefixed form) — each match's cache is
    # independent of the other.
    assert sessions.active[0].game_server_url == "host-1:8000"
    assert sessions.active[1].game_server_url == "host-2:8000"


def test_active_match_game_performs_competition_lookup_and_builds_mcp_session():
    lookups = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            lookups.append(request.url.path)
            return httpx.Response(
                200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]
    game = match.game()

    assert lookups == ["/competitions/s-1"]
    # match.game() is the production path (MCP) — see match.rest_game() below
    # for the lower-level REST debug path.
    assert isinstance(game, MCPGameSession)
    assert game.session_id == "s-1"
    assert game.game_server_url == "https://host:8000"


def test_active_match_rest_game_performs_competition_lookup_and_builds_gamesession():
    # rest_game() is the debug-only REST path (scripts/check_game.py) — kept
    # separate from game() so run_match/python -m agent never touch it.
    lookups = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            lookups.append(request.url.path)
            return httpx.Response(
                200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]
    game = match.rest_game()

    assert lookups == ["/competitions/s-1"]
    assert isinstance(game, GameSession)
    assert game.session_id == "s-1"
    assert game.game_server_url == "https://host:8000"


def test_game_and_rest_game_share_the_same_cached_game_server_url():
    # Both transports are resolved by the same _resolve_game_server_url —
    # confirming one Match instance can hand out both without a second lookup.
    lookups = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            lookups.append(request.url.path)
            return httpx.Response(
                200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]

    mcp_game = match.game()
    rest_game = match.rest_game()

    assert lookups == ["/competitions/s-1"]  # exactly one lookup for both
    assert isinstance(mcp_game, MCPGameSession)
    assert isinstance(rest_game, GameSession)
    assert mcp_game.game_server_url == rest_game.game_server_url == "https://host:8000"


def test_resolved_game_server_url_is_cached_and_not_refetched():
    lookups = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            lookups.append(request.url.path)
            return httpx.Response(
                200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]

    match.game()
    match.game()
    match.game()

    assert lookups == ["/competitions/s-1"]  # exactly one lookup despite three calls


def test_waiting_match_game_fails_clearly_without_network_call():
    competitions_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path.startswith("/competitions/"):
            competitions_calls.append(request.url.path)
        return httpx.Response(
            200, json=sessions_payload(joined_sessions=[match_payload("s-1", "waiting")])
        )

    client = make_client(handler)
    match = client.sessions().waiting[0]

    with pytest.raises(ValueError):
        match.game()

    assert competitions_calls == []


def test_completed_match_game_fails_clearly_without_network_call():
    competitions_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path.startswith("/competitions/"):
            competitions_calls.append(request.url.path)
        return httpx.Response(
            200, json=sessions_payload(completed_sessions=[match_payload("s-1", "completed")])
        )

    client = make_client(handler)
    match = client.sessions().completed[0]

    with pytest.raises(ValueError):
        match.game()

    assert competitions_calls == []


def test_sessions_401_triggers_one_relogin_and_one_retry():
    login_count = 0
    sessions_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        sessions_auth_headers.append(request.headers["Authorization"])
        if len(sessions_auth_headers) == 1:
            return httpx.Response(401, json={"error": "Invalid or expired token"})
        return httpx.Response(200, json=sessions_payload())

    client = make_client(handler)
    sessions = client.sessions()

    assert sessions.waiting == []
    assert login_count == 2
    assert sessions_auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_sessions_second_401_stops_retrying():
    login_count = 0
    sessions_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count, sessions_attempts
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        sessions_attempts += 1
        return httpx.Response(401, json={"error": "Invalid or expired token"})

    client = make_client(handler)

    with pytest.raises(AuthenticationError):
        client.sessions()

    assert login_count == 2
    assert sessions_attempts == 2


def test_lazy_resolution_401_triggers_one_relogin_and_one_retry():
    login_count = 0
    competition_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        if request.url.path == "/competitions/s-1":
            competition_auth_headers.append(request.headers["Authorization"])
            if len(competition_auth_headers) == 1:
                return httpx.Response(401, json={"error": "Invalid or expired token"})
            return httpx.Response(
                200, json={"session_id": "s-1", "status": "in_progress", "game_server_url": "host:8000"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]
    game = match.game()

    assert isinstance(game, MCPGameSession)
    assert login_count == 2
    assert competition_auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_malformed_non_json_error_during_lazy_resolution_is_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path == "/competitions/s-1":
            return httpx.Response(
                502, text="<html>Bad Gateway</html>", headers={"content-type": "text/html"}
            )
        return httpx.Response(
            200, json=sessions_payload(active_sessions=[match_payload("s-1", "in_progress")])
        )

    client = make_client(handler)
    match = client.sessions().active[0]

    with pytest.raises(PlatformError) as exc_info:
        match.game()

    assert exc_info.value.status_code == 502
    assert "Bad Gateway" in str(exc_info.value)
