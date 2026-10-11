"""Unit tests for AltruAgentClient.tournaments()/tournament()/join_tournament()/
leave_tournament(). All HTTP is mocked via httpx.MockTransport — none of
these require a running platform.
"""

from __future__ import annotations

import httpx
import pytest

from altruagent.client import AltruAgentClient
from altruagent.errors import AuthenticationError, PlatformError

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


def tournament_row(tournament_id: str, status: str, **overrides) -> dict:
    payload = {
        "tournament_id": tournament_id,
        "game_type": "tic_tac_toe",
        "status": status,
        "max_participants": 2,
        "current_participants": 1,
        "max_active_matches": 1,
        "queue_id": None,
        "created_at": "2026-01-01T00:00:00Z",
    }
    payload.update(overrides)
    return payload


# -- client.tournaments() --------------------------------------------------


def test_tournaments_makes_one_request_and_parses_list():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.url.path == "/tournaments"
        return httpx.Response(
            200,
            json={
                "tournaments": [
                    tournament_row("t-1", "waiting"),
                    tournament_row("t-2", "in_progress"),
                ]
            },
        )

    client = make_client(handler)
    tournaments = client.tournaments()

    assert [t.tournament_id for t in tournaments] == ["t-1", "t-2"]
    assert tournaments[0].status == "waiting"
    assert tournaments[1].status == "in_progress"


def test_tournaments_preserves_server_ordering():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            200,
            json={
                "tournaments": [
                    tournament_row("t-3", "waiting"),
                    tournament_row("t-1", "waiting"),
                    tournament_row("t-2", "in_progress"),
                ]
            },
        )

    client = make_client(handler)
    tournaments = client.tournaments()

    assert [t.tournament_id for t in tournaments] == ["t-3", "t-1", "t-2"]


def test_tournaments_does_no_detail_enrichment():
    detail_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        if request.url.path.startswith("/tournaments/"):
            detail_calls.append(request.url.path)
        return httpx.Response(200, json={"tournaments": [tournament_row("t-1", "waiting")]})

    client = make_client(handler)
    client.tournaments()

    assert detail_calls == []


# -- client.tournament(id) --------------------------------------------------


def test_tournament_detail_sends_authenticated_request_and_parses_viewer():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.url.path == "/tournaments/t-1"
        assert request.headers["Authorization"] == "Bearer jwt-1"
        return httpx.Response(
            200,
            json={
                "tournament": tournament_row("t-1", "in_progress", game_server_url="host:8000"),
                "viewer": {
                    "agent_id": "agent-1",
                    "is_tournament_participant": True,
                    "active_child_session_ids": ["session-1"],
                    "should_join_tournament": False,
                    "should_wait_for_child_match": False,
                    "next_actions": [{"action": "play_child_session", "hint": "Play it."}],
                },
                "roster": [],
                "participants": [],
                "counts": {"queued": 0, "active": 1, "completed": 0},
            },
        )

    client = make_client(handler)
    tournament = client.tournament("t-1")

    assert tournament.tournament_id == "t-1"
    assert tournament.game_server_url == "host:8000"
    assert tournament.viewer is not None
    assert tournament.viewer.is_tournament_participant is True
    assert tournament.viewer.active_child_session_ids == ["session-1"]


def test_tournament_detail_without_viewer_is_none():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            200,
            json={
                "tournament": tournament_row("t-1", "waiting"),
                "roster": [],
                "participants": [],
                "competitions": [],
                "counts": {"queued": 1, "active": 0, "completed": 0},
            },
        )

    client = make_client(handler)
    tournament = client.tournament("t-1")

    assert tournament.viewer is None


def test_get_nonexistent_tournament_404_preserved():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(404, json={"error": "Tournament not found"})

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.tournament("nonexistent")

    assert exc_info.value.status_code == 404


# -- client.join_tournament() / leave_tournament() --------------------------


def test_join_tournament_exact_path_and_no_body():
    request_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "POST"
        assert request.url.path == "/tournaments/t-1/join"
        request_bodies.append(request.content)
        return httpx.Response(
            200,
            json={
                "success": True,
                "tournament_id": "t-1",
                "status": "waiting",
                "position": 1,
                "next_actions": [
                    {"action": "poll_tournament", "endpoint": "GET /tournaments/t-1", "hint": "Poll it."}
                ],
            },
        )

    client = make_client(handler)
    result = client.join_tournament("t-1")

    assert request_bodies == [b""]  # no body sent
    assert result["success"] is True
    assert result["next_actions"][0]["action"] == "poll_tournament"


def test_leave_tournament_exact_path_and_no_body():
    request_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        assert request.method == "POST"
        assert request.url.path == "/tournaments/t-1/leave"
        request_bodies.append(request.content)
        return httpx.Response(200, json={"success": True, "status": "left", "tournament_id": "t-1"})

    client = make_client(handler)
    result = client.leave_tournament("t-1")

    assert request_bodies == [b""]
    assert result == {"success": True, "status": "left", "tournament_id": "t-1"}


def test_join_tournament_already_joined_success_tolerance():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "already_joined": True,
                "tournament_id": "t-1",
                "status": "in_progress",
                "position": 1,
            },
        )

    client = make_client(handler)
    result = client.join_tournament("t-1")

    assert result["already_joined"] is True


def test_leave_tournament_not_in_tournament_success_tolerance():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            200, json={"success": True, "status": "not_in_tournament", "tournament_id": "t-1"}
        )

    client = make_client(handler)
    result = client.leave_tournament("t-1")

    assert result["status"] == "not_in_tournament"


def test_join_tournament_failed_platform_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(400, json={"error": "join_failed", "detail": "Tournament is full"})

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.join_tournament("t-1")

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "join_failed"
    assert "Tournament is full" in str(exc_info.value)


def test_leave_tournament_failed_platform_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(
            400, json={"error": "leave_failed", "detail": "Cannot leave a tournament that has already started"}
        )

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.leave_tournament("t-1")

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "leave_failed"


def test_join_nonexistent_tournament_returns_400_join_failed_not_404():
    # Verified against Agent_ACP: unlike GET /tournaments/{id} (404 for a
    # missing tournament), POST .../join collapses every failure — including
    # "not found" — into the same 400 join_failed.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return login_ok(request)
        return httpx.Response(400, json={"error": "join_failed", "detail": "Tournament not found"})

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.join_tournament("nonexistent")

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "join_failed"


def test_join_tournament_401_triggers_one_relogin_and_one_retry():
    login_count = 0
    join_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        join_auth_headers.append(request.headers["Authorization"])
        if len(join_auth_headers) == 1:
            return httpx.Response(401, json={"error": "Invalid or expired token"})
        return httpx.Response(200, json={"success": True, "tournament_id": "t-1", "status": "waiting"})

    client = make_client(handler)
    result = client.join_tournament("t-1")

    assert result["success"] is True
    assert login_count == 2
    assert join_auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_join_tournament_second_401_stops_retrying():
    login_count = 0
    join_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count, join_attempts
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        join_attempts += 1
        return httpx.Response(401, json={"error": "Invalid or expired token"})

    client = make_client(handler)

    with pytest.raises(AuthenticationError):
        client.join_tournament("t-1")

    assert login_count == 2
    assert join_attempts == 2
