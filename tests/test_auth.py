"""Unit tests for altruagent.auth (ApiKeyAuth/SeatGrantAuth), the SeatGrant
model, and how AltruAgentClient's existing one-retry-after-401 policy drives
seat renewal — over REST ``request()`` and over the real MCP transport
(``call_tool``, exercising the official ``mcp`` SDK against a mocked HTTP
layer, same technique as tests/test_mcp_transport.py). No real network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from altruagent.auth import SEAT_CLAIM_PATH, ApiKeyAuth, SeatClaimError, SeatGrantAuth
from altruagent.client import AltruAgentClient
from altruagent.errors import AuthenticationError, ConfigurationError, PlatformError
from altruagent.mcp_transport import MCPToolError, call_tool
from altruagent.models import SeatGrant

CONTROL_URL = "https://control.example.test"
CLAIM_TOKEN = "seatclaim_abc123secret"
MCP_URL = "https://game.example.test/mcp"


def grant_body(access_token: str = "seat-jwt-1", **overrides) -> dict:
    body = {
        "access_token": access_token,
        "expires_at": "2026-09-26T12:00:00.000Z",
        "agent_id": "synthetic-agent-0",
        "gameapi_server_url": "https://gameapi.example.test",
        "game_session_id": "game-session-1",
        "match_id": "match-1",
        "seat_id": "seat-0",
        "seat_position": 0,
        "seat_count": 2,
        "game_type": "pokemon_vgc_doubles_draft",
        "match_status": "starting",
    }
    body.update(overrides)
    return body


class ClaimServer:
    """Mock control plane: records every claim request body, and answers
    each one with the next queued response (the last one repeats).
    """

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses) or [httpx.Response(200, json=grant_body())]
        self.claims: list[dict] = []
        self.other_requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == SEAT_CLAIM_PATH and request.method == "POST":
            self.claims.append(json.loads(request.content))
            index = min(len(self.claims) - 1, len(self.responses) - 1)
            return self.responses[index]
        self.other_requests.append(request.url.path)
        return httpx.Response(404)


def seat_client(server, auth: SeatGrantAuth | None = None) -> AltruAgentClient:
    return AltruAgentClient(
        control_url=CONTROL_URL,
        auth=auth or SeatGrantAuth(CLAIM_TOKEN),
        transport=httpx.MockTransport(server),
        load_env_file=False,
    )


# -- SeatGrant ---------------------------------------------------------------


def test_seat_grant_parses_every_backend_field():
    grant = SeatGrant.from_dict(grant_body())

    assert grant.access_token == "seat-jwt-1"
    assert grant.expires_at == "2026-09-26T12:00:00.000Z"
    assert grant.agent_id == "synthetic-agent-0"
    assert grant.gameapi_server_url == "https://gameapi.example.test"
    assert grant.game_session_id == "game-session-1"
    assert grant.match_id == "match-1"
    assert grant.seat_id == "seat-0"
    assert grant.seat_position == 0
    assert grant.seat_count == 2
    assert grant.game_type == "pokemon_vgc_doubles_draft"
    assert grant.match_status == "starting"


def test_seat_grant_never_exposes_access_token_in_repr_or_raw():
    grant = SeatGrant.from_dict(grant_body(access_token="very-secret-jwt"))

    assert "very-secret-jwt" not in repr(grant)
    assert "very-secret-jwt" not in str(grant)
    assert "access_token" not in grant.raw
    assert grant.raw["match_id"] == "match-1"


def test_seat_grant_tolerates_missing_and_extra_fields():
    grant = SeatGrant.from_dict({"access_token": "t", "future_field": 1})

    assert grant.seat_position is None
    assert grant.game_session_id == ""
    assert grant.raw["future_field"] == 1


# -- SeatGrantAuth: claim request / claim_key ----------------------------------


def test_claim_success_posts_token_and_key_and_returns_grant_token():
    server = ClaimServer()
    auth = SeatGrantAuth(CLAIM_TOKEN)
    client = seat_client(server, auth)

    client.login()

    assert len(server.claims) == 1
    body = server.claims[0]
    assert set(body) == {"claim_token", "claim_key"}
    assert body["claim_token"] == CLAIM_TOKEN
    assert client._access_token == "seat-jwt-1"
    assert auth.grant.game_session_id == "game-session-1"
    assert auth.grant.seat_position == 0


def test_claim_key_is_strong_random_and_stable_within_one_auth():
    server = ClaimServer()
    auth = SeatGrantAuth(CLAIM_TOKEN)
    client = seat_client(server, auth)

    client.login()
    client.login()
    client.login()

    keys = {claim["claim_key"] for claim in server.claims}
    assert len(keys) == 1
    key = keys.pop()
    assert len(key) >= 43  # 32 random bytes, urlsafe-base64 — well above the backend's 16-char floor
    assert key != CLAIM_TOKEN


def test_separate_auth_objects_get_independent_claim_keys():
    server = ClaimServer()
    seat_client(server, SeatGrantAuth(CLAIM_TOKEN)).login()
    seat_client(server, SeatGrantAuth(CLAIM_TOKEN)).login()

    assert server.claims[0]["claim_key"] != server.claims[1]["claim_key"]


def test_claim_token_whitespace_is_stripped():
    server = ClaimServer()
    seat_client(server, SeatGrantAuth(f"  {CLAIM_TOKEN}\n")).login()

    assert server.claims[0]["claim_token"] == CLAIM_TOKEN


@pytest.mark.parametrize("bad", ["", "sk_agent_abc", "not-a-token"])
def test_non_seatclaim_token_is_rejected_locally(bad):
    with pytest.raises(SeatClaimError) as exc_info:
        SeatGrantAuth(bad)

    assert exc_info.value.error_code == "invalid_claim_token"
    if bad:
        assert bad not in str(exc_info.value)


def test_auth_repr_never_shows_claim_token_or_key():
    server = ClaimServer(httpx.Response(200, json=grant_body(access_token="secret-access")))
    auth = SeatGrantAuth(CLAIM_TOKEN)
    seat_client(server, auth).login()

    text = repr(auth)
    assert CLAIM_TOKEN not in text
    assert server.claims[0]["claim_key"] not in text
    assert "secret-access" not in text
    assert "ApiKeyAuth(api_key=<redacted>)" == repr(ApiKeyAuth("sk_agent_x"))


# -- SeatGrantAuth: error mapping ------------------------------------------------


@pytest.mark.parametrize(
    "status, code, phrase",
    [
        (404, "invalid_claim_token", "isn't valid"),
        (409, "seat_already_claimed", "already claimed"),
        (410, "claim_expired", "expired"),
        (409, "match_not_claimable", "no longer active"),
        (429, "rate_limited", "Too many"),
    ],
)
def test_claim_errors_map_to_clear_seat_claim_errors(status, code, phrase):
    server = ClaimServer(
        httpx.Response(status, json={"error": code, "detail": "internal backend detail xyz"})
    )
    client = seat_client(server)

    with pytest.raises(SeatClaimError) as exc_info:
        client.login()

    error = exc_info.value
    assert isinstance(error, AuthenticationError)
    assert error.error_code == code
    assert error.status_code == status
    assert phrase in str(error)
    assert "internal backend detail xyz" not in str(error)
    assert CLAIM_TOKEN not in str(error)
    assert client._access_token is None


def test_retired_claim_codes_point_to_the_official_agent_key():
    server = ClaimServer(httpx.Response(410, json={
        "error": "claim_codes_retired",
        "detail": "Testing now uses your Official Agent Key: run your agent with --match and it picks up your Testing games automatically.",
    }))

    with pytest.raises(SeatClaimError) as exc_info:
        seat_client(server).login()

    error = exc_info.value
    assert (error.status_code, error.error_code) == (410, "claim_codes_retired")
    assert str(error).startswith("Testing claim codes were retired; run with --match and your Official Agent Key")
    assert "ALTRUAGENT_OFFICIAL_AGENT_KEY" in str(error) and CLAIM_TOKEN not in str(error)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(410, json={"error": "platform_agents_retired", "detail": "AltruAgent now runs on the UCLA site."}),
        httpx.Response(410, text="Gone"),
    ],
)
def test_retired_platform_api_key_login_points_to_the_official_agent_key(response):
    client = AltruAgentClient(
        control_url=CONTROL_URL, api_key="sk_agent_old_platform_key",
        transport=httpx.MockTransport(lambda request: response), load_env_file=False,
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.login()

    message = str(exc_info.value)
    assert exc_info.value.status_code == 410
    assert "API-key mode (ALTRUAGENT_API_KEY) was retired" in message
    assert "python -m agent --tournament" in message and "sk_agent_old_platform_key" not in message


def test_unknown_claim_error_still_raises_seat_claim_error():
    server = ClaimServer(httpx.Response(500, json={"error": "internal_error", "detail": "boom"}))

    with pytest.raises(SeatClaimError) as exc_info:
        seat_client(server).login()

    assert "HTTP 500" in str(exc_info.value)
    assert exc_info.value.error_code == "internal_error"


def test_claim_response_missing_required_fields_is_rejected():
    server = ClaimServer(httpx.Response(200, json={"access_token": "t"}))

    with pytest.raises(SeatClaimError):
        seat_client(server).login()


def test_claim_network_failure_is_platform_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(PlatformError) as exc_info:
        seat_client(handler).login()

    assert exc_info.value.status_code is None
    assert CLAIM_TOKEN not in str(exc_info.value)


def test_renewal_failure_says_it_was_a_renewal():
    server = ClaimServer(
        httpx.Response(200, json=grant_body()),
        httpx.Response(409, json={"error": "match_not_claimable"}),
    )
    auth = SeatGrantAuth(CLAIM_TOKEN)
    client = seat_client(server, auth)
    client.login()

    with pytest.raises(SeatClaimError) as exc_info:
        client.login()

    assert "renew" in str(exc_info.value)
    assert exc_info.value.error_code == "match_not_claimable"


def test_renewal_returning_a_different_seat_is_rejected():
    server = ClaimServer(
        httpx.Response(200, json=grant_body()),
        httpx.Response(200, json=grant_body(access_token="seat-jwt-2", seat_id="someone-else")),
    )
    auth = SeatGrantAuth(CLAIM_TOKEN)
    client = seat_client(server, auth)
    client.login()

    with pytest.raises(SeatClaimError):
        client.login()
    assert auth.grant.seat_id == "seat-0"  # the original grant is kept


# -- AltruAgentClient configuration ----------------------------------------------


def test_seat_auth_client_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("ALTRUAGENT_API_KEY", raising=False)
    monkeypatch.setenv("ALTRUAGENT_CONTROL_URL", CONTROL_URL)

    client = AltruAgentClient(auth=SeatGrantAuth(CLAIM_TOKEN), load_env_file=False)

    assert isinstance(client.auth, SeatGrantAuth)
    assert client.control_url == CONTROL_URL


def test_seat_auth_client_still_needs_control_url(monkeypatch):
    monkeypatch.delenv("ALTRUAGENT_CONTROL_URL", raising=False)

    with pytest.raises(ConfigurationError):
        AltruAgentClient(auth=SeatGrantAuth(CLAIM_TOKEN), load_env_file=False)


def test_api_key_and_auth_together_are_rejected():
    with pytest.raises(ConfigurationError):
        AltruAgentClient(
            control_url=CONTROL_URL, api_key="sk_agent_x", auth=SeatGrantAuth(CLAIM_TOKEN), load_env_file=False
        )


def test_default_auth_is_api_key_auth():
    client = AltruAgentClient(control_url=CONTROL_URL, api_key="sk_agent_x", load_env_file=False)

    assert isinstance(client.auth, ApiKeyAuth)


# -- renewal through the client's one-retry-after-401 policy (REST) --------------


def test_rest_401_renews_with_same_token_and_key_then_retries_once():
    tokens_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == SEAT_CLAIM_PATH:
            return claims(request)
        tokens_seen.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer seat-jwt-1":
            return httpx.Response(401, json={"detail": "Invalid or expired token"})
        return httpx.Response(200, json={"ok": True})

    claims = ClaimServer(
        httpx.Response(200, json=grant_body("seat-jwt-1")),
        httpx.Response(200, json=grant_body("seat-jwt-2")),
    )
    client = seat_client(handler)

    result = client.request("GET", "https://gameapi.example.test/games/game-session-1")

    assert result == {"ok": True}
    assert tokens_seen == ["Bearer seat-jwt-1", "Bearer seat-jwt-2"]
    assert len(claims.claims) == 2
    assert claims.claims[0] == claims.claims[1]  # same claim_token AND same claim_key
    assert client._access_token == "seat-jwt-2"
    assert client.auth.grant.access_token == "seat-jwt-2"


def test_rest_second_401_after_renewal_stops_without_looping():
    claims = ClaimServer()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == SEAT_CLAIM_PATH:
            return claims(request)
        return httpx.Response(401, json={"detail": "Invalid or expired token"})

    with pytest.raises(AuthenticationError):
        seat_client(handler).request("GET", "https://gameapi.example.test/games/x")

    assert len(claims.claims) == 2  # initial claim + exactly one renewal


def test_rest_failed_renewal_raises_authentication_error():
    claims = ClaimServer(
        httpx.Response(200, json=grant_body()),
        httpx.Response(410, json={"error": "claim_expired"}),
    )
    game_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == SEAT_CLAIM_PATH:
            return claims(request)
        game_calls.append(1)
        return httpx.Response(401)

    with pytest.raises(SeatClaimError) as exc_info:
        seat_client(handler).request("GET", "https://gameapi.example.test/games/x")

    assert isinstance(exc_info.value, AuthenticationError)
    assert len(claims.claims) == 2
    assert len(game_calls) == 1  # never retried after the renewal itself failed


# -- renewal through the real MCP transport ---------------------------------------


def _initialize_response(req_id) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "gameapi-test", "version": "0.0.0"},
        },
    }


def _mcp_server(accepted_token: str, attempts: list):
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        if method == "initialize":
            attempts.append(request.headers.get("authorization"))
            if request.headers.get("authorization") != f"Bearer {accepted_token}":
                return httpx.Response(401, json={"error": "invalid token"})
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            tools = [{"name": "get_game_state", "inputSchema": {"type": "object"}, "outputSchema": None}]
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
        if method == "tools/call":
            payload = {"ok": True}
            result = {"content": [{"type": "text", "text": json.dumps(payload)}], "structuredContent": payload, "isError": False}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": req_id, "result": result})
        return httpx.Response(404)

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers, timeout=30.0)

    return factory


def test_mcp_401_renews_seat_and_retries_with_new_token():
    claims = ClaimServer(
        httpx.Response(200, json=grant_body("seat-jwt-1")),
        httpx.Response(200, json=grant_body("seat-jwt-2")),
    )
    client = seat_client(claims)
    client.login()
    attempts: list = []

    result = call_tool(
        client, MCP_URL, "get_game_state", {"session_id": "game-session-1"},
        httpx_client_factory=_mcp_server("seat-jwt-2", attempts),
    )

    assert result == {"ok": True}
    assert attempts == ["Bearer seat-jwt-1", "Bearer seat-jwt-2"]
    assert len(claims.claims) == 2
    assert claims.claims[0] == claims.claims[1]
    assert client._access_token == "seat-jwt-2"


def test_mcp_failed_renewal_raises_authentication_error_without_looping():
    claims = ClaimServer(
        httpx.Response(200, json=grant_body("seat-jwt-1")),
        httpx.Response(409, json={"error": "match_not_claimable"}),
    )
    client = seat_client(claims)
    client.login()
    attempts: list = []

    with pytest.raises(AuthenticationError):
        call_tool(
            client, MCP_URL, "get_game_state", {"session_id": "game-session-1"},
            httpx_client_factory=_mcp_server("never-accepted", attempts),
        )

    assert len(attempts) == 1
    assert len(claims.claims) == 2


def test_mcp_second_401_after_successful_renewal_stops():
    claims = ClaimServer(
        httpx.Response(200, json=grant_body("seat-jwt-1")),
        httpx.Response(200, json=grant_body("seat-jwt-2")),
    )
    client = seat_client(claims)
    client.login()
    attempts: list = []

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(
            client, MCP_URL, "get_game_state", {"session_id": "game-session-1"},
            httpx_client_factory=_mcp_server("never-accepted", attempts),
        )

    assert exc_info.value.status_code == 401
    assert len(attempts) == 2
    assert len(claims.claims) == 2
