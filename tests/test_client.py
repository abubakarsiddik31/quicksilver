"""Unit tests for AltruAgentClient. All HTTP is mocked via httpx.MockTransport —
none of these require a running platform.
"""

from __future__ import annotations

import json

import httpx
import pytest

from altruagent.client import AltruAgentClient
from altruagent.errors import AuthenticationError, ConfigurationError, PlatformError


def make_client(handler, *, api_key: str = "sk_agent_test") -> AltruAgentClient:
    transport = httpx.MockTransport(handler)
    return AltruAgentClient(
        control_url="https://example.test",
        api_key=api_key,
        transport=transport,
        load_env_file=False,
    )


def test_login_and_me_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/auth/agent/login":
            assert json.loads(request.content) == {"api_key": "sk_agent_test"}
            return httpx.Response(200, json={"access_token": "jwt-1"})
        if request.url.path == "/auth/agent/me":
            assert request.headers["Authorization"] == "Bearer jwt-1"
            return httpx.Response(
                200,
                json={
                    "id": "agent-1",
                    "name": "MyAgent",
                    "status": "claimed",
                    "claimed_by_user_id": "human-1",
                    "claimed_at": "2026-01-01T00:00:00Z",
                    "created_at": "2026-01-01T00:00:00Z",
                    "api_key_hash": "should-not-leak",
                    "claim_token_hash": "should-not-leak-either",
                    "next_actions": [{"action": "discover_competitions"}],
                },
            )
        raise AssertionError(f"unexpected request to {request.url.path}")

    client = make_client(handler)
    agent = client.me()

    assert agent.id == "agent-1"
    assert agent.name == "MyAgent"
    assert agent.status == "claimed"
    assert agent.is_claimed is True
    assert calls == ["/auth/agent/login", "/auth/agent/me"]


def test_invalid_api_key_raises_authentication_error():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/auth/agent/login"
        return httpx.Response(401, json={"error": "Invalid API key"})

    client = make_client(handler, api_key="sk_agent_bad")

    with pytest.raises(AuthenticationError) as exc_info:
        client.me()

    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)


def test_automatic_relogin_and_single_retry_after_401():
    login_count = 0
    me_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        if request.url.path == "/auth/agent/me":
            me_auth_headers.append(request.headers["Authorization"])
            if len(me_auth_headers) == 1:
                # Server treats the first token as stale/expired.
                return httpx.Response(401, json={"error": "Invalid or expired token"})
            return httpx.Response(
                200, json={"id": "agent-1", "name": "A", "status": "claimed"}
            )
        raise AssertionError(f"unexpected request to {request.url.path}")

    client = make_client(handler)
    agent = client.me()

    assert agent.id == "agent-1"
    assert login_count == 2  # initial login + exactly one re-login after the 401
    assert me_auth_headers == ["Bearer jwt-1", "Bearer jwt-2"]


def test_retry_stops_after_second_401():
    login_count = 0
    me_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count, me_attempts
        if request.url.path == "/auth/agent/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"jwt-{login_count}"})
        if request.url.path == "/auth/agent/me":
            me_attempts += 1
            return httpx.Response(401, json={"error": "Invalid or expired token"})
        raise AssertionError(f"unexpected request to {request.url.path}")

    client = make_client(handler)

    with pytest.raises(AuthenticationError):
        client.me()

    assert login_count == 2  # initial login + exactly one re-login, no loop
    assert me_attempts == 2  # exactly one retry


def test_missing_configuration_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("ALTRUAGENT_CONTROL_URL", raising=False)
    monkeypatch.delenv("ALTRUAGENT_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        AltruAgentClient(load_env_file=False)


def test_missing_api_key_only_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("ALTRUAGENT_CONTROL_URL", "https://example.test")
    monkeypatch.delenv("ALTRUAGENT_API_KEY", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        AltruAgentClient(load_env_file=False)

    # The default API-key mode is retired: the error says what to do instead.
    assert "was retired" in str(exc_info.value)
    assert "ALTRUAGENT_OFFICIAL_AGENT_KEY" in str(exc_info.value)


def test_malformed_non_json_error_response_is_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            return httpx.Response(200, json={"access_token": "jwt-1"})
        if request.url.path == "/auth/agent/me":
            return httpx.Response(
                502, text="<html>Bad Gateway</html>", headers={"content-type": "text/html"}
            )
        raise AssertionError(f"unexpected request to {request.url.path}")

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.me()

    assert exc_info.value.status_code == 502
    assert "Bad Gateway" in str(exc_info.value)


def test_network_failure_is_wrapped_as_platform_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler)

    with pytest.raises(PlatformError) as exc_info:
        client.me()

    assert exc_info.value.status_code is None
    assert "example.test" in str(exc_info.value)
