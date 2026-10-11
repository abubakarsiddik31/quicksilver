"""Unit tests for altruagent.mcp_transport.call_tool — built on the OFFICIAL
`mcp` SDK client (mcp.ClientSession + mcp.client.streamable_http), not a
hand-rolled JSON-RPC implementation (see mcp_transport.py's module docstring
for why: the first hand-rolled version failed against the real deployed
platform with "Invalid Host header", a server-side DNS-rebinding-protection
misconfiguration unrelated to which client sends the request).

These tests exercise the REAL `streamablehttp_client`/`ClientSession` code
paths (JSON-RPC framing, the initialize handshake, tools/list output-schema
validation, error unwrapping) against a fully mocked async HTTP layer, via
`streamablehttp_client`'s own `httpx_client_factory` seam — not hand-built
fakes standing in for the SDK's classes. This matters: it was exactly this
kind of "what does the real SDK actually do" gap that caused the original
transport's live failure, so these tests are deliberately close to the real
thing rather than mocking away the parts that turned out to matter (the
anyio ExceptionGroup wrapping below was only discovered by testing this way).

Every mocked server handles four JSON-RPC methods, mirroring what the real
`mcp.ClientSession` actually sends per call (confirmed empirically, not
assumed): `initialize`, `notifications/initialized`, `tools/call`, and
`tools/list` (call_tool always validates output structure against the
tool's schema, refreshing that schema via list_tools() once per fresh
session — which this transport always is, by design; see mcp_transport.py).
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from altruagent import mcp_transport
from altruagent.errors import AuthenticationError, is_transient_error
from altruagent.mcp_transport import MCPToolError, call_tool

MCP_URL = "http://game.example.test/mcp"


class FakeClient:
    """Stand-in for AltruAgentClient — only the one method call_tool()
    actually needs (`_current_access_token`), so tests don't need a real
    AltruAgentClient/httpx transport at all.
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = list(tokens or ["jwt-1"])
        self._issued: list[tuple[str, bool]] = []
        self._relogin_calls = 0

    def _current_access_token(self, *, force_relogin: bool = False) -> str:
        if force_relogin:
            self._relogin_calls += 1
        index = min(self._relogin_calls, len(self._tokens) - 1)
        token = self._tokens[index]
        self._issued.append((token, force_relogin))
        return token


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


def _tools_list_response(req_id) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": [{"name": "get_game_state", "inputSchema": {"type": "object"}, "outputSchema": None}]},
    }


def _tool_call_response(req_id, payload: dict, *, is_error: bool = False) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "structuredContent": None if is_error else payload,
            "isError": is_error,
        },
    }


def make_factory(handler):
    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers, timeout=30.0)

    return factory


def test_call_tool_success_sends_correct_arguments_and_auth_header():
    seen_auth_headers = []
    seen_arguments = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization"))
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        if method == "initialize":
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            seen_arguments.append(body["params"]["arguments"])
            return httpx.Response(200, json=_tool_call_response(req_id, {"session_id": "s-1"}))
        return httpx.Response(404)

    client = FakeClient()
    result = call_tool(
        client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler)
    )

    assert result == {"session_id": "s-1"}
    assert seen_arguments == [{"session_id": "s-1"}]
    assert all(h == "Bearer jwt-1" for h in seen_auth_headers)


def test_call_tool_unwraps_structured_content():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        if method == "initialize":
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            return httpx.Response(200, json=_tool_call_response(req_id, {"state_version": 3, "actions": []}))
        return httpx.Response(404)

    client = FakeClient()
    result = call_tool(
        client, MCP_URL, "get_legal_actions", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler)
    )

    assert result == {"state_version": 3, "actions": []}


def test_call_tool_application_level_error_raises_mcp_tool_error_with_code():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        if method == "initialize":
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            payload = {
                "error": "STALE_STATE",
                "detail": "state_version 3 is stale; current is 4.",
                "next_actions": [{"tool": "get_game_state", "hint": "Re-fetch state."}],
            }
            return httpx.Response(200, json=_tool_call_response(req_id, payload))
        return httpx.Response(404)

    client = FakeClient()

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(
            client,
            MCP_URL,
            "play_action",
            {"session_id": "s-1", "state_version": 3, "action_id": "0"},
            httpx_client_factory=make_factory(handler),
        )

    assert exc_info.value.error_code == "STALE_STATE"
    assert "stale" in str(exc_info.value).lower()
    assert exc_info.value.next_action == {"tool": "get_game_state", "hint": "Re-fetch state."}


def test_call_tool_protocol_level_error_raises_mcp_tool_error_with_no_code():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        if method == "initialize":
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            return httpx.Response(200, json=_tool_call_response(req_id, {}, is_error=True))
        return httpx.Response(404)

    client = FakeClient()

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(client, MCP_URL, "bogus_tool", {}, httpx_client_factory=make_factory(handler))

    assert exc_info.value.error_code is None


def test_a_tool_error_answer_is_a_protocol_error_not_a_connection_problem():
    # The server answered, with isError (FastMCP: arguments it couldn't
    # accept, or the tool crashed). Retrying for 90 s as if the connection
    # had dropped would ask the agent again and again for nothing.
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "tools/call":
            return httpx.Response(200, json=_tool_call_response(body.get("id"), {}, is_error=True))
        return _healthy_handler(request)

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(FakeClient(), MCP_URL, "play_action", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler))

    assert exc_info.value.protocol_error is True
    assert (exc_info.value.error_code, exc_info.value.status_code) == (None, None)
    assert not is_transient_error(exc_info.value)


class _AgentsOwnType:
    """Stands in for a numpy number or a class of the contestant's own."""


@pytest.mark.parametrize("value", [_AgentsOwnType(), b"\xff\xfe"], ids=["own-class", "bad-bytes"])
def test_arguments_that_cant_be_sent_as_json_are_a_local_error_and_nothing_is_sent(value):
    # Was: the SDK's serialization error came back wrapped in ExceptionGroups,
    # as an MCPToolError with no code, which is_transient_error took for a
    # dropped connection. The runner then asked the agent again and again for
    # 90 s ("Connection problem"), for a bug in the agent's own code.
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _healthy_handler(request)

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(FakeClient(), MCP_URL, "play_action", {"session_id": "s-1", "action": {"x": value}},
                  httpx_client_factory=make_factory(handler))

    assert exc_info.value.local_error is True and exc_info.value.protocol_error is False
    assert "can't send its arguments as JSON" in str(exc_info.value)
    assert not is_transient_error(exc_info.value)
    assert requests == []  # checked before any connection is opened


def test_a_serialization_error_inside_the_sdk_is_still_a_local_error(monkeypatch):
    # Backstop, should the SDK ever convert arguments differently from the
    # up-front check: its own PydanticSerializationError, found inside the
    # ExceptionGroup wrapping, is still not taken for a connection problem.
    monkeypatch.setattr(mcp_transport, "_check_arguments", lambda name, arguments: None)

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(FakeClient(), MCP_URL, "play_action", {"session_id": "s-1", "action": {"x": _AgentsOwnType()}},
                  httpx_client_factory=make_factory(_healthy_handler))

    assert exc_info.value.local_error is True
    assert not is_transient_error(exc_info.value)


def test_plain_arguments_pass_the_up_front_check():
    result = call_tool(FakeClient(), MCP_URL, "play_action",
                       {"session_id": "s-1", "action": {"type": "move", "x": 3, "ids": [1, 2], "ok": None}},
                       httpx_client_factory=make_factory(_healthy_handler))

    assert result == {"session_id": "s-1"}


def test_invalid_host_header_421_surfaces_as_mcp_tool_error_with_status_code():
    """Regression test for the real failure this transport was rewritten
    for: a 421 on the very first request (before any tool executes) must
    surface as a clear MCPToolError, not an unhandled ExceptionGroup.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(421, text="Invalid Host header")

    client = FakeClient()

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler))

    assert exc_info.value.status_code == 421
    assert "Invalid Host header" in str(exc_info.value)
    assert client._relogin_calls == 0  # 421 is not 401 — must never trigger a re-login


def test_401_triggers_exactly_one_relogin_and_retry_succeeds():
    attempts = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        auth = request.headers.get("authorization")
        if method == "initialize":
            attempts["n"] += 1
            if auth == "Bearer jwt-1":
                return httpx.Response(401, json={"error": "invalid token"})
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            return httpx.Response(200, json=_tool_call_response(req_id, {"ok": True}))
        return httpx.Response(404)

    client = FakeClient(tokens=["jwt-1", "jwt-2"])
    result = call_tool(
        client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler)
    )

    assert result == {"ok": True}
    assert client._issued == [("jwt-1", False), ("jwt-2", True)]
    assert client._relogin_calls == 1
    assert attempts["n"] == 2  # exactly one retry, never looped


def test_second_401_after_retry_still_fails_clearly():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(401, json={"error": "invalid token"})
        return httpx.Response(404)

    client = FakeClient(tokens=["jwt-1", "jwt-2"])

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler))

    assert exc_info.value.status_code == 401
    assert client._relogin_calls == 1  # retried exactly once, then gave up


def test_call_tool_reuses_client_login_state_not_a_second_auth_system():
    # AltruAgentClient itself, not a fake — proves _current_access_token()
    # really does reuse the one existing login()/AuthenticationError path.
    from altruagent.client import AltruAgentClient

    login_calls = []

    def handler_sync(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/agent/login":
            login_calls.append(1)
            return httpx.Response(200, json={"access_token": f"jwt-{len(login_calls)}"})
        return httpx.Response(200, json={})

    real_client = AltruAgentClient(
        control_url="https://example.test",
        api_key="sk_agent_test",
        transport=httpx.MockTransport(handler_sync),
        load_env_file=False,
    )

    async def mcp_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, req_id = body.get("method"), body.get("id")
        auth = request.headers.get("authorization")
        if method == "initialize":
            if auth == "Bearer jwt-1":
                return httpx.Response(401, json={"error": "invalid"})
            return httpx.Response(200, json=_initialize_response(req_id))
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json=_tools_list_response(req_id))
        if method == "tools/call":
            return httpx.Response(200, json=_tool_call_response(req_id, {"ok": True}))
        return httpx.Response(404)

    result = call_tool(
        real_client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(mcp_handler)
    )

    assert result == {"ok": True}
    assert len(login_calls) == 2  # first login() (lazy) + one forced re-login after 401
    assert real_client._access_token == "jwt-2"


class _ClosedWithoutReadStream(httpx.AsyncByteStream):
    """An httpx response stream that has been closed without ever being
    read — reproduces exactly what mcp.client.streamable_http's own
    ``stream_within_origin`` leaves behind: it calls ``raise_for_status()``
    on a streaming response, then closes it in its own ``finally`` clause as
    the exception unwinds, all before ``call_tool()``'s except clause (and
    ``_wrap_http_status_error``) ever sees it. Confirmed against the real
    live platform: a real 401/421 response reaching ``_wrap_http_status_error``
    is always in this already-closed, never-read state.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data

    async def aclose(self) -> None:
        pass


def test_wrap_http_status_error_survives_unread_closed_response():
    """Regression test: _wrap_http_status_error must not itself raise
    httpx.ResponseNotRead when handed a response whose body was never read
    (the real shape of every HTTP-level failure by the time it gets here) —
    it must produce a clean MCPToolError using httpx's own exception message
    instead of crashing with a second, unrelated exception on top of the
    original HTTP status.
    """
    from altruagent.mcp_transport import _wrap_http_status_error

    request = httpx.Request("POST", MCP_URL)
    response = httpx.Response(421, stream=_ClosedWithoutReadStream(b"Invalid Host header"), request=request)

    async def _close_without_reading() -> None:
        await response.aclose()

    asyncio.run(_close_without_reading())

    with pytest.raises(httpx.ResponseNotRead):
        response.text  # sanity check: confirms this reproduces the real failure mode

    exc = httpx.HTTPStatusError("Client error '421 Misdirected Request'", request=request, response=response)
    wrapped = _wrap_http_status_error("get_game_state", exc)

    assert wrapped.status_code == 421
    assert "421" in str(wrapped)
    assert "Misdirected Request" in str(wrapped)


def test_call_tool_handles_unread_closed_error_response_end_to_end():
    """End-to-end version of the above: a mocked server whose 421 response
    is only ever accessed in the closed-without-reading state that the real
    SDK transport leaves it in, driven through the actual call_tool() path.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(421, stream=_ClosedWithoutReadStream(b"Invalid Host header"))

    client = FakeClient()

    with pytest.raises(MCPToolError) as exc_info:
        call_tool(client, MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=make_factory(handler))

    assert exc_info.value.status_code == 421


# -- timeouts: a dead connection must not freeze the agent for minutes -------------------


def _healthy_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    method, req_id = body.get("method"), body.get("id")
    if method == "initialize":
        return httpx.Response(200, json=_initialize_response(req_id))
    if method == "notifications/initialized":
        return httpx.Response(202)
    if method == "tools/list":
        return httpx.Response(200, json=_tools_list_response(req_id))
    return httpx.Response(200, json=_tool_call_response(req_id, {"session_id": "s-1"}))


def test_the_http_client_gets_a_short_connect_and_a_40s_read_timeout_not_the_sdks_300s():
    # The SDK's own default is httpx.Timeout(30, read=300): one silently dropped
    # connection would freeze a game for five minutes.
    seen = []

    def factory(headers=None, timeout=None, auth=None):
        seen.append(timeout)
        return httpx.AsyncClient(transport=httpx.MockTransport(_healthy_handler), headers=headers, timeout=timeout)

    call_tool(FakeClient(), MCP_URL, "get_game_state", {"session_id": "s-1"}, httpx_client_factory=factory)

    (timeout,) = seen
    assert timeout == httpx.Timeout(mcp_transport.MCP_CONNECT_TIMEOUT_SECONDS, read=mcp_transport.MCP_READ_TIMEOUT_SECONDS)
    assert (timeout.connect, timeout.read) == (10.0, 40.0)
    # Above the server's longest healthy call: wait_for_update holds at most 25 s.
    assert mcp_transport.MCP_READ_TIMEOUT_SECONDS >= 25.0 + 10.0
    assert mcp_transport.MCP_SESSION_TIMEOUT_SECONDS > mcp_transport.MCP_READ_TIMEOUT_SECONDS


def test_a_server_that_never_answers_fails_the_call_after_the_session_timeout(monkeypatch):
    # A mock transport ignores httpx's socket timeouts, so this exercises the
    # session-level backstop: the call fails as a transient MCPToolError
    # (no error code) instead of hanging.
    monkeypatch.setattr(mcp_transport, "MCP_SESSION_TIMEOUT_SECONDS", 0.3)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "tools/call":
            await asyncio.sleep(30)
        return _healthy_handler(request)

    started = time.monotonic()
    with pytest.raises(MCPToolError) as exc_info:
        call_tool(FakeClient(), MCP_URL, "wait_for_update", {"session_id": "s-1"},
                  httpx_client_factory=make_factory(handler))

    assert time.monotonic() - started < 10
    assert exc_info.value.error_code is None and exc_info.value.status_code is None
    assert exc_info.value.protocol_error is False  # no answer at all, unlike an isError answer
    assert is_transient_error(exc_info.value)


@pytest.fixture
def silent_server():
    """A real local TCP server that accepts each connection, reads the
    request, and never answers: what a silently dropped connection looks like
    from the client's side. Yields its MCP URL."""
    import socket
    import threading

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    held: list[socket.socket] = []

    def accept() -> None:
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            held.append(connection)
            threading.Thread(target=connection.recv, args=(65536,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}/mcp"
    server.close()
    for connection in held:
        connection.close()


def test_the_read_timeout_reaches_the_real_connection(monkeypatch, silent_server):
    # Over a real socket (not a mock transport): the HTTP read timeout itself
    # ends a call whose server went silent, long before the session backstop.
    monkeypatch.setattr(mcp_transport, "MCP_READ_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(mcp_transport, "MCP_SESSION_TIMEOUT_SECONDS", 30.0)

    started = time.monotonic()
    with pytest.raises(MCPToolError) as exc_info:
        call_tool(FakeClient(), silent_server, "wait_for_update", {"session_id": "s-1"})

    assert time.monotonic() - started < 10
    assert "ReadTimeout" in str(exc_info.value)
    assert is_transient_error(exc_info.value)
