"""The kept-alive JSON-RPC fast path for MCP tool calls (no per-call handshake)."""

import json

import httpx
import pytest

from altruagent import mcp_transport as mt


class Auth:
    def __init__(self):
        self.relogins = 0

    def _current_access_token(self, force_relogin=False):
        if force_relogin:
            self.relogins += 1
            return "fresh"
        return "stale" if self.relogins == 0 else "fresh"


@pytest.fixture
def server(monkeypatch):
    seen = []

    def install(handler):
        def wrapped(request):
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(mt, "_RAW_CLIENT", httpx.Client(transport=httpx.MockTransport(wrapped)))
        monkeypatch.delenv("ALTRUAGENT_MCP_SESSION", raising=False)
        return seen

    return install


def ok(payload, sse=False):
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": payload, "isError": False, "content": []},
    }
    if sse:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="event: message\ndata: " + json.dumps(body) + "\n\n",
        )
    return httpx.Response(200, json=body)


def test_one_post_per_call_with_the_documented_headers(server):
    seen = server(lambda r: ok({"state_version": 7}))
    out = mt.call_tool(Auth(), "https://g/mcp", "get_game_state", {"session_id": "s"})
    assert out == {"state_version": 7} and len(seen) == 1
    req = seen[0]
    body = json.loads(req.content)
    assert body["method"] == "tools/call" and body["params"] == {
        "name": "get_game_state",
        "arguments": {"session_id": "s"},
    }
    assert req.headers["authorization"] == "Bearer stale"
    assert req.headers["accept"] == "application/json, text/event-stream"


def test_sse_reply_tool_errors_and_401_relogin(server):
    server(lambda r: ok({"a": 1}, sse=True))
    assert mt.call_tool(Auth(), "https://g/mcp", "t", {}) == {"a": 1}

    server(
        lambda r: ok(
            {"error": "STALE_STATE", "detail": "old", "next_actions": [{"tool": "get_game_state"}]}
        )
    )
    with pytest.raises(mt.MCPToolError) as e:
        mt.call_tool(Auth(), "https://g/mcp", "play_action", {})
    assert e.value.error_code == "STALE_STATE"

    auth = Auth()
    server(
        lambda r: (
            httpx.Response(401)
            if r.headers["authorization"] == "Bearer stale"
            else ok({"ok": True})
        )
    )
    assert mt.call_tool(auth, "https://g/mcp", "t", {}) == {"ok": True} and auth.relogins == 1

    server(lambda r: httpx.Response(503, text="busy"))
    with pytest.raises(mt.MCPToolError) as e:
        mt.call_tool(Auth(), "https://g/mcp", "t", {})
    assert e.value.status_code == 503

    def drop(r):
        raise httpx.ConnectError("gone")

    server(drop)
    with pytest.raises(mt.MCPToolError) as e:
        mt.call_tool(Auth(), "https://g/mcp", "t", {})
    assert (
        e.value.error_code is None and e.value.status_code is None
    )  # transient: the runner retries

    server(
        lambda r: httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"isError": True, "content": [{"type": "text", "text": "bad args"}]},
            },
        )
    )
    with pytest.raises(mt.MCPToolError) as e:
        mt.call_tool(Auth(), "https://g/mcp", "t", {})
    assert e.value.protocol_error
