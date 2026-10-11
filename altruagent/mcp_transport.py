"""Synchronous bridge to Agent_ACP's MCP gameplay server, built on the
OFFICIAL `mcp` Python SDK client — not a hand-rolled JSON-RPC-over-HTTP
implementation (see "why" below).

History: an earlier version of this module hand-rolled the JSON-RPC 2.0
envelope directly over ``AltruAgentClient.request()``. It was replaced with
the official ``mcp.client.streamable_http.streamablehttp_client`` +
``mcp.ClientSession`` pattern Agent_ACP's own scripts use, which removes any
doubt about hand-rolled-protocol correctness. (The "Invalid Host header"
HTTP 421 first seen against the deployed server was a platform-side
allowed-hosts issue, since fixed in Agent_ACP; no client change was needed.)

Sync bridge design: the official SDK is async-only, but the rest of this
starter (``altruagent.runner``, ``altruagent.worker``, etc.) is deliberately
synchronous — turning that into asyncio-based code was explicitly ruled
out. Each call here therefore opens a **fresh** connection + MCP session
for that ONE tool call, via ``asyncio.run(...)``:

    asyncio.run() -> streamablehttp_client (fresh HTTP connection)
                  -> ClientSession -> initialize() -> call_tool() -> close

rather than keeping one long-lived async session open across an entire
match (which could span many minutes between turns). This is deliberate,
not a shortcut:

- The server is ``stateless_http=True`` — confirmed in server.py, this
  means the server itself holds no per-connection session state between
  requests, so there is no server-side affinity to preserve across calls.
- ``asyncio.run()`` is documented-safe to call repeatedly, sequentially, in
  the same process (each call creates and cleanly tears down its own event
  loop) — the only hard restriction is not calling it from *inside* an
  already-running loop, which never happens here since nothing else in this
  process runs asyncio.
- This sidesteps every cross-call event-loop-lifetime question entirely
  (Windows' default Proactor loop, multiprocessing ``spawn`` workers each
  starting with no inherited loop state, a match potentially idling a long
  time between turns) — there is no persistent
  loop or connection to keep alive, reconnect, or leak.
- The cost is one extra TCP handshake + MCP ``initialize`` round-trip per
  tool call versus a kept-open session. For a turn-based game polling at
  most a few times a second, this is immaterial.

JWT refresh: this bypasses ``AltruAgentClient.request()`` (the official SDK
owns its own HTTP transport), so the request()-level one-retry-after-401
logic can't be reused directly — but the *policy* is preserved exactly,
through ``AltruAgentClient._current_access_token()`` (same client, same
auth state, not a second auth system): fetch the current token, try once;
on a 401 (surfaced by the SDK's own ``response.raise_for_status()`` as
``httpx.HTTPStatusError`` — confirmed in mcp/client/streamable_http.py),
force a fresh login and retry exactly once, exactly like ``request()``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from pydantic_core import PydanticSerializationError

from .errors import PlatformError

if TYPE_CHECKING:
    from .client import AltruAgentClient

# How long one tool call may take. The SDK's own defaults are 30 s to connect
# and 300 s to read, so a connection that silently died (a Wi-Fi or VPN
# switch, a laptop waking up) would freeze the agent for five minutes. The
# longest healthy call is wait_for_update: the server holds it at most 25 s
# (GameAPI's WAIT_MAX_TIMEOUT_SECONDS), so the read timeout stays well above
# that plus network time. The session timeout is a backstop for an answer
# that trickles in without ever finishing. A call that times out raises
# MCPToolError (no error code), which the runner retries in place.
MCP_CONNECT_TIMEOUT_SECONDS = 10.0
MCP_READ_TIMEOUT_SECONDS = 40.0
MCP_SESSION_TIMEOUT_SECONDS = 45.0


class MCPToolError(PlatformError):
    """A tool call reached the server but failed — either a structured
    application-level error the tool itself returned (``error_code`` is one
    of MCP's own codes, e.g. ``STALE_STATE``/``GAME_ALREADY_COMPLETE``), or a
    protocol/transport-level failure (``error_code`` is ``None`` in that
    case — including an HTTP-level rejection, surfaced with its
    ``status_code``).

    ``protocol_error`` is True when the server answered, but with an MCP tool
    error (``isError``) rather than a result: the tool's arguments failed
    validation, or the tool itself crashed. GameAPI sends every game error
    as a normal result with an ``error`` code, so this is a call the server
    can't handle, likely to fail the same way again. It is not a connection
    problem (``errors.is_transient_error`` is False for it).

    ``local_error`` is True when the call never left this process: its
    arguments can't be sent as JSON (for example a numpy number, bytes, or an
    object of the agent's own class inside a move). Neither a connection nor
    a server problem; sending the same arguments again fails the same way
    (``errors.is_transient_error`` is False for it too).
    """

    def __init__(
        self, message: str, *, protocol_error: bool = False, local_error: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.protocol_error = protocol_error
        self.local_error = local_error


def _unsendable_arguments(name: str, exc: BaseException) -> MCPToolError:
    return MCPToolError(
        f"MCP tool {name!r} can't send its arguments as JSON: {exc}",
        status_code=None,
        error_code=None,
        local_error=True,
    )


def _check_arguments(name: str, arguments: dict) -> None:
    """Raise ``MCPToolError(local_error=True)`` if the SDK couldn't send
    ``arguments``: the same JSON conversion ``ClientSession.call_tool`` does,
    done before any connection is opened. Inside the SDK that failure would
    surface wrapped in anyio ExceptionGroups, indistinguishable from a
    dropped connection.
    """
    try:
        types.CallToolRequestParams(name=name, arguments=arguments).model_dump(
            by_alias=True, mode="json", exclude_none=True
        )
    except ValueError as exc:  # PydanticSerializationError, UnicodeDecodeError, ValidationError
        raise _unsendable_arguments(name, exc) from exc


def _find_serialization_error(exc: BaseException) -> PydanticSerializationError | None:
    """A ``PydanticSerializationError`` anywhere in anyio's ExceptionGroup
    wrapping: the SDK failed to turn this call's arguments into JSON (a
    backstop for ``_check_arguments``)."""
    if isinstance(exc, PydanticSerializationError):
        return exc
    for sub in getattr(exc, "exceptions", None) or ():
        found = _find_serialization_error(sub)
        if found is not None:
            return found
    return None


async def _call_tool_once(
    mcp_url: str, token: str, name: str, arguments: dict, *, httpx_client_factory=None
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    kwargs: dict[str, Any] = {
        "headers": headers,
        # The SDK builds httpx.Timeout(timeout, read=sse_read_timeout) from these.
        "timeout": MCP_CONNECT_TIMEOUT_SECONDS,
        "sse_read_timeout": MCP_READ_TIMEOUT_SECONDS,
    }
    if httpx_client_factory is not None:
        # Test-only seam: lets unit tests exercise the real streamablehttp_client
        # / ClientSession code paths (JSON-RPC framing, error unwrapping) against
        # an httpx.MockTransport-backed AsyncClient instead of real sockets,
        # rather than replacing the SDK's classes with hand-built fakes.
        kwargs["httpx_client_factory"] = httpx_client_factory
    async with streamablehttp_client(mcp_url, **kwargs) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=MCP_SESSION_TIMEOUT_SECONDS)
        ) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)

    if result.isError:
        raise MCPToolError(
            f"MCP tool {name!r} failed at the protocol level: {_extract_text(result)}",
            status_code=None,
            error_code=None,
            protocol_error=True,
        )

    payload = result.structuredContent
    if payload is None:
        text = _extract_text(result)
        try:
            payload = json.loads(text) if text else {}
        except ValueError:
            payload = {}

    if isinstance(payload, dict) and "error" in payload:
        raise MCPToolError(
            payload.get("detail") or f"MCP tool {name!r} returned error {payload['error']!r}.",
            status_code=None,
            error_code=payload["error"],
            next_action=(payload.get("next_actions") or [None])[0],
        )

    return payload if isinstance(payload, dict) else {}


def _extract_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text is not None:
            return str(text)
    return str(result)


def _find_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    """Unwrap anyio's ExceptionGroup wrapping.

    Confirmed empirically (not assumed): ``streamablehttp_client`` runs its
    read/write loops inside ``anyio.create_task_group()``
    (mcp/client/streamable_http.py), so an HTTP-level failure — including
    the "Invalid Host header" 421 this module's docstring describes — never
    arrives as a bare ``httpx.HTTPStatusError``; it's always wrapped in an
    ``ExceptionGroup``/``BaseExceptionGroup``, one or more levels deep. A
    plain ``except httpx.HTTPStatusError`` here would silently never match,
    which would have made the 401-retry-once behavior in ``call_tool`` dead
    code — probed directly against a real ``httpx.MockTransport`` before
    trusting this, not inferred from documentation.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc
    for sub in getattr(exc, "exceptions", None) or ():
        found = _find_http_status_error(sub)
        if found is not None:
            return found
    return None


def _wrap_http_status_error(name: str, exc: httpx.HTTPStatusError) -> MCPToolError:
    """Build a clean ``MCPToolError`` from an HTTP-level failure.

    ``exc.response`` is a *streaming* response: the SDK's own transport
    (``mcp.client.streamable_http``'s ``stream_within_origin``) calls
    ``raise_for_status()`` while the response is still open, then closes it
    via its own ``finally: await response.aclose()`` as this exception
    unwinds — before it ever reaches ``call_tool()``'s except clause, and
    without the body ever having been read. Accessing ``.text``/``.content``
    on a closed, never-read streaming response raises
    ``httpx.ResponseNotRead`` — confirmed empirically (a real 421/401
    response body is fully drained by the time it gets here) — which would
    otherwise mask the original HTTP status behind an unrelated crash. We
    have no way to read the body earlier (that would mean patching the SDK's
    own transport), so fall back to ``str(exc)`` instead: httpx's own
    ``HTTPStatusError`` message already includes the status code, reason
    phrase, and URL, which is enough to act on even without the body.
    """
    try:
        body = exc.response.text
        detail = body[:500] if body else str(exc)
    except httpx.ResponseNotRead:
        detail = str(exc)
    return MCPToolError(
        f"MCP tool {name!r} request failed with HTTP {exc.response.status_code}: {detail}",
        status_code=exc.response.status_code,
        error_code=None,
    )


def call_tool(
    client: "AltruAgentClient",
    mcp_url: str,
    name: str,
    arguments: dict,
    *,
    httpx_client_factory=None,
) -> dict:
    """Call one MCP tool and return its parsed result payload as a plain dict.

    ``mcp_url`` is ``{game_server_url}/mcp`` (see ``MCPGameSession``). Raises
    ``MCPToolError`` (a ``PlatformError`` subclass) for both application-level
    tool errors and transport/HTTP-level failures, so callers only need one
    except clause. A 401 triggers exactly one re-login + retry, reusing
    ``AltruAgentClient``'s own auth state — see the module docstring.

    ``httpx_client_factory`` is a test-only seam (see ``_call_tool_once``) —
    production callers never pass it.
    """
    _check_arguments(name, arguments)
    token = client._current_access_token()
    try:
        return asyncio.run(
            _call_tool_once(mcp_url, token, name, arguments, httpx_client_factory=httpx_client_factory)
        )
    except MCPToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad: network errors,
        # anyio ExceptionGroups, or any other mcp-SDK-internal failure should
        # all surface as an AltruAgentError, not a raw third-party exception
        # leaking out of this SDK. _find_http_status_error unwraps whatever
        # anyio's task-group wrapping did, so the 401 check below still works
        # regardless of how many ExceptionGroup layers deep it's nested.
        serialization_exc = _find_serialization_error(exc)
        if serialization_exc is not None:
            raise _unsendable_arguments(name, serialization_exc) from exc
        http_exc = _find_http_status_error(exc)
        if http_exc is None:
            raise MCPToolError(
                f"MCP tool {name!r} failed: {exc!r}", status_code=None, error_code=None
            ) from exc
        if http_exc.response.status_code != 401:
            raise _wrap_http_status_error(name, http_exc) from exc

        token = client._current_access_token(force_relogin=True)
        try:
            return asyncio.run(
                _call_tool_once(mcp_url, token, name, arguments, httpx_client_factory=httpx_client_factory)
            )
        except MCPToolError:
            raise
        except Exception as exc2:  # noqa: BLE001
            http_exc2 = _find_http_status_error(exc2)
            if http_exc2 is not None:
                raise _wrap_http_status_error(name, http_exc2) from exc2
            raise MCPToolError(
                f"MCP tool {name!r} failed after re-login: {exc2!r}", status_code=None, error_code=None
            ) from exc2
