"""Exception hierarchy for the AltruAgent SDK.

The platform's error responses are not perfectly uniform across endpoints.
Most carry ``{"error": "<machine_code>", "detail": "...", "next_actions": [...]}``;
GameAPI errors instead carry a single ``recovery_action`` object; and a few
control-plane auth errors (e.g. a bad API key on login) carry only
``{"error": "<human-readable message>"}`` with no ``detail`` at all. Parsing
in ``client.py`` is defensive about all of this: unknown/missing fields
degrade to ``None`` rather than raising, and a non-JSON body (e.g. an
upstream gateway error page) is captured as raw text instead of crashing.
"""

from __future__ import annotations


class AltruAgentError(Exception):
    """Base class for all errors raised by this SDK."""


class ConfigurationError(AltruAgentError):
    """Required configuration (e.g. control URL or API key) is missing or invalid.

    Raised locally, before any request is made.
    """


class AuthenticationError(AltruAgentError):
    """Login failed, or a request was rejected as unauthenticated after one retry.

    The platform has no refresh-token grant, so this is also what you get if
    an API key is revoked or simply wrong.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


class PlatformError(AltruAgentError):
    """The control plane returned a non-2xx response, or could not be reached at all
    (e.g. a network failure or timeout — `status_code` is `None` in that case).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        error_code: str | None = None,
        detail: str | None = None,
        next_action: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.next_action = next_action


# The platform has answered a failed agent-session mint (its own sign-in
# service rejecting or rate-limiting the request) with 401
# invalid_official_agent_key and a detail starting with this. A temporary
# platform problem, not a key problem (see ``altruagent.official``).
MINT_FAILURE_PREFIX = "Failed to mint agent session"

# Machine codes that mean "a temporary problem, try again shortly".
_TRANSIENT_ERROR_CODES = frozenset(
    {
        "RUNTIME_TEMPORARILY_UNAVAILABLE",  # GameAPI: the game's engine is busy or restarting
        "rate_limited",                     # control plane: too many requests just now
        "agent_session_unavailable",        # control plane: couldn't start or check a session just now
    }
)


def is_transient_error(exc: BaseException) -> bool:
    """True for a temporary problem between this agent and the platform, or
    on the platform: no network or a dropped connection, a gateway or server
    error (5xx), too many requests (429), a game engine that is busy right
    now, or the platform failing to start a session. Retrying after a short
    pause can help, and none of it is the agent's fault.

    False for a definite answer: a game rule (``STALE_STATE``,
    ``INVALID_ACTION``, ``SESSION_NOT_FOUND``, ...), a refused key or
    registration, a seat held by another runtime, a request the platform
    rejected as wrong (other 4xx), a call the game server answered with an
    MCP tool error (``protocol_error``: arguments it couldn't accept, or the
    tool crashed), which would most likely fail the same way again, or a call
    that never left this process because its arguments can't be sent as JSON
    (``local_error``: a bug in the agent's own code, not a connection problem).
    """
    if not isinstance(exc, (PlatformError, AuthenticationError)):
        return False
    if getattr(exc, "protocol_error", False) or getattr(exc, "local_error", False):
        return False
    code, status = exc.error_code, exc.status_code
    if code in _TRANSIENT_ERROR_CODES:
        return True
    if code == "invalid_official_agent_key":
        return (exc.detail or "").startswith(MINT_FAILURE_PREFIX)
    if status is not None and (status >= 500 or status in (408, 429)):
        return True
    if type(exc) is AuthenticationError and status == 401:
        # A freshly issued agent session the platform didn't accept yet (the
        # retry right after signing in again still got 401).
        return True
    if code is not None:
        return False
    # No answer at all: the network, or the connection to the game server.
    return status is None and isinstance(exc, PlatformError)
