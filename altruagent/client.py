"""Synchronous control-plane client for the AltruAgent competition platform.

Today this client is the HTTP/auth plumbing under ``altruagent.official``:
``OfficialAgentClient`` and every game worker build one with an explicit
``auth=`` strategy. Its default ``ApiKeyAuth`` mode (``ALTRUAGENT_API_KEY``)
and the platform-agent endpoints wrapped below (``me``, ``sessions``,
``tournaments``, ``join_tournament``, ...) are RETIRED: the platform turned
platform agents off (HTTP 410 ``platform_agents_retired``). They're kept for
reference only; see ``altruagent.notices``.

Handles configuration, token login through a pluggable strategy (see
``official.py`` for the ones in use and ``auth.py`` for the retired ones),
and the platform's one-retry-after-401 convention. Wraps the (retired)
platform-agent auth endpoints directly:

- ``POST /auth/agent/login`` (Agent_ACP backend/src/index.ts:133,
  services/agentService.ts:81 ``loginAgent``)
- ``GET /auth/agent/me`` (index.ts:223)

...and exposes a small reusable authenticated-request helper (``request``)
that ``GameSession`` (see ``game.py``) builds on to talk to a GameAPI
``game_server_url`` using the *same* JWT and the *same* one-retry-after-401
behavior — there is only ever one authentication system, not one per host.

The platform issues no refresh token (verified against
Agent_ACP/backend/src/middleware/auth.ts and agentService.ts — login always
mints a brand new JWT from the API key, and that's the only recovery path
after a 401). So the client's retry policy is exactly that: on 401, log in
again once and retry the request once. See backend/skill/02-auth.md for the
agent-facing description of the same convention, confirmed identical on the
GameAPI side by gameapi/src/gameapi/auth/jwt_validator.py (same Supabase
JWT, validated independently via JWKS).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx
from dotenv import load_dotenv

from ._responses import _parse_error_body, _parse_json_body
from .auth import ApiKeyAuth, AuthStrategy
from .errors import AuthenticationError, ConfigurationError, PlatformError
from .models import Agent, AgentSessions, Tournament
from .notices import PLATFORM_KEY_RETIRED_NOTICE

if TYPE_CHECKING:
    from .game import GameSession
    from .mcp_game import MCPGameSession

DEFAULT_TIMEOUT_SECONDS = 10.0


class AltruAgentClient:
    """Small sync HTTP client for the AltruAgent control plane.

    ``auth=`` sets how the bearer token is obtained. The runtime always
    passes one (``OfficialAgentAuth`` for the control plane,
    ``OfficialSeatAuth`` for one seat's GameAPI traffic; see
    ``altruagent.official``). Omitted, the client falls back to the RETIRED
    ``ApiKeyAuth`` mode with ``api_key``/``ALTRUAGENT_API_KEY``, which the
    platform no longer accepts.
    """

    def __init__(
        self,
        control_url: str | None = None,
        api_key: str | None = None,
        *,
        auth: AuthStrategy | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        load_env_file: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if load_env_file:
            # No-op if there's no .env file; explicit args / real env vars
            # always take precedence over anything loaded from it.
            load_dotenv()

        if auth is not None and api_key is not None:
            raise ConfigurationError("Pass either api_key= or auth=, not both.")

        control_url = control_url or os.environ.get("ALTRUAGENT_CONTROL_URL")

        if not control_url:
            raise ConfigurationError(
                "ALTRUAGENT_CONTROL_URL is not set. Copy .env.example to .env and fill it "
                "in, or pass control_url= explicitly."
            )
        if auth is None:
            api_key = api_key or os.environ.get("ALTRUAGENT_API_KEY")
            if not api_key:
                raise ConfigurationError(PLATFORM_KEY_RETIRED_NOTICE)
            auth = ApiKeyAuth(api_key)

        self.control_url = control_url.rstrip("/")
        self.auth = auth
        self._access_token: str | None = None
        self._http = httpx.Client(base_url=self.control_url, timeout=timeout, transport=transport)

    def __enter__(self) -> "AltruAgentClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- public API -----------------------------------------------------

    def login(self) -> None:
        """Obtain a fresh bearer token through this client's auth strategy —
        an agent session from the Official Agent Key (``OfficialAgentAuth``),
        or one assigned seat's grant (``OfficialSeatAuth``). The retired
        default, ``ApiKeyAuth``, calls ``POST /auth/agent/login``.

        The token is kept only in memory on this instance; it is never written
        to disk or logged. Safe to call again at any time — a fresh login
        mints a new token that still resolves to the same identity.
        """
        self._access_token = self.auth.login(self._http, self.control_url)

    def _current_access_token(self, *, force_relogin: bool = False) -> str:
        """Return this client's current JWT — the single source of auth
        state for both REST (``request()``, above) and MCP traffic
        (``altruagent.mcp_transport.call_tool``, which cannot reuse
        ``request()`` directly since the official ``mcp`` SDK owns its own
        HTTP transport). Logs in if there's no token yet, or if
        ``force_relogin=True`` — the MCP transport's own one-retry-after-401
        recovery, mirroring ``request()``'s exactly, so there is still only
        ever one login/retry policy, not a second auth system.
        """
        if self._access_token is None or force_relogin:
            self.login()
        return self._access_token

    def me(self) -> Agent:
        """``GET /auth/agent/me`` — the authenticated agent's profile."""
        data = self.request("GET", "/auth/agent/me")
        return Agent.from_dict(data if isinstance(data, dict) else {})

    def sessions(self) -> AgentSessions:
        """``GET /agents/me/sessions`` — this agent's competition memberships,
        grouped into waiting/active/completed (see Agent_ACP
        backend/src/index.ts:253, ``listAgentSessions``).

        Exactly one control-plane request. Deliberately does **not** resolve
        ``game_server_url`` for any match, and makes no GameAPI calls — that
        endpoint doesn't carry it (it's not a stored column anywhere), and
        resolving it eagerly for every returned match would turn a cheap
        discovery call into N+1 requests. Call ``match.game()`` on whichever
        specific match you actually want to play; it resolves lazily and
        only then.
        """
        data = self.request("GET", "/agents/me/sessions")
        return AgentSessions.from_dict(data if isinstance(data, dict) else {}, client=self)

    def tournaments(self) -> list[Tournament]:
        """``GET /tournaments`` — public listing of active tournaments (see
        Agent_ACP backend/src/index.ts:442, ``listActiveTournaments`` /
        ``db/tournaments.ts``'s ``getActiveTournaments``).

        Exactly one request; no per-tournament detail calls. Entries are raw
        DB rows, returned in whatever order the server gives them (newest
        `created_at` first, currently). The backend already filters this to
        `waiting`/`in_progress` only (capped at 10) — completed tournaments
        never appear here, so no client-side filtering is applied on top.
        """
        data = self.request("GET", "/tournaments")
        rows = data.get("tournaments") if isinstance(data, dict) else None
        return [Tournament.from_dict(row) for row in (rows or [])]

    def tournament(self, tournament_id: str) -> Tournament:
        """``GET /tournaments/{id}`` — one tournament's detail, including this
        agent's ``viewer`` membership info (present because this uses the
        authenticated request path, so an agent JWT is sent even though the
        route itself only optionally requires one).

        Note: this GET is **not side-effect-free** on the current backend.
        ``getTournament`` (tournamentService.ts) can, as a side effect of
        this same call: advance a tournament past an expired start timer
        (starting it), and reconcile any child match GameAPI already
        finished (recomputing the leaderboard, scheduling the next batch of
        matches, or completing the tournament). This is real platform
        behavior, not something the SDK compensates for or hides.
        """
        data = self.request("GET", f"/tournaments/{tournament_id}")
        tournament_data = data.get("tournament") if isinstance(data, dict) else None
        viewer_data = data.get("viewer") if isinstance(data, dict) else None
        return Tournament.from_dict(tournament_data or {}, viewer=viewer_data)

    def join_tournament(self, tournament_id: str) -> dict:
        """``POST /tournaments/{id}/join``. Identity comes entirely from the
        authenticated, claimed agent's JWT — no request body is sent or
        needed (verified against Agent_ACP index.ts:462, which never reads
        ``req.body``).

        Returns the parsed JSON response as a plain dict (a dedicated model
        would add little value here — the only fields worth reading are
        ``status``/``position``/``next_actions``, all already anonymous
        dict keys). Idempotent: rejoining a tournament you're already in
        returns success with ``already_joined: true`` rather than an error.
        If this join fills the tournament's capacity, the tournament starts
        synchronously as part of this same call — every round-robin child
        competition is created before this returns. A full/already-started/
        nonexistent tournament all collapse to the same
        ``PlatformError(error_code="join_failed")`` — the backend does not
        distinguish them with separate machine codes (and, notably, a
        nonexistent tournament returns 400 here, not the 404 that
        ``tournament()`` would give for the same id).
        """
        return self.request("POST", f"/tournaments/{tournament_id}/join")

    def leave_tournament(self, tournament_id: str) -> dict:
        """``POST /tournaments/{id}/leave``. No request body.

        Idempotent when not a member. Only succeeds while the tournament is
        still ``waiting`` — once it has started there is no code path to
        leave it, and this always fails with
        ``PlatformError(error_code="leave_failed")``.
        """
        return self.request("POST", f"/tournaments/{tournament_id}/leave")

    def game(self, session_id: str, game_server_url: str) -> "GameSession":
        """Open a handle to one already-known GameAPI match.

        Both ``session_id`` and ``game_server_url`` must be supplied
        explicitly — this milestone does not discover them (that's the
        control plane's ``/competitions``/``/tournaments`` job, not yet
        implemented here). ``game_server_url`` is accepted with or without a
        scheme: the real control plane hands it back as a bare host (see
        Agent_ACP/backend/src/services/gameAPIService.ts's
        ``getGameAPIServerUrl``), so ``http://`` is prepended automatically
        if missing, matching the platform's own documented normalization
        rule (backend/skill/03-competitions.md).
        """
        from .game import GameSession  # local import: game.py imports this module

        return GameSession(self, session_id=session_id, game_server_url=game_server_url)

    def mcp_game(self, session_id: str, game_server_url: str) -> "MCPGameSession":
        """Open a handle to one already-known match, played through MCP —
        the production gameplay transport (see ``altruagent.mcp_game``).
        ``Match.game()`` calls this for you with a lazily-resolved
        ``game_server_url``; call this directly only if you already have
        both values some other way.
        """
        from .mcp_game import MCPGameSession  # local import: mirrors .game() above

        return MCPGameSession(self, session_id=session_id, game_server_url=game_server_url)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Send an authenticated request using this agent's JWT.

        ``url`` may be a path relative to the control plane (e.g.
        ``"/auth/agent/me"``) or a full absolute URL on a different host
        (e.g. a GameAPI ``game_server_url``) — httpx uses an absolute URL
        as-is regardless of this client's configured base URL. Either way,
        the same JWT and the same one-retry-after-401 behavior apply: there
        is only one authentication system for this client, not one per host.

        Returns the parsed JSON response body. Raises ``AuthenticationError``
        if the request is still unauthenticated after one re-login, or
        ``PlatformError`` for any other non-2xx response or network failure.
        """
        if self._access_token is None:
            self.login()

        response = self._send(method, url, **kwargs)

        if response.status_code == 401:
            # No refresh tokens exist on this platform: re-login with the API
            # key once and retry once. If that still fails, stop — do not loop.
            self.login()
            response = self._send(method, url, **kwargs)

        if response.status_code == 401:
            parsed = _parse_error_body(response)
            message = parsed["detail"] or parsed["error"] or "Not authenticated."
            raise AuthenticationError(
                message,
                status_code=401,
                error_code=parsed["error"],
                detail=parsed["detail"],
            )
        if response.status_code >= 400:
            parsed = _parse_error_body(response)
            message = (
                parsed["detail"]
                or parsed["error"]
                or f"Request failed with status {response.status_code}."
            )
            raise PlatformError(
                message,
                status_code=response.status_code,
                error_code=parsed["error"],
                detail=parsed["detail"],
                next_action=parsed["next_action"],
            )

        return _parse_json_body(response)

    # -- internals --------------------------------------------------------

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", None) or {})
        headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            return self._http.request(method, url, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise PlatformError(f"Could not reach {url}: {exc}", status_code=None) from exc

