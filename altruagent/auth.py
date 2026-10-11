"""Authentication strategies — how an ``AltruAgentClient`` obtains the bearer
token it sends on every control-plane, REST GameAPI, and MCP request.

The client owns the token cache and the platform's one-retry-after-401
policy (see ``client.py``); a strategy only answers "give me a fresh token
now". The client calls ``login()`` lazily before the first request and again,
exactly once, after a 401 — so the retry policy stays in one place, and
gameplay code (``MCPGameSession``, ``run_game``) never knows which strategy
is in use.

- ``ApiKeyAuth`` — RETIRED: a platform agent's long-lived ``sk_agent_...``
  key, exchanged for a JWT via ``POST /auth/agent/login``. The platform turned
  platform agents off and answers HTTP 410 ``platform_agents_retired``; this
  strategy then raises ``AuthenticationError`` with a notice pointing to the
  Official Agent Key.
- ``SeatGrantAuth`` — RETIRED: one self-hosted Testing seat, claimed with a
  one-time ``seatclaim_...`` token via
  ``POST /tournament/agent/test-matches/seats/claim``. Testing now uses the
  Official Agent Key too, and the platform answers HTTP 410
  ``claim_codes_retired``, which this strategy reports with the same kind of
  notice.

Agents connect with ``altruagent.official`` (``OfficialAgentClient``,
``OfficialSeatAuth``), run by ``python -m agent --tournament``/``--match``.
"""

from __future__ import annotations

import secrets
from typing import Protocol

import httpx

from ._responses import _parse_error_body, _parse_json_body
from .errors import AuthenticationError, PlatformError
from .models import SeatGrant
from .notices import CLAIM_CODES_RETIRED_NOTICE, PLATFORM_KEY_RETIRED_NOTICE

SEAT_CLAIM_PATH = "/tournament/agent/test-matches/seats/claim"
SEAT_CLAIM_TOKEN_PREFIX = "seatclaim_"


class AuthStrategy(Protocol):
    def login(self, http: httpx.Client, control_url: str) -> str:
        """Obtain a fresh bearer token, or raise ``AuthenticationError``
        (``PlatformError`` if the control plane can't be reached at all).
        """
        ...


class ApiKeyAuth:
    """RETIRED. A platform agent's API key -> ``POST /auth/agent/login`` -> JWT.

    The platform issues no refresh token, so "renewing" is just logging in
    again with the same key. The platform now answers 410
    ``platform_agents_retired``, reported as ``AuthenticationError`` with
    ``altruagent.notices.PLATFORM_KEY_RETIRED_NOTICE``.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __repr__(self) -> str:
        return "ApiKeyAuth(api_key=<redacted>)"

    def login(self, http: httpx.Client, control_url: str) -> str:
        try:
            response = http.post("/auth/agent/login", json={"api_key": self._api_key})
        except httpx.RequestError as exc:
            raise PlatformError(
                f"Could not reach the control plane at {control_url}: {exc}",
                status_code=None,
            ) from exc

        if response.status_code != 200:
            parsed = _parse_error_body(response)
            message = parsed["detail"] or parsed["error"] or "Login failed."
            if response.status_code == 410 or parsed["error"] == "platform_agents_retired":
                message = PLATFORM_KEY_RETIRED_NOTICE
            raise AuthenticationError(
                message,
                status_code=response.status_code,
                error_code=parsed["error"],
                detail=parsed["detail"],
            )

        body = _parse_json_body(response)
        token = body.get("access_token") if isinstance(body, dict) else None
        if not token:
            raise AuthenticationError(
                "Login response did not include an access_token.",
                status_code=response.status_code,
            )
        return token


class SeatClaimError(AuthenticationError):
    """Claiming (or renewing) a Testing seat failed. ``error_code`` is the
    platform's machine code (``invalid_claim_token``, ``seat_already_claimed``,
    ``claim_expired``, ``match_not_claimable``, ``match_not_ready``,
    ``rate_limited``, ...) when the platform sent one; the message is already
    written for a human.

    ``retry_after_seconds`` is the platform's own retry hint, when it sent one
    (``match_not_ready``: the match's open seats are still being filled).
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_seconds = retry_after_seconds


# Messages for the platform's documented claim error codes. Deliberately
# replace the backend's own `detail` rather than echoing it: these are what a
# contestant sees in their terminal.
_CLAIM_ERROR_MESSAGES = {
    "invalid_claim_token": (
        "That seat claim token isn't valid. Copy the command for this seat "
        "again from the tournament dashboard."
    ),
    "seat_already_claimed": (
        "This seat is already claimed by another running agent. Each seat can "
        "be claimed by exactly one process — use a different seat's command."
    ),
    "claim_expired": (
        "This seat claim token expired before it was used. Reissue the seat's "
        "claim token from the tournament dashboard."
    ),
    "match_not_claimable": (
        "This test match is no longer active (it finished, failed, or was "
        "cancelled), so its seats can't be claimed."
    ),
    "rate_limited": "Too many seat claim attempts. Wait a minute and try again.",
    "match_not_ready": (
        "This test match is still waiting for its open seats to be filled. "
        "Keep this process running; it claims the seat as soon as the match fills."
    ),
    "claim_codes_retired": CLAIM_CODES_RETIRED_NOTICE,
}


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The platform's ``retry_after_seconds`` hint, if the body has a usable one."""
    try:
        body = response.json()
    except ValueError:
        return None
    value = body.get("retry_after_seconds") if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


class SeatGrantAuth:
    """RETIRED (the platform answers 410 ``claim_codes_retired``; Testing now
    uses the Official Agent Key). Authorization for exactly one self-hosted
    Testing seat.

    Holds the one-time ``claim_token``, a ``claim_key`` generated here (256
    bits from ``secrets``, never shown, never written anywhere) that stays
    fixed for this object's lifetime, and the latest ``SeatGrant``. All three
    live only in memory. Build one per process and never share it: a second
    ``SeatGrantAuth`` for the same token has a different key, and the
    platform rejects it with ``seat_already_claimed``.
    """

    def __init__(self, claim_token: str) -> None:
        claim_token = (claim_token or "").strip()
        if not claim_token.startswith(SEAT_CLAIM_TOKEN_PREFIX):
            raise SeatClaimError(
                f"That doesn't look like a seat claim token (expected one "
                f"starting with {SEAT_CLAIM_TOKEN_PREFIX!r}).",
                error_code="invalid_claim_token",
            )
        self._claim_token = claim_token
        self._claim_key = secrets.token_urlsafe(32)
        self._grant: SeatGrant | None = None

    def __repr__(self) -> str:
        return f"SeatGrantAuth(grant={self._grant!r})"

    @property
    def grant(self) -> SeatGrant | None:
        """The most recent grant, or ``None`` before the first claim."""
        return self._grant

    def login(self, http: httpx.Client, control_url: str) -> str:
        """Claim the seat (first call) or renew its authorization (every
        later call) — the same request either way. One attempt per call; the
        client's one-retry-after-401 policy decides when to call again.
        """
        renewing = self._grant is not None
        try:
            response = http.post(
                SEAT_CLAIM_PATH,
                json={"claim_token": self._claim_token, "claim_key": self._claim_key},
            )
        except httpx.RequestError as exc:
            raise PlatformError(
                f"Could not reach the control plane at {control_url}: {exc}",
                status_code=None,
            ) from exc

        if response.status_code != 200:
            raise self._claim_error(response, renewing=renewing)

        body = _parse_json_body(response)
        grant = SeatGrant.from_dict(body if isinstance(body, dict) else {})
        if not (grant.access_token and grant.game_session_id and grant.gameapi_server_url and grant.agent_id):
            raise SeatClaimError(
                "The seat claim response was missing required fields.",
                status_code=response.status_code,
            )
        if renewing and (grant.seat_id, grant.game_session_id) != (self._grant.seat_id, self._grant.game_session_id):
            raise SeatClaimError(
                "Seat renewal returned a different seat than the one originally claimed.",
                status_code=response.status_code,
            )
        self._grant = grant
        return grant.access_token

    @staticmethod
    def _claim_error(response: httpx.Response, *, renewing: bool) -> SeatClaimError:
        parsed = _parse_error_body(response)
        code = parsed["error"]
        message = _CLAIM_ERROR_MESSAGES.get(code) or (
            f"Seat claim failed (HTTP {response.status_code}"
            + (f", {code}" if code else "")
            + ")."
        )
        if renewing:
            message = f"Could not renew this seat's GameAPI authorization: {message}"
        return SeatClaimError(
            message,
            status_code=response.status_code,
            error_code=code,
            detail=parsed["detail"],
            retry_after_seconds=_retry_after_seconds(response),
        )
