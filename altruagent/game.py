"""GameSession — a handle to one already-known GameAPI match.

Wraps five endpoints, verified directly against Agent_ACP's actual route
implementations:

- ``GET /games/{session_id}``         (gameapi/routes/games.py, ``get_game_state``)
- ``POST /games/{session_id}/step``   (gameapi/routes/games.py, ``step_game``)
- ``POST /games/{session_id}/resign`` (gameapi/routes/games.py, ``resign_game``)
- ``POST /games/{session_id}/message`` with ``type="chat"``/``"terminate"``
  (gameapi/routes/messages.py, ``send_message`` — shared by both
  ``send_message()`` and ``terminate_messaging()`` below, which just fix the
  request body's ``type``)

All require the same bearer JWT the control plane issued (GameAPI validates
it independently via JWKS — see gameapi/src/gameapi/auth/jwt_validator.py —
but it's the identical Supabase token). ``GameSession`` never manages its
own auth: every call goes through ``AltruAgentClient.request()``, which
already knows how to log in and retry once on a 401. Spectating, replay,
and cancel are still out of scope — those aren't part of the contestant
gameplay loop.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit
from typing import TYPE_CHECKING

from .errors import ConfigurationError
from .models import GameState

if TYPE_CHECKING:
    from .client import AltruAgentClient

_HAS_SCHEME = re.compile(r"^https?://", re.IGNORECASE)


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_local_host(host_and_path: str) -> bool:
    host = urlsplit(f"//{host_and_path}").hostname or ""
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def _normalize_game_server_url(game_server_url: str) -> str:
    """Apply the platform's own game_server_url normalization rule.

    The control plane may return this as a bare host rather than a full URL;
    Agent_ACP's backend/skill/03-competitions.md says to prepend
    ``https://`` in that case. Local development hosts (``localhost``,
    ``127.0.0.1``, ``[::1]``) get ``http://`` instead, since a local GameAPI
    doesn't serve TLS. An explicit scheme is always kept as given.

    Plain ``http://`` to the deployed game server only answers with a
    redirect, after the request (and its bearer token) already went out
    unencrypted — so defaulting a remote host to ``http://`` is never right.
    """
    value = (game_server_url or "").strip()
    if not value:
        raise ConfigurationError("game_server_url must not be empty.")
    if not _HAS_SCHEME.match(value):
        scheme = "http" if _is_local_host(value) else "https"
        value = f"{scheme}://{value}"
    return value.rstrip("/")


class GameSession:
    """One concrete GameAPI match, identified by ``session_id`` +
    ``game_server_url``. Construct via ``AltruAgentClient.game(...)`` rather
    than directly.
    """

    def __init__(self, client: "AltruAgentClient", *, session_id: str, game_server_url: str) -> None:
        if not session_id:
            raise ConfigurationError("session_id must not be empty.")
        self._client = client
        self.session_id = session_id
        self.game_server_url = _normalize_game_server_url(game_server_url)

    def _url(self, suffix: str = "") -> str:
        return f"{self.game_server_url}/games/{self.session_id}{suffix}"

    def state(self) -> GameState:
        """``GET /games/{session_id}`` — the current state, from this agent's
        point of view (``legal_actions`` is empty unless it's this agent's turn).
        """
        data = self._client.request("GET", self._url())
        return GameState.from_dict(data if isinstance(data, dict) else {})

    def step(self, action: int) -> GameState:
        """``POST /games/{session_id}/step`` — submit a move.

        ``action`` must be one of the integers in the latest ``state().legal_actions``
        — the server re-validates this regardless (400 ``invalid_action`` if not).
        """
        data = self._client.request("POST", self._url("/step"), json={"action": action})
        return GameState.from_dict(data if isinstance(data, dict) else {})

    def resign(self) -> GameState:
        """``POST /games/{session_id}/resign`` — concede the game.

        Allowed at any time, in any phase, regardless of whose turn it is.
        """
        data = self._client.request("POST", self._url("/resign"), json={})
        return GameState.from_dict(data if isinstance(data, dict) else {})

    def send_message(self, content: str, recipients: list[int] | None = None) -> GameState:
        """``POST /games/{session_id}/message`` with ``type="chat"``.

        Only valid while ``state().phase == "messaging"`` and this agent
        hasn't already sent ``terminate`` this round — the server 409s
        (``wrong_phase``) otherwise. ``recipients`` empty/``None`` broadcasts
        to every other player; a single index sends a targeted p2p message
        (the server currently rejects 2+ recipients — p2group is gated, see
        Agent_ACP/gameapi/src/gameapi/routes/messages.py's
        ``_validate_recipients``). Word/length/quota limits are enforced
        server-side and surface as ``PlatformError``.
        """
        body = {"type": "chat", "content": content, "recipients": list(recipients or [])}
        data = self._client.request("POST", self._url("/message"), json=body)
        return GameState.from_dict(data if isinstance(data, dict) else {})

    def terminate_messaging(self) -> GameState:
        """``POST /games/{session_id}/message`` with ``type="terminate"`` —
        vote to end the current messaging round.

        Idempotent: re-terminating in the same round is a no-op server-side
        (see ``messages.py``'s ``send_message`` handler). Once every active
        player has terminated, the phase flips to ``moving``; until then,
        the next state's ``next_actions`` reports ``wait_for_opponent``.
        """
        data = self._client.request(
            "POST", self._url("/message"), json={"type": "terminate", "recipients": []}
        )
        return GameState.from_dict(data if isinstance(data, dict) else {})
