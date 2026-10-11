"""MCPGameSession — a handle to one already-known match, played through
Agent_ACP's MCP gameplay tools rather than the REST GameAPI.

This is the production gameplay transport: ``run_match``/``python -m agent``
always play through this class (see ``altruagent.runner``). ``GameSession``
(``game.py``, REST) remains available only as a lower-level debug/manual-
testing tool (``scripts/check_game.py``) — kept unmodified, not used here.

Method names mirror Agent_ACP's MCP tool names directly (``get_game_state``,
``wait_for_update``, ``get_legal_actions``, ``play_action``, ``send_message``,
``get_messages``, ``resign``, ``get_result`` —
gameapi/src/gameapi/mcp_server/server.py) rather
than REST's ``state()``/``step()``/``resign()`` shape, since the whole point
is that this class is a thin, honest transport — all orchestration (when to
call what, how to interpret the result) lives in ``altruagent.runner``, not
here.

Auth/retry/error-parsing all go through the exact same
``AltruAgentClient.request()`` this session's REST counterpart uses (via
``altruagent.mcp_transport.call_tool``) — there is only one HTTP/auth stack
in this SDK, not one per transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import ConfigurationError
from .game import _normalize_game_server_url
from .mcp_transport import call_tool
from .models import GameState

if TYPE_CHECKING:
    from .client import AltruAgentClient


class MCPGameSession:
    """One concrete match, identified by ``session_id`` + ``game_server_url``,
    played through MCP. Construct via ``AltruAgentClient.mcp_game(...)`` or
    ``Match.game()`` rather than directly.
    """

    def __init__(self, client: "AltruAgentClient", *, session_id: str, game_server_url: str) -> None:
        if not session_id:
            raise ConfigurationError("session_id must not be empty.")
        self._client = client
        self.session_id = session_id
        # Same host REST's game_server_url already resolves to (confirmed:
        # Agent_ACP mounts both REST routers and the MCP server on one
        # FastAPI app, gameapi/src/gameapi/app.py) — only the path differs.
        self.game_server_url = _normalize_game_server_url(game_server_url)

    def _mcp_url(self) -> str:
        return f"{self.game_server_url}/mcp"

    def _call(self, tool: str, arguments: dict) -> dict:
        return call_tool(self._client, self._mcp_url(), tool, arguments)

    def get_state(self) -> GameState:
        """``get_game_state`` — this agent's current view of the match.
        When this agent can act, the payload embeds ``legal_actions`` for the
        same ``state_version``, parsed onto ``GameState.legal_actions`` — no
        separate ``get_legal_actions`` call needed. ``returns``/
        ``termination_reason`` are still absent (see ``get_result``, only
        relevant once terminal).
        """
        data = self._call("get_game_state", {"session_id": self.session_id})
        return GameState.from_mcp_state(data)

    def wait_for_update(
        self,
        *,
        since_version: int,
        since_message_seq: int | None = None,
        since_is_current_actor: bool | None = None,
        since_phase: str | None = None,
        timeout_seconds: float | None = None,
    ) -> GameState:
        """``wait_for_update`` — server-side long-poll. Returns as soon as
        ``state_version`` differs from ``since_version``, a message newer than
        ``since_message_seq`` is visible, whose-turn/phase changes, or the
        game ends; otherwise after ``timeout_seconds`` (server default 20,
        max 25). Same payload as ``get_state()`` plus ``updated`` (in
        ``raw``), so the result can be used directly as the new state.

        ``since_is_current_actor``/``since_phase`` are what the caller last
        saw: whose turn it is and the phase can change without
        ``state_version`` moving, and without them the server compares
        against its own reading when the call arrives, missing a change that
        landed just before. Servers predating these parameters ignore them.
        """
        arguments: dict = {"session_id": self.session_id, "since_version": since_version}
        if since_message_seq is not None:
            arguments["since_message_seq"] = since_message_seq
        if since_is_current_actor is not None:
            arguments["since_is_current_actor"] = since_is_current_actor
        if since_phase is not None:
            arguments["since_phase"] = since_phase
        if timeout_seconds is not None:
            arguments["timeout_seconds"] = timeout_seconds
        return GameState.from_mcp_state(self._call("wait_for_update", arguments))

    def get_game_config(self, game_type: str) -> dict:
        """``get_game_config`` — the game's static reference (rules, action or
        order formats, timeouts, maps). The runner fetches it once per
        real-time match for ``DecisionContext.game_config``.
        """
        return self._call("get_game_config", {"game_type": game_type})

    def get_legal_actions(self) -> dict:
        """``get_legal_actions`` — the actions available right now, plus the
        ``state_version`` to use for the next ``play_action`` call. Usually
        unnecessary (``get_state()`` already embeds it when this agent can
        act); the runner only calls it as a fallback when the server omitted
        it (the state moved between the server's two reads). Returns the raw
        dict (``{"session_id","state_version","actions"}``).
        """
        return self._call("get_legal_actions", {"session_id": self.session_id})

    def play_action(
        self,
        *,
        action_id: str | None = None,
        action: dict | None = None,
        state_version: int,
        reasoning_summary: str | None = None,
    ) -> dict:
        """``play_action`` — submit a move. Exactly one of ``action_id``
        (matched from ``get_legal_actions``) or ``action`` (a structured
        payload for constructive actions, e.g. Pokémon's ``submit_team``)
        should be given — mirrors the tool's own ``action_id``/``action``
        precedence (``action`` wins if both are given, per
        gameapi/src/gameapi/mcp_server/server.py's ``play_action``).

        Returns the raw ``{"accepted","session_id","state_version","status"}``
        dict. While the game continues it also carries ``state``: this
        agent's ``get_game_state`` payload after the move (with
        ``legal_actions`` if it acts again), which the runner uses directly
        instead of re-fetching.
        """
        arguments: dict = {"session_id": self.session_id, "state_version": state_version}
        if action is not None:
            arguments["action"] = action
        if action_id is not None:
            arguments["action_id"] = action_id
        if reasoning_summary is not None:
            arguments["reasoning_summary"] = reasoning_summary
        return self._call("play_action", arguments)

    def send_message(self, *, message_type: str, content: str | None = None, recipients: list[int] | None = None) -> dict:
        """``send_message`` — send a chat message (``message_type="chat"``)
        or vote to end the messaging round (``message_type="terminate"``).

        Deliberately takes no ``state_version`` — confirmed against
        ``gameapi/src/gameapi/runtime_adapters/openspiel_adapter.py``'s
        ``send_message``: unlike ``apply_action``, it never checks
        ``state_version`` for staleness, only ``session.phase`` (raising
        ``WrongPhaseError`` -> MCP's ``STALE_STATE``-adjacent race handling
        in the runner covers this the same way).
        """
        arguments = {
            "session_id": self.session_id,
            "message_type": message_type,
            "content": content,
            "recipients": recipients,
        }
        return self._call("send_message", arguments)

    def get_messages(self, *, since: int = -1) -> dict:
        """``get_messages`` — full/incremental transcript. Not needed for the
        normal ``choose_message`` flow (``get_state()``'s own ``new_messages``
        already carries what's arrived since the last phase flip — confirmed
        directly in ``openspiel_adapter.get_state``); available for a
        contestant that wants more history than that.
        """
        return self._call("get_messages", {"session_id": self.session_id, "since": since})

    def resign(self) -> dict:
        """``resign`` — concede the game. Returns the same shape as
        ``get_result()`` (confirmed identical dict literal in
        ``openspiel_adapter.py``). In most games the resign ends the game, so
        this is the final result and no follow-up call is needed. In Werewolf
        it takes only you out and the game goes on: ``is_terminal`` is false
        and ``eliminated`` true, and ``get_result()`` gives the final result
        once the game ends.
        """
        return self._call("resign", {"session_id": self.session_id})

    def get_result(self) -> dict:
        """``get_result`` — final outcome. Only meaningful once ``is_terminal``
        — ``get_state()``/``play_action()`` don't carry ``returns``/
        ``termination_reason`` themselves, confirmed narrower than REST's
        single-response terminal state.
        """
        return self._call("get_result", {"session_id": self.session_id})
