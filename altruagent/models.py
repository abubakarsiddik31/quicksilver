"""Response models for the AltruAgent SDK.

Models are deliberately tolerant of extra/unknown fields from the backend —
new fields the platform adds later should never break parsing. ``raw`` keeps
the full server response for anything not (yet) promoted to a typed
attribute, with one exception: fields the SDK never surfaces at all (see
``Agent`` below), regardless of what the live endpoint happens to return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import AltruAgentClient
    from .game import GameSession
    from .mcp_game import MCPGameSession

# The real GET /auth/agent/me and POST /auth/human/claim responses currently
# include these (they're the agent's raw DB row — see
# Agent_ACP/backend/src/models/agent.ts and index.ts's `/auth/agent/me`
# handler, which spreads the row as-is). They're one-way hashes, not the
# actual api_key/claim_token secrets, but contestant code has no reason to
# ever see them, so they're stripped before `Agent.raw` is populated.
_SENSITIVE_FIELDS = ("api_key_hash", "claim_token_hash")


@dataclass
class Agent:
    """The authenticated agent, as returned by ``GET /auth/agent/me``."""

    id: str
    name: str
    status: str  # "unclaimed" or "claimed"
    description: str | None = None
    claimed_by_user_id: str | None = None
    claimed_at: str | None = None
    created_at: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_claimed(self) -> bool:
        return self.status == "claimed"

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        raw = {k: v for k, v in data.items() if k not in _SENSITIVE_FIELDS}
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            description=data.get("description"),
            claimed_by_user_id=data.get("claimed_by_user_id"),
            claimed_at=data.get("claimed_at"),
            created_at=data.get("created_at"),
            raw=raw,
        )


@dataclass
class NextAction:
    """Machine-readable "what can I do right now" guidance — supplementary
    context only, never the runner's primary turn-detection mechanism (see
    ``altruagent.runner``, which drives off ``is_current_actor``/``phase``/
    ``is_terminal`` instead).

    REST shapes this as ``{"action", "endpoint", "hint", "required_fields"}``
    (gameapi/src/gameapi/models/responses.py's ``NextAction``); MCP shapes it
    as ``{"tool", "hint"}`` (gameapi/src/gameapi/mcp_server/catalog.py's
    ``game_state_payload``/``play_action_payload``/etc.) — this model reads
    either key name into the same ``action`` field so callers don't need to
    know which transport produced it.
    """

    action: str
    hint: str
    endpoint: str | None = None
    required_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "NextAction":
        return cls(
            action=data.get("action") or data.get("tool", ""),
            hint=data.get("hint", ""),
            endpoint=data.get("endpoint"),
            required_fields=list(data.get("required_fields") or []),
        )


@dataclass
class PlayerRef:
    """A player reference as GameAPI serializes it: just a display name.

    See gameapi/src/gameapi/domain/__init__.py's ``Player`` dataclass and
    ``GameStateResponse.current_player: Player | None``.
    """

    name: str


@dataclass
class Message:
    """A single messaging-phase message, as GameAPI serializes it. Mirrors
    gameapi/src/gameapi/models/responses.py's ``MessageResponse`` exactly
    (``index``, ``sender``, ``recipients``, ``content``, ``type``,
    ``sent_at``, ``move_index``). ``type`` is ``"chat"`` or ``"terminate"``
    (see gameapi/src/gameapi/domain/__init__.py's ``MessageType``);
    ``recipients`` empty means broadcast, one entry means a targeted p2p
    message (the server rejects 2+ today — p2group is gated).
    """

    index: int
    sender: int
    recipients: list[int]
    content: str
    type: str
    sent_at: str
    move_index: int

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            # Platform messages carry `seq` (the MCP envelope); `index` is the
            # legacy alias. Prefer `seq`, which wait_for_update compares on.
            index=data.get("seq", data.get("index", 0)),
            sender=data.get("sender", 0),
            recipients=list(data.get("recipients") or []),
            content=data.get("content", ""),
            type=data.get("type", "chat"),
            sent_at=data.get("sent_at", ""),
            move_index=data.get("move_index", 0),
        )


@dataclass
class LegalAction:
    """One action a contestant may currently take, as MCP's ``get_legal_actions``
    tool returns it (and as REST's ``legal_actions``/``legal_actions_str`` pair
    is synthesized into, for uniformity — see ``GameState.from_dict`` below).

    ``action_id`` is always a string and is the only thing ``play_action``
    needs back — for OpenSpiel-family games it's a stringified int (e.g.
    ``"0"``); for structured RuntimeAdapter games (e.g. Pokémon) it can be
    anything the adapter defines (``"move:0"``, ``"switch:1"``,
    ``"draft_pick:<card_id>"``, ``"submit_team"``). ``input`` is the adapter's
    own structured payload for this action (empty ``{}`` for a REST-synthesized
    entry, since REST has nothing equivalent) — for constructive actions like
    Pokémon's ``submit_team`` it's a template the server expects a contestant
    to build a real value from, not something to submit verbatim (see
    ``altruagent.runner``'s dict-decision escape hatch).
    """

    action_id: str
    label: str | None
    input: dict
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "LegalAction":
        return cls(
            action_id=str(data.get("action_id", "")),
            label=data.get("label"),
            input=dict(data.get("input") or {}),
            raw=data,
        )


@dataclass
class GameState:
    """A game state, normalized across both the REST GameAPI
    (``GET/POST /games/{session_id}...`` — see ``GameSession``, kept only as
    a lower-level debug/manual-testing tool) and MCP's gameplay tools (see
    ``MCPGameSession``, what ``python -m agent`` actually plays through).

    ``legal_actions`` is always ``list[LegalAction]`` regardless of which
    transport produced it: MCP's ``get_legal_actions`` already returns that
    shape; REST's separate ``legal_actions``/``legal_actions_str`` int/str
    lists are zipped into synthetic ``LegalAction`` entries (``input={}``) so
    contestant code never has to care which transport it's talking to. The
    universal pattern ``return state.legal_actions[0]`` works unchanged either
    way.

    ``state_version``/``is_current_actor`` are MCP concepts with no REST
    equivalent (default to ``0``/``None`` when parsed from a REST response,
    since REST has no optimistic-concurrency counter and infers turn-taking
    from ``current_player`` instead — preserved on this model only for the
    debug path, not used by the MCP-driven runner).

    Richer per-game fields (repeated_pd's round history, Avalon's
    ``avalon_*`` fields, Pokémon's draft/roster detail) are not individually
    modeled — always available via ``raw``, the complete unmodified response
    for whichever tool/endpoint produced this state.

    ``final_result`` is the server's whole ``get_result`` (or ``resign``)
    answer once the game is over (``your_return``, ``winner_agent_id``, the
    game's own per-player ``result`` where it has one), and ``None`` before.
    """

    session_id: str
    game_name: str
    status: str
    observation: str
    current_player: PlayerRef | None
    legal_actions: list[LegalAction]
    is_terminal: bool
    is_current_actor: bool | None
    state_version: int
    returns: dict[str, float] | None
    move_count: int
    termination_reason: str | None
    messaging_enabled: bool
    phase: str
    next_actions: list[NextAction]
    new_messages: list[Message]
    terminated_messaging: list[int]
    messaging_mode: str
    raw: dict = field(default_factory=dict, repr=False)
    final_result: dict | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        """Parse a REST ``GameStateResponse`` (``GameSession`` — the debug
        path). Synthesizes ``LegalAction`` entries from the separate
        ``legal_actions``/``legal_actions_str`` int/str lists REST returns.
        """
        current_player_data = data.get("current_player")
        current_player = (
            PlayerRef(name=current_player_data["name"])
            if current_player_data
            else None
        )
        next_actions = [
            NextAction.from_dict(a) for a in (data.get("next_actions") or [])
        ]
        new_messages = [
            Message.from_dict(m) for m in (data.get("new_messages") or [])
        ]
        raw_actions = list(data.get("legal_actions") or [])
        raw_labels = list(data.get("legal_actions_str") or [])
        legal_actions = [
            LegalAction(
                action_id=str(action),
                label=raw_labels[i] if i < len(raw_labels) else None,
                input={},
                raw={"action": action, "label": raw_labels[i] if i < len(raw_labels) else None},
            )
            for i, action in enumerate(raw_actions)
        ]
        return cls(
            session_id=data.get("session_id", ""),
            game_name=data.get("game_name", ""),
            status=data.get("status", "unknown"),
            observation=data.get("observation", ""),
            current_player=current_player,
            legal_actions=legal_actions,
            is_terminal=bool(data.get("is_terminal", False)),
            is_current_actor=None,
            state_version=0,
            returns=data.get("returns"),
            move_count=data.get("move_count", 0),
            termination_reason=data.get("termination_reason"),
            messaging_enabled=bool(data.get("messaging_enabled", False)),
            phase=data.get("phase", "moving"),
            next_actions=next_actions,
            new_messages=new_messages,
            terminated_messaging=list(data.get("terminated_messaging") or []),
            messaging_mode=data.get("messaging_mode", "per_move"),
            raw=data,
        )

    @classmethod
    def from_mcp_state(
        cls,
        state: dict,
        *,
        legal_actions: dict | None = None,
        result: dict | None = None,
    ) -> "GameState":
        """Parse an MCP ``get_game_state``/``wait_for_update`` result (the
        production path — see ``MCPGameSession``). ``legal_actions`` defaults
        to the one the server embeds in the state when this agent can act;
        pass a ``get_legal_actions`` result to override it (the runner's
        fallback when the server omitted it). ``result`` merges in a
        ``get_result`` result (only fetched once ``is_terminal`` — MCP's
        ``get_game_state`` doesn't embed ``returns``/``termination_reason``
        the way REST's does).

        Confirmed field names directly against
        ``gameapi/src/gameapi/runtime_adapters/{openspiel_adapter,pokemon_adapter}.py``:
        both adapters return the same generic keys (``session_id``,
        ``state_version``, ``observation``, ``phase``, ``is_terminal``,
        ``is_current_actor``, ``status``) regardless of which one is behind a
        given session — this is what lets one runner drive both without any
        per-game branching.
        """
        current_actor = state.get("current_actor")
        current_player = (
            PlayerRef(name=str(current_actor.get("agent_id")))
            if isinstance(current_actor, dict) and current_actor.get("agent_id")
            else None
        )
        if legal_actions is None and isinstance(state.get("legal_actions"), dict):
            legal_actions = state["legal_actions"]
        actions = list((legal_actions or {}).get("actions") or [])
        parsed_actions = [LegalAction.from_dict(a) for a in actions]
        new_messages = [
            Message.from_dict(m) for m in (state.get("new_messages") or [])
        ]
        result = result or {}
        # `result` (get_result/resign) is authoritative for is_terminal/status
        # when present — it's fetched precisely because `state` (an earlier
        # get_game_state, or the pre-resign state) may be stale on exactly
        # these two fields (e.g. built right before resigning, when the
        # match wasn't terminal yet).
        is_terminal = bool(result.get("is_terminal", state.get("is_terminal", False)))
        status = result.get("status", state.get("status", "unknown"))
        return cls(
            session_id=state.get("session_id", ""),
            game_name=state.get("game_type", ""),
            status=status,
            observation=str(state.get("observation", "")),
            current_player=current_player,
            legal_actions=parsed_actions,
            is_terminal=is_terminal,
            is_current_actor=state.get("is_current_actor"),
            state_version=int(
                (legal_actions or {}).get("state_version", state.get("state_version", 0))
            ),
            returns=result.get("returns"),
            move_count=int(state.get("state_version", 0)),
            termination_reason=result.get("termination_reason"),
            messaging_enabled=bool(state.get("messaging_enabled", False)),
            phase=state.get("phase", "moving"),
            next_actions=[
                NextAction.from_dict(a) for a in (state.get("next_actions") or [])
            ],
            new_messages=new_messages,
            terminated_messaging=list(state.get("terminated_messaging") or []),
            messaging_mode=state.get("messaging_mode", "per_move"),
            raw=state,
            final_result=dict(result) if result else None,
        )


@dataclass
class Match:
    """A contestant-facing view over one Competition row, as returned by
    ``GET /agents/me/sessions`` (verified against Agent_ACP
    backend/src/db/competitions.ts's ``getCompetitionsForAgent`` and
    services/competitionService.ts's ``listAgentSessions``). Called "Match",
    not "Assignment" — the backend has no such concept; this simply wraps a
    raw ``competitions`` table row.

    ``game_server_url`` is never present on this endpoint's rows (confirmed —
    it is not a stored column anywhere; it's computed only by
    ``GET /competitions/{id}`` and ``GET /tournaments/{id}``). It starts
    ``None`` here and is resolved lazily, only when ``game()`` is actually
    called — see ``game()`` below.
    """

    session_id: str
    status: str
    game_type: str | None = None
    tournament_id: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    game_server_url: str | None = None
    raw: dict = field(default_factory=dict, repr=False)
    _client: Any = field(default=None, repr=False, compare=False, init=False)

    @classmethod
    def from_dict(cls, data: dict, *, client: "AltruAgentClient | None" = None) -> "Match":
        match = cls(
            session_id=data.get("session_id", ""),
            status=data.get("status", "unknown"),
            game_type=data.get("game_type"),
            tournament_id=data.get("tournament_id"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            raw=data,
        )
        match._client = client
        return match

    def _resolve_game_server_url(self) -> str:
        """Shared lazy-resolution logic for both ``game()`` and
        ``rest_game()`` — MCP is mounted on the same host REST's
        ``game_server_url`` already points at, so one resolution serves both
        transports; only the path suffix differs (see each method below).

        - If ``game_server_url`` was already resolved (cached from a prior
          call on this same ``Match`` instance), this makes no network
          request at all.
        - If not, and ``status == "in_progress"``, this makes exactly one
          request — ``GET /competitions/{session_id}`` (the same endpoint
          that computes ``game_server_url`` for a single competition) — and
          caches the result on this instance so repeated calls don't repeat
          the lookup.
        - A ``waiting`` match has no GameAPI session to open yet, and a
          ``completed`` one no longer has a playable one; both raise
          ``ValueError`` immediately, with no network request.
        """
        if self._client is None:
            raise ValueError(
                "This Match has no client attached (it wasn't returned by "
                "AltruAgentClient.sessions()), so game_server_url cannot be resolved."
            )

        if self.game_server_url:
            return self.game_server_url

        if self.status != "in_progress":
            raise ValueError(
                f"Match {self.session_id!r} is {self.status!r}, not 'in_progress' — "
                "it has no playable GameAPI session right now."
            )

        data = self._client.request("GET", f"/competitions/{self.session_id}")
        if isinstance(data, dict) and data.get("session_id") not in (None, self.session_id):
            raise ValueError(
                f"GET /competitions/{self.session_id} returned session_id "
                f"{data.get('session_id')!r}, which does not match."
            )
        game_server_url = data.get("game_server_url") if isinstance(data, dict) else None
        if not game_server_url:
            raise ValueError(
                f"GET /competitions/{self.session_id} did not include a game_server_url "
                "even though the match is in_progress."
            )

        self.game_server_url = game_server_url
        return game_server_url

    def game(self) -> "MCPGameSession":
        """Return a playable ``MCPGameSession`` for this match — the
        production gameplay path ``run_match``/``python -m agent`` use.
        Resolves ``game_server_url`` lazily (see ``_resolve_game_server_url``)
        — MCP is mounted on the same host REST's ``game_server_url`` already
        resolves to, so no new discovery step is needed; only the path
        suffix (``/mcp`` vs ``/games/{id}``) differs.
        """
        game_server_url = self._resolve_game_server_url()
        return self._client.mcp_game(session_id=self.session_id, game_server_url=game_server_url)

    def rest_game(self) -> "GameSession":
        """Return a playable REST ``GameSession`` for this match — the
        lower-level debug/manual-testing path (see ``scripts/check_game.py``),
        not used by ``run_match``/``python -m agent``. Same lazy resolution
        as ``game()``.
        """
        game_server_url = self._resolve_game_server_url()
        return self._client.game(session_id=self.session_id, game_server_url=game_server_url)


@dataclass
class AgentSessions:
    """This agent's competition memberships, as returned by
    ``GET /agents/me/sessions``, grouped exactly as the server groups them:
    ``joined_sessions`` -> ``waiting``, ``active_sessions`` -> ``active``,
    ``completed_sessions`` -> ``completed``. Includes both standalone
    competitions and tournament-created child matches (the endpoint does not
    distinguish at the query level — see ``Match.tournament_id``).

    The backend caps this at the 50 most recently joined memberships in
    total, not per group (``getCompetitionsForAgent``'s ``.limit(50)``) — a
    long-lived agent's oldest completed matches can silently drop off before
    its current ones would.
    """

    waiting: list[Match] = field(default_factory=list)
    active: list[Match] = field(default_factory=list)
    completed: list[Match] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict, *, client: "AltruAgentClient | None" = None) -> "AgentSessions":
        def _matches(key: str) -> list[Match]:
            return [Match.from_dict(m, client=client) for m in (data.get(key) or [])]

        return cls(
            waiting=_matches("joined_sessions"),
            active=_matches("active_sessions"),
            completed=_matches("completed_sessions"),
            raw=data,
        )


@dataclass
class TournamentViewer:
    """The calling agent's membership view of a tournament — present only on
    an authenticated ``GET /tournaments/{id}`` (see Agent_ACP
    backend/src/services/tournamentService.ts's ``getTournament``, which only
    builds a ``viewer`` object when the request carried a JWT that resolved
    to an agent).
    """

    agent_id: str | None = None
    is_tournament_participant: bool = False
    active_child_session_ids: list[str] = field(default_factory=list)
    should_join_tournament: bool | None = None
    should_wait_for_child_match: bool | None = None
    next_actions: list[NextAction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "TournamentViewer":
        return cls(
            agent_id=data.get("agent_id"),
            is_tournament_participant=bool(data.get("is_tournament_participant", False)),
            active_child_session_ids=list(data.get("active_child_session_ids") or []),
            should_join_tournament=data.get("should_join_tournament"),
            should_wait_for_child_match=data.get("should_wait_for_child_match"),
            next_actions=[NextAction.from_dict(a) for a in (data.get("next_actions") or [])],
        )


@dataclass
class Tournament:
    """A tournament, as returned by either ``GET /tournaments`` (list — a raw
    ``tournaments`` DB row) or ``GET /tournaments/{id}`` (detail — a curated
    ``compactTournament()`` subset with a *different* field set — see
    Agent_ACP backend/src/services/tournamentService.ts). Both shapes are
    tolerated: only fields useful and common enough to model are typed;
    everything else (list-only fields like ``created_at``/``metadata``, or
    detail-only ``leaderboard``/``participants``) stays reachable via ``raw``.

    There is no ``name`` field anywhere on the backend (confirmed against
    the DB schema) — a tournament is identified only by ``tournament_id`` +
    ``game_type``.

    ``viewer`` is ``None`` unless this came from an authenticated
    ``GET /tournaments/{id}`` call that returned one (never present on a
    ``GET /tournaments`` list entry).
    """

    tournament_id: str
    status: str
    game_type: str | None = None
    max_participants: int | None = None
    current_participants: int | None = None
    max_active_matches: int | None = None
    queue_id: str | None = None
    game_server_url: str | None = None
    viewer: TournamentViewer | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict, *, viewer: dict | None = None) -> "Tournament":
        return cls(
            tournament_id=data.get("tournament_id", ""),
            status=data.get("status", "unknown"),
            game_type=data.get("game_type"),
            max_participants=data.get("max_participants"),
            current_participants=data.get("current_participants"),
            max_active_matches=data.get("max_active_matches"),
            queue_id=data.get("queue_id"),
            game_server_url=data.get("game_server_url"),
            viewer=TournamentViewer.from_dict(viewer) if isinstance(viewer, dict) else None,
            raw=data,
        )


@dataclass(frozen=True)
class SeatGrant:
    """One assigned seat's GameAPI authorization, as returned by
    ``POST /tournament/agent/assignments/:seatId/grant`` (Agent_ACP
    backend/src/services/officialAgentConnectionService.ts's
    ``buildSeatGrant``). Represents exactly one seat of one match.

    ``access_token`` is temporary (see ``expires_at``) and renewed by
    ``OfficialSeatAuth``; it is excluded from ``repr`` and ``raw``, so
    printing or logging a grant never shows it. Never persisted.
    """

    access_token: str = field(repr=False)
    agent_id: str
    game_session_id: str
    gameapi_server_url: str
    expires_at: str | None = None
    match_id: str | None = None
    seat_id: str | None = None
    seat_position: int | None = None
    seat_count: int | None = None
    game_type: str | None = None
    match_status: str | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict) -> "SeatGrant":
        return cls(
            access_token=data.get("access_token") or "",
            agent_id=data.get("agent_id") or "",
            game_session_id=data.get("game_session_id") or "",
            gameapi_server_url=data.get("gameapi_server_url") or "",
            expires_at=data.get("expires_at"),
            match_id=data.get("match_id"),
            seat_id=data.get("seat_id"),
            seat_position=data.get("seat_position"),
            seat_count=data.get("seat_count"),
            game_type=data.get("game_type"),
            match_status=data.get("match_status"),
            raw={k: v for k, v in data.items() if k != "access_token"},
        )


def _text_or_none(value: Any) -> str | None:
    """A non-empty string from the server, else ``None`` (tolerates any type)."""
    return value if isinstance(value, str) and value.strip() else None


def _opponent_names(value: Any) -> tuple[str, ...]:
    """``[{"name": ...}, ...]`` (or plain strings) -> a tuple of names."""
    if not isinstance(value, list):
        return ()
    names = []
    for item in value:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class OfficialAssignment:
    """One active game seat assigned to this event agent, as listed by
    ``GET /tournament/agent/assignments`` (Agent_ACP
    backend/src/services/officialAgentConnectionService.ts's
    ``OfficialAssignment``). ``seat_id`` is the stable identity: the runtime
    keeps at most one worker per seat.

    The newer fields are optional; an older backend simply leaves them out.
    ``context`` decides which runtime plays the game; the rest are only for
    display:

    - ``context``: ``"testing"`` (a test match, played by ``--match``) or
      ``"tournament"`` (played by ``--tournament``). Missing means a
      tournament game (``altruagent.supervisor.assignment_kind``).
    - ``tournament_id``/``tournament_name``/``round_label``: which tournament
      and round this game belongs to (tournament games only).
    - ``opponents``: the other agents' display names, as a tuple of strings
      (the server sends ``[{"name": ...}]``).
    - ``connect_deadline_at``: ISO 8601 time by which this agent must have
      connected; an agent that isn't connected by then forfeits the game.
    """

    match_id: str
    seat_id: str
    game_type: str | None = None
    seat_position: int | None = None
    seat_count: int | None = None
    match_status: str | None = None
    seat_status: str | None = None
    context: str | None = None
    tournament_id: str | None = None
    tournament_name: str | None = None
    round_label: str | None = None
    opponents: tuple[str, ...] = ()
    connect_deadline_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "OfficialAssignment":
        return cls(
            match_id=str(data.get("match_id") or ""),
            seat_id=str(data.get("seat_id") or ""),
            game_type=data.get("game_type"),
            seat_position=data.get("seat_position"),
            seat_count=data.get("seat_count"),
            match_status=data.get("match_status"),
            seat_status=data.get("seat_status"),
            context=_text_or_none(data.get("context")),
            tournament_id=_text_or_none(data.get("tournament_id")),
            tournament_name=_text_or_none(data.get("tournament_name")),
            round_label=_text_or_none(data.get("round_label")),
            opponents=_opponent_names(data.get("opponents")),
            connect_deadline_at=_text_or_none(data.get("connect_deadline_at")),
        )


@dataclass(frozen=True)
class DecisionContext:
    """The minimal identifying information handed to contestant decision
    logic alongside a ``GameState`` (see ``altruagent.runner``).

    Deliberately carries no client/session object — a decision function
    should be able to reason about the game without being handed enough
    power to mutate an unrelated match. ``session_id``/``game_type`` are
    also available on ``GameState`` itself (as ``session_id``/``game_name``)
    but are repeated here so contestant code doesn't need to thread
    ``state`` through just to log/key by them; ``tournament_id``/
    ``agent_id`` are not available anywhere else.

    ``seat_position`` (0-based) is your seat in the game, set for every game
    played by ``python -m agent`` (``--tournament``/``--match``).
    ``tournament_id`` is set for a tournament game and ``None`` for a Testing
    game.

    ``game_config`` is the game's ``get_game_config`` reference (rules, order
    formats, maps). The runner fills it in for real-time games only, once,
    before the first decision; it's ``None`` otherwise or if the server
    couldn't provide it.
    """

    session_id: str
    tournament_id: str | None
    game_type: str | None
    agent_id: str
    seat_position: int | None = None
    game_config: dict | None = None
