"""Unit tests for altruagent.models."""

from __future__ import annotations

import pytest

from altruagent.models import (
    Agent,
    AgentSessions,
    GameState,
    LegalAction,
    Match,
    Message,
    NextAction,
    OfficialAssignment,
    Tournament,
    TournamentViewer,
)


def test_agent_parsing_excludes_sensitive_hash_fields():
    data = {
        "id": "agent-1",
        "name": "MyAgent",
        "status": "unclaimed",
        "description": None,
        "api_key_hash": "deadbeef",
        "claim_token_hash": "cafebabe",
    }

    agent = Agent.from_dict(data)

    assert agent.id == "agent-1"
    assert agent.name == "MyAgent"
    assert agent.is_claimed is False
    assert "api_key_hash" not in agent.raw
    assert "claim_token_hash" not in agent.raw
    assert not hasattr(agent, "api_key_hash")
    assert not hasattr(agent, "claim_token_hash")


def test_agent_is_claimed_reflects_status():
    claimed = Agent.from_dict({"id": "a", "name": "A", "status": "claimed"})
    unclaimed = Agent.from_dict({"id": "b", "name": "B", "status": "unclaimed"})

    assert claimed.is_claimed is True
    assert unclaimed.is_claimed is False


def test_agent_from_dict_tolerates_unknown_extra_fields():
    agent = Agent.from_dict(
        {
            "id": "agent-1",
            "name": "MyAgent",
            "status": "claimed",
            "some_future_field": {"nested": True},
        }
    )

    assert agent.id == "agent-1"
    assert agent.raw["some_future_field"] == {"nested": True}


# -- GameState / NextAction -----------------------------------------------


def _base_game_state_payload(**overrides):
    payload = {
        "session_id": "session-1",
        "game_name": "tic_tac_toe",
        "status": "active",
        "observation": "...",
        "current_player": {"name": "Alice"},
        "legal_actions": [0, 1, 2],
        "legal_actions_str": ["top-left", "top-middle", "top-right"],
        "is_terminal": False,
        "returns": None,
        "move_count": 3,
        "termination_reason": None,
        "messaging_enabled": False,
        "phase": "moving",
        "next_actions": [
            {
                "action": "make_move",
                "endpoint": "POST /games/session-1/step",
                "hint": "It is your turn.",
                "required_fields": ["action"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_game_state_parses_core_fields():
    state = GameState.from_dict(_base_game_state_payload())

    assert state.session_id == "session-1"
    assert state.game_name == "tic_tac_toe"
    assert state.status == "active"
    assert state.is_terminal is False
    assert state.current_player.name == "Alice"
    assert state.move_count == 3


def test_game_state_legal_actions_parsing():
    # REST's separate legal_actions/legal_actions_str int/str lists are
    # synthesized into LegalAction entries — the same shape MCP's
    # get_legal_actions returns natively (see from_mcp_state below) — so
    # contestant code never has to care which transport produced a GameState.
    state = GameState.from_dict(
        _base_game_state_payload(legal_actions=[0, 4, 8], legal_actions_str=["a", "b", "c"])
    )

    assert [a.action_id for a in state.legal_actions] == ["0", "4", "8"]
    assert [a.label for a in state.legal_actions] == ["a", "b", "c"]
    assert all(isinstance(a, LegalAction) for a in state.legal_actions)


def test_game_state_current_player_none_when_not_your_turn():
    state = GameState.from_dict(
        _base_game_state_payload(current_player=None, legal_actions=[])
    )

    assert state.current_player is None
    assert state.legal_actions == []


def test_game_state_next_actions_can_have_multiple_entries():
    # During a messaging round, GameAPI returns both send_message and
    # terminate_messaging together (see next_actions.py compute_next_actions).
    state = GameState.from_dict(
        _base_game_state_payload(
            messaging_enabled=True,
            phase="messaging",
            next_actions=[
                {"action": "send_message", "hint": "chat or terminate", "required_fields": ["type"]},
                {"action": "terminate_messaging", "hint": "end the round", "required_fields": ["type"]},
            ],
        )
    )

    assert [a.action for a in state.next_actions] == ["send_message", "terminate_messaging"]
    assert all(isinstance(a, NextAction) for a in state.next_actions)


# -- messaging (Milestone 5) ----------------------------------------------


def test_message_from_dict_parses_all_fields():
    message = Message.from_dict(
        {
            "index": 3,
            "sender": 0,
            "recipients": [1],
            "content": "let's cooperate",
            "type": "chat",
            "sent_at": "2026-01-01T00:00:00Z",
            "move_index": 2,
        }
    )

    assert message.index == 3
    assert message.sender == 0
    assert message.recipients == [1]
    assert message.content == "let's cooperate"
    assert message.type == "chat"
    assert message.move_index == 2


def test_message_from_dict_tolerates_missing_fields():
    message = Message.from_dict({})

    assert message.index == 0
    assert message.sender == 0
    assert message.recipients == []
    assert message.content == ""
    assert message.type == "chat"
    assert message.move_index == 0


def test_game_state_parses_new_messages_as_typed_message_objects():
    state = GameState.from_dict(
        _base_game_state_payload(
            messaging_enabled=True,
            phase="messaging",
            messaging_mode="per_all_moves",
            terminated_messaging=[1],
            new_messages=[
                {
                    "index": 0,
                    "sender": 1,
                    "recipients": [],
                    "content": "hello",
                    "type": "chat",
                    "sent_at": "2026-01-01T00:00:00Z",
                    "move_index": 0,
                }
            ],
        )
    )

    assert state.messaging_mode == "per_all_moves"
    assert state.terminated_messaging == [1]
    assert len(state.new_messages) == 1
    assert isinstance(state.new_messages[0], Message)
    assert state.new_messages[0].content == "hello"
    assert state.new_messages[0].sender == 1


def test_game_state_messaging_fields_default_when_absent():
    payload = _base_game_state_payload()
    state = GameState.from_dict(payload)

    assert state.new_messages == []
    assert state.terminated_messaging == []
    assert state.messaging_mode == "per_move"
    assert state.messaging_enabled is False
    assert state.phase == "moving"


def test_game_state_preserves_unknown_game_specific_extra_fields():
    state = GameState.from_dict(
        _base_game_state_payload(
            avalon_round=2,
            avalon_leader="Bob",
            round_history=[{"round": 1, "actions": {}}],
            cumulative_scores={"Alice": 2.0},
        )
    )

    assert state.raw["avalon_round"] == 2
    assert state.raw["avalon_leader"] == "Bob"
    assert state.raw["round_history"] == [{"round": 1, "actions": {}}]
    assert state.raw["cumulative_scores"] == {"Alice": 2.0}


def test_match_from_dict_basic_fields():
    match = Match.from_dict(
        {
            "session_id": "session-1",
            "game_type": "tic_tac_toe",
            "status": "waiting",
            "tournament_id": None,
            "created_at": "2026-01-01T00:00:00Z",
        }
    )

    assert match.session_id == "session-1"
    assert match.game_type == "tic_tac_toe"
    assert match.status == "waiting"
    assert match.tournament_id is None
    assert match.game_server_url is None  # never present on this endpoint's rows


def test_match_preserves_unknown_extra_fields_in_raw():
    match = Match.from_dict(
        {
            "session_id": "session-1",
            "status": "waiting",
            "winner_agent_id": None,
            "results": None,
            "runtime_adapter": "openspiel",
        }
    )

    assert match.raw["runtime_adapter"] == "openspiel"
    assert "winner_agent_id" in match.raw


def test_match_tolerates_missing_optional_fields():
    match = Match.from_dict({"session_id": "session-1", "status": "in_progress"})

    assert match.game_type is None
    assert match.tournament_id is None
    assert match.created_at is None
    assert match.started_at is None
    assert match.completed_at is None


def test_match_tournament_id_preserved_for_child_matches():
    match = Match.from_dict(
        {"session_id": "session-1", "status": "waiting", "tournament_id": "tournament-9"}
    )

    assert match.tournament_id == "tournament-9"


def test_match_game_without_client_raises_value_error():
    match = Match.from_dict({"session_id": "session-1", "status": "in_progress"})

    with pytest.raises(ValueError):
        match.game()


def test_agent_sessions_maps_server_groups_to_waiting_active_completed():
    sessions = AgentSessions.from_dict(
        {
            "joined_sessions": [{"session_id": "s-waiting", "status": "waiting"}],
            "active_sessions": [{"session_id": "s-active", "status": "in_progress"}],
            "completed_sessions": [{"session_id": "s-done", "status": "completed"}],
        }
    )

    assert [m.session_id for m in sessions.waiting] == ["s-waiting"]
    assert [m.session_id for m in sessions.active] == ["s-active"]
    assert [m.session_id for m in sessions.completed] == ["s-done"]


def test_agent_sessions_tolerates_missing_groups():
    sessions = AgentSessions.from_dict({})

    assert sessions.waiting == []
    assert sessions.active == []
    assert sessions.completed == []


def test_game_state_terminal_with_returns():
    state = GameState.from_dict(
        _base_game_state_payload(
            is_terminal=True,
            current_player=None,
            legal_actions=[],
            returns={"Alice": 1.0, "Bob": -1.0},
            termination_reason="completed",
            next_actions=[{"action": "game_over", "hint": "Game finished."}],
        )
    )

    assert state.is_terminal is True
    assert state.returns == {"Alice": 1.0, "Bob": -1.0}
    assert state.termination_reason == "completed"
    assert [a.action for a in state.next_actions] == ["game_over"]


# -- LegalAction / GameState.from_mcp_state (Milestone 6: MCP-first) -------


def test_legal_action_from_dict_openspiel_shape():
    action = LegalAction.from_dict(
        {"action_id": "0", "label": "cooperate", "input": {"session_id": "s-1", "action_id": "0"}}
    )

    assert action.action_id == "0"
    assert action.label == "cooperate"
    assert action.input == {"session_id": "s-1", "action_id": "0"}


def test_legal_action_from_dict_structured_shape():
    # Pokemon-shaped: action_id is not int-coercible, input carries a richer
    # structured payload than OpenSpiel-family games ever need.
    action = LegalAction.from_dict(
        {
            "action_id": "move:0",
            "label": "Use Thunderbolt",
            "input": {"type": "move", "slot": 0, "move_id": "thunderbolt", "base_power": 90},
        }
    )

    assert action.action_id == "move:0"
    assert action.input["type"] == "move"
    assert action.input["base_power"] == 90


def _mcp_state_payload(**overrides) -> dict:
    payload = {
        "session_id": "session-1",
        "game_type": "tic_tac_toe",
        "runtime_adapter": "openspiel",
        "status": "in_progress",
        "state_version": 3,
        "observation": "...",
        "phase": "moving",
        "messaging_enabled": False,
        "terminated_messaging": [],
        "new_messages": [],
        "current_actor": {"agent_id": "agent-1", "position": 0},
        "is_current_actor": True,
        "is_terminal": False,
        "legal_action_count": 3,
    }
    payload.update(overrides)
    return payload


def test_game_state_from_mcp_state_parses_generic_fields():
    state = GameState.from_mcp_state(_mcp_state_payload())

    assert state.session_id == "session-1"
    assert state.game_name == "tic_tac_toe"
    assert state.state_version == 3
    assert state.phase == "moving"
    assert state.is_current_actor is True
    assert state.is_terminal is False
    assert state.current_player.name == "agent-1"
    # get_game_state alone carries no legal_actions (a separate tool) —
    # confirmed empty until the runner merges in a get_legal_actions result.
    assert state.legal_actions == []


def test_game_state_from_mcp_state_merges_legal_actions_and_its_state_version():
    legal_actions = {
        "session_id": "session-1",
        "state_version": 4,  # freshest — the one that should win
        "actions": [
            {"action_id": "0", "label": "a", "input": {}},
            {"action_id": "1", "label": "b", "input": {}},
        ],
    }
    state = GameState.from_mcp_state(_mcp_state_payload(state_version=3), legal_actions=legal_actions)

    assert state.state_version == 4
    assert [a.action_id for a in state.legal_actions] == ["0", "1"]


def test_game_state_from_mcp_state_merges_result_for_terminal_fields():
    # get_game_state/play_action never carry returns/termination_reason —
    # confirmed against openspiel_adapter.py; only get_result/resign do.
    result = {
        "session_id": "session-1",
        "is_terminal": True,
        "status": "completed",
        "returns": {"Alice": 1.0, "Bob": -1.0},
        "your_return": 1.0,
        "termination_reason": "completed",
    }
    state = GameState.from_mcp_state(
        _mcp_state_payload(is_terminal=True, is_current_actor=False), result=result
    )

    assert state.is_terminal is True
    assert state.returns == {"Alice": 1.0, "Bob": -1.0}
    assert state.termination_reason == "completed"


def test_game_state_from_mcp_state_pokemon_shaped_never_reports_messaging():
    # Confirmed against pokemon_adapter.py: phases are draft/draft_complete/
    # teambuild/moving, never "messaging" — messaging_enabled is always False.
    state = GameState.from_mcp_state(
        _mcp_state_payload(
            game_type="pokemon_gen9ou_draft",
            phase="draft",
            messaging_enabled=False,
            current_actor=None,
            is_current_actor=True,
        )
    )

    assert state.phase == "draft"
    assert state.messaging_enabled is False


def test_game_state_from_mcp_state_new_messages_parsed_as_message_objects():
    state = GameState.from_mcp_state(
        _mcp_state_payload(
            phase="messaging",
            messaging_enabled=True,
            new_messages=[
                {
                    "message_id": "session-1:0",
                    "index": 0,
                    "sender": 1,
                    "recipients": [],
                    "content": "hello",
                    "type": "chat",
                    "sent_at": "2026-01-01T00:00:00Z",
                    "move_index": 2,
                }
            ],
        )
    )

    assert len(state.new_messages) == 1
    assert state.new_messages[0].content == "hello"
    assert state.new_messages[0].sender == 1


# -- Tournament / TournamentViewer -----------------------------------------


def test_tournament_from_dict_list_shape():
    # GET /tournaments list entries are raw DB rows: no viewer, no
    # game_server_url (never a stored column).
    tournament = Tournament.from_dict(
        {
            "tournament_id": "t-1",
            "game_type": "tic_tac_toe",
            "status": "waiting",
            "max_participants": 2,
            "current_participants": 1,
            "max_active_matches": 1,
            "queue_id": None,
            "created_by_user_id": "user-1",
            "metadata": None,
            "created_at": "2026-01-01T00:00:00Z",
        }
    )

    assert tournament.tournament_id == "t-1"
    assert tournament.status == "waiting"
    assert tournament.current_participants == 1
    assert tournament.game_server_url is None
    assert tournament.viewer is None
    assert tournament.raw["created_by_user_id"] == "user-1"


def test_tournament_from_dict_detail_shape_with_viewer():
    # GET /tournaments/{id}'s `tournament` sub-object (compactTournament) can
    # include game_server_url; `viewer` is passed separately (it's a sibling
    # key in the response body, not nested inside `tournament`).
    tournament = Tournament.from_dict(
        {
            "tournament_id": "t-1",
            "game_type": "tic_tac_toe",
            "status": "in_progress",
            "max_participants": 2,
            "current_participants": 2,
            "max_active_matches": 1,
            "queue_id": None,
            "game_server_url": "host:8000",
        },
        viewer={
            "agent_id": "agent-1",
            "is_tournament_participant": True,
            "active_child_session_ids": ["session-1"],
            "should_join_tournament": False,
            "should_wait_for_child_match": False,
            "next_actions": [
                {"action": "play_child_session", "endpoint": "GET .../games/session-1", "hint": "Play it."}
            ],
        },
    )

    assert tournament.game_server_url == "host:8000"
    assert tournament.viewer is not None
    assert tournament.viewer.agent_id == "agent-1"
    assert tournament.viewer.is_tournament_participant is True
    assert tournament.viewer.active_child_session_ids == ["session-1"]
    assert [a.action for a in tournament.viewer.next_actions] == ["play_child_session"]


def test_tournament_viewer_absent_gives_none():
    tournament = Tournament.from_dict({"tournament_id": "t-1", "status": "waiting"})

    assert tournament.viewer is None


def test_tournament_tolerates_missing_optional_fields():
    tournament = Tournament.from_dict({"tournament_id": "t-1", "status": "waiting"})

    assert tournament.game_type is None
    assert tournament.max_participants is None
    assert tournament.current_participants is None
    assert tournament.queue_id is None
    assert tournament.game_server_url is None


def test_tournament_preserves_unknown_fields_in_raw():
    tournament = Tournament.from_dict(
        {"tournament_id": "t-1", "status": "waiting", "leaderboard": [], "messaging_config": {"messaging_enabled": True}}
    )

    assert tournament.raw["leaderboard"] == []
    assert tournament.raw["messaging_config"] == {"messaging_enabled": True}


def test_tournament_has_no_name_attribute():
    # There is no `name` field anywhere on the backend's tournaments table —
    # guard against ever accidentally assuming one exists.
    tournament = Tournament.from_dict({"tournament_id": "t-1", "status": "waiting", "name": "Ignored"})

    assert not hasattr(tournament, "name")
    # If a "name" key is ever present in a payload (it shouldn't be), it's
    # only reachable via raw, never promoted to a typed attribute.
    assert tournament.raw.get("name") == "Ignored"


def test_tournament_viewer_from_dict_defaults():
    viewer = TournamentViewer.from_dict({})

    assert viewer.agent_id is None
    assert viewer.is_tournament_participant is False
    assert viewer.active_child_session_ids == []
    assert viewer.next_actions == []


def test_message_from_dict_prefers_platform_seq_over_legacy_index():
    assert Message.from_dict({"seq": 14, "index": 2}).index == 14
    assert Message.from_dict({"index": 2}).index == 2


# -- OfficialAssignment ------------------------------------------------------------


def test_official_assignment_parses_the_base_fields_only_from_an_older_backend():
    assignment = OfficialAssignment.from_dict({
        "match_id": "m-1", "seat_id": "s-1", "game_type": "werewolf", "seat_position": 2,
        "seat_count": 7, "match_status": "starting", "seat_status": "pending",
    })

    assert (assignment.match_id, assignment.seat_id, assignment.game_type) == ("m-1", "s-1", "werewolf")
    assert (assignment.context, assignment.tournament_id, assignment.tournament_name) == (None, None, None)
    assert (assignment.round_label, assignment.opponents, assignment.connect_deadline_at) == (None, (), None)


def test_official_assignment_parses_the_tournament_fields():
    assignment = OfficialAssignment.from_dict({
        "match_id": "m-1", "seat_id": "s-1", "context": "tournament", "tournament_id": "t-1",
        "tournament_name": "Fall Cup", "round_label": "Final", "opponents": [{"name": "Alpha"}],
        "connect_deadline_at": "2026-10-16T15:04:00Z",
    })

    assert assignment.context == "tournament"
    assert (assignment.tournament_id, assignment.tournament_name, assignment.round_label) == ("t-1", "Fall Cup", "Final")
    assert assignment.opponents == ("Alpha",)
    assert assignment.connect_deadline_at == "2026-10-16T15:04:00Z"


@pytest.mark.parametrize(
    "opponents, expected",
    [
        ([{"name": "A"}, "B", {"name": ""}, {"name": None}, {}, 7, None, {"name": "C", "id": "x"}], ("A", "B", "C")),
        ({"name": "A"}, ()),
        ("Alpha", ()),
        (None, ()),
    ],
)
def test_official_assignment_tolerates_odd_opponent_lists(opponents, expected):
    assignment = OfficialAssignment.from_dict({"match_id": "m", "seat_id": "s", "opponents": opponents})

    assert assignment.opponents == expected


def test_official_assignment_ignores_non_text_optional_fields():
    assignment = OfficialAssignment.from_dict({
        "match_id": "m", "seat_id": "s", "context": 3, "tournament_id": None, "tournament_name": "  ",
        "round_label": ["x"], "connect_deadline_at": 1760000000,
    })

    assert (assignment.context, assignment.tournament_id, assignment.tournament_name) == (None, None, None)
    assert (assignment.round_label, assignment.connect_deadline_at) == (None, None)


def test_official_assignment_is_hashable_with_opponents():
    assignment = OfficialAssignment.from_dict({"match_id": "m", "seat_id": "s", "opponents": [{"name": "A"}]})

    assert hash(assignment) == hash(OfficialAssignment.from_dict({"match_id": "m", "seat_id": "s", "opponents": ["A"]}))
