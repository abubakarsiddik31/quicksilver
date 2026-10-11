"""Unit tests for examples/smoke_agent.py, built from the exact template shapes
Agent_ACP's GameAPI emits (gameapi/src/pokemon_adapter/teampreview_mapper.py
``teampreview_action_schema`` and doubles_action_mapper.py
``build_doubles_legal_actions``), wrapped the way GameAPI's
``get_legal_actions`` wraps them (``input["action"]`` is the template).
"""

from __future__ import annotations

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState, LegalAction
from altruagent.runner import _validate_decision
from examples import smoke_agent
from examples.smoke_agent import SmokeAgentError, choose_action, create_agent

CONTEXT = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="a")
ROSTER = ["Incineroar", "Rillaboom", "Urshifu", "Amoonguss", "Tornadus", "Flutter Mane"]


def _state(*legal: dict) -> GameState:
    # get_game_state embeds legal_actions in get_legal_actions' own shape.
    return GameState.from_mcp_state(
        {
            "session_id": "s",
            "status": "in_progress",
            "is_current_actor": True,
            "state_version": 3,
            "legal_actions": {"session_id": "s", "state_version": 3, "actions": list(legal)},
        }
    )


def _templated(action_id: str, template) -> dict:
    return {
        "action_id": action_id,
        "label": action_id,
        "input": {"session_id": "s", "action_id": action_id, "state_version": 3, "action": template},
    }


def _lineup(roster=ROSTER) -> dict:
    return _templated(
        "select_lineup",
        {"type": "select_lineup", "roster": roster, "bring_count": 4, "lead_count": 2, "instructions": "..."},
    )


def _move(move_id: str, targets: list[int]) -> dict:
    return {
        "type": "move", "move_id": move_id, "base_power": 80, "category": "physical",
        "move_type": "normal", "current_pp": 10, "accuracy": 100, "targets": targets,
    }


def _switch(species: str) -> dict:
    return {"type": "switch", "species": species}


PASS = {"type": "pass"}


def _doubles(slot_0_options: list[dict], slot_1_options: list[dict], *, forced=(False, False)) -> dict:
    slots = [
        {"slot": 0, "board_position": -1, "active": None, "force_switch": forced[0], "options": slot_0_options},
        {"slot": 1, "board_position": -2, "active": None, "force_switch": forced[1], "options": slot_1_options},
    ]
    return _templated(
        "doubles_turn",
        {"type": "doubles_turn", "slots": slots, "target_legend": {}, "instructions": "..."},
    )


def _decide(*legal: dict):
    return choose_action(_state(*legal), CONTEXT)


# -- ordinary actions ------------------------------------------------------------------


def test_create_agent_returns_the_decision_function():
    assert create_agent() is choose_action


def test_draft_pick_returns_first_legal_action():
    decision = _decide(
        {"action_id": "draft_pick:card-7", "label": "Draft card-7", "input": {}},
        {"action_id": "draft_pick:card-9", "label": "Draft card-9", "input": {}},
    )

    assert isinstance(decision, LegalAction)
    assert decision.action_id == "draft_pick:card-7"


def test_werewolf_action_returns_first_legal_action():
    decision = _decide({"action_id": "3", "label": "Vote to lynch Player3"}, {"action_id": "7", "label": "Abstain"})

    assert decision.action_id == "3"
    assert _validate_decision(decision, _state({"action_id": "3"}, {"action_id": "7"}).legal_actions)


def test_no_legal_actions_is_a_clear_error():
    with pytest.raises(SmokeAgentError):
        choose_action(GameState.from_mcp_state({"session_id": "s", "legal_actions": []}), CONTEXT)


def test_submit_team_is_refused_rather_than_sent_as_a_template():
    with pytest.raises(SmokeAgentError, match="submit_team"):
        _decide(_templated("submit_team", {"type": "submit_team", "team": "Pass a list..."}))


# -- Team Preview -------------------------------------------------------------------------


def test_select_lineup_brings_first_four_and_leads_first_two():
    decision = _decide(_lineup())

    assert decision == {
        "type": "select_lineup",
        "bring": ["Incineroar", "Rillaboom", "Urshifu", "Amoonguss"],
        "leads": ["Incineroar", "Rillaboom"],
    }
    # The runner passes a dict straight through as the MCP `action` payload.
    assert _validate_decision(decision, _state(_lineup()).legal_actions).action == decision


@pytest.mark.parametrize(
    "template",
    [
        None,
        {"type": "doubles_turn", "roster": ROSTER},
        {"type": "select_lineup"},
        {"type": "select_lineup", "roster": ROSTER[:3]},
        {"type": "select_lineup", "roster": ["A", "B", None, "D"]},
        {"type": "select_lineup", "roster": ["A", "A", "B", "C"]},
        {"type": "select_lineup", "roster": "Incineroar,Rillaboom"},
    ],
)
def test_malformed_select_lineup_template_raises(template):
    with pytest.raises(SmokeAgentError):
        _decide(_templated("select_lineup", template))


# -- doubles turns -------------------------------------------------------------------------


def test_normal_two_slot_turn_uses_first_move_per_slot():
    decision = _decide(
        _doubles(
            [_move("fakeout", [1, 2]), _move("flareblitz", [1, 2]), _switch("Amoonguss")],
            [_move("grassyglide", [1, 2]), _switch("Amoonguss")],
        )
    )

    assert decision == {
        "type": "doubles_turn",
        "slot_0": {"type": "move", "move_id": "fakeout", "target": 1},
        "slot_1": {"type": "move", "move_id": "grassyglide", "target": 1},
    }


def test_move_requiring_target_always_sends_one():
    decision = _decide(_doubles([_move("spore", [2])], [_move("pollenpuff", [-1])]))

    assert decision["slot_0"] == {"type": "move", "move_id": "spore", "target": 2}
    assert decision["slot_1"] == {"type": "move", "move_id": "pollenpuff", "target": -1}


def test_target_prefers_first_positive_over_ally():
    decision = _decide(_doubles([_move("pollenpuff", [-2, 1, 2])], [_move("tackle", [-1, -2])]))

    assert decision["slot_0"]["target"] == 1
    assert decision["slot_1"]["target"] == -1  # no positive target: first listed


def test_targetless_move_omits_target():
    decision = _decide(_doubles([_move("protect", [])], [_move("heatwave", [])]))

    assert decision["slot_0"] == {"type": "move", "move_id": "protect"}
    assert decision["slot_1"] == {"type": "move", "move_id": "heatwave"}


def test_forced_switch_picks_first_switch():
    decision = _decide(
        _doubles([_switch("Amoonguss"), _switch("Tornadus")], [PASS], forced=(True, False))
    )

    assert decision["slot_0"] == {"type": "switch", "species": "Amoonguss"}
    assert decision["slot_1"] == PASS


def test_one_pass_only_slot_and_one_forced_switch_slot():
    decision = _decide(_doubles([PASS], [_switch("Tornadus"), _switch("Flutter Mane")], forced=(False, True)))

    assert decision == {
        "type": "doubles_turn",
        "slot_0": PASS,
        "slot_1": {"type": "switch", "species": "Tornadus"},
    }


def test_two_forced_switches_choose_distinct_reserves():
    decision = _decide(
        _doubles(
            [_switch("Amoonguss"), _switch("Tornadus")],
            [_switch("Amoonguss"), _switch("Tornadus")],
            forced=(True, True),
        )
    )

    assert decision["slot_0"] == {"type": "switch", "species": "Amoonguss"}
    assert decision["slot_1"] == {"type": "switch", "species": "Tornadus"}


def test_two_forced_switches_one_reserve_switches_once_and_passes():
    # GameAPI's one-shared-reserve case: both slots offer [switch X, pass].
    decision = _decide(
        _doubles([_switch("Amoonguss"), PASS], [_switch("Amoonguss"), PASS], forced=(True, True))
    )

    assert decision["slot_0"] == {"type": "switch", "species": "Amoonguss"}
    assert decision["slot_1"] == PASS


def test_pass_only_slot_passes():
    decision = _decide(_doubles([_move("fakeout", [1, 2])], [PASS]))

    assert decision["slot_0"]["type"] == "move"
    assert decision["slot_1"] == PASS


def test_slot_with_no_distinct_switch_and_no_pass_raises():
    with pytest.raises(SmokeAgentError, match="slot 1"):
        _decide(_doubles([_switch("Amoonguss")], [_switch("Amoonguss")], forced=(True, True)))


@pytest.mark.parametrize(
    "template",
    [
        None,
        {"type": "select_lineup", "slots": []},
        {"type": "doubles_turn"},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [PASS]}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [PASS]}, {"slot": 0, "options": [PASS]}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [PASS]}, {"slot": 1, "options": []}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [PASS]}, {"slot": 1, "options": [{"type": "mega"}]}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [PASS]}, "not-a-slot"]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [_move("", [1])]}, {"slot": 1, "options": [PASS]}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [_move("tackle", ["1"])]}, {"slot": 1, "options": [PASS]}]},
        {"type": "doubles_turn", "slots": [{"slot": 0, "options": [{"type": "switch"}]}, {"slot": 1, "options": [PASS]}]},
    ],
)
def test_malformed_doubles_turn_template_raises(template):
    with pytest.raises(SmokeAgentError):
        _decide(_templated("doubles_turn", template))


def test_smoke_agent_error_is_a_value_error_so_the_runner_reports_it():
    assert issubclass(smoke_agent.SmokeAgentError, ValueError)
