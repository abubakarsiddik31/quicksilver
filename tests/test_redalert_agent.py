"""Tests verifying Quicksilver's Red Alert autonomous agent."""

from __future__ import annotations

from types import SimpleNamespace
from altruagent import DecisionContext, GameState, WAIT, WithReasoning
from common.dispatch import QuicksilverAgent
from redalert.agent import RedAlertAgent


def make_ra_state(*, tick=1, mcv_id=None, ready_items=(), avail=(), cash=5000, power_provided=0, power_drained=0):
    obs = {
        "tick": tick,
        "economy": {
            "cash": cash,
            "power_provided": power_provided,
            "power_drained": power_drained,
        },
        "production": [{"item": item, "ready": True} for item in ready_items],
        "available_production": list(avail),
        "buildings": [],
        "units": [],
    }
    legal = {
        "deploy": [mcv_id] if mcv_id else [],
        "combat_units": [],
        "idle_units": [],
    }
    raw = {
        "observation": obs,
        "legal_actions": legal,
    }
    return SimpleNamespace(raw=raw, legal_actions=legal, is_current_actor=True)


def test_redalert_deploys_mcv_on_start():
    agent = RedAlertAgent()
    ctx = DecisionContext(
        session_id="ra-1",
        tournament_id=None,
        game_type="red_alert",
        agent_id="quicksilver",
        seat_position=0,
    )
    state = make_ra_state(tick=0, mcv_id=101, avail=["powr"])
    action = agent.choose_action(state, ctx)

    assert isinstance(action, WithReasoning)
    payload = action.action
    assert payload["type"] == "orders"
    assert any(o.get("cmd") == "deploy" and o.get("units") == [101] for o in payload["orders"])


def test_redalert_places_finished_buildings():
    agent = RedAlertAgent()
    ctx = DecisionContext(
        session_id="ra-1",
        tournament_id=None,
        game_type="red_alert",
        agent_id="quicksilver",
        seat_position=0,
    )
    state = make_ra_state(tick=50, ready_items=["powr"], avail=["tent", "barr"])
    action = agent.choose_action(state, ctx)

    assert isinstance(action, WithReasoning)
    payload = action.action
    assert any(o.get("cmd") == "place" and o.get("item") == "powr" for o in payload["orders"])


def test_dispatch_routes_red_alert_to_agent(monkeypatch):
    monkeypatch.setenv("ENABLE_RED_ALERT", "true")
    from common.dispatch import _create_delegate
    agent = _create_delegate("red_alert")
    assert isinstance(agent, RedAlertAgent)
