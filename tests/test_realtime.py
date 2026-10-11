"""Real-time games: the runner's real-time path (altruagent.runner) and the
example LLM agent's Red Alert player (examples/llm/redalert.py).

Game payloads mirror Agent_ACP GameAPI's Red Alert state (backend/skill/
redalert.md): ``pacing.mode == "realtime"``, an ``observation`` with units,
buildings, production and ``available_production``, ``legal_actions`` as a
dict of short lists (``attack_now``, ``enemy_base``, ...), and a
``play_action`` answer that carries per-order verdicts but no state. The
model is always a scripted fake; nothing reaches OpenAI.
"""

from __future__ import annotations

import json

import pytest

from altruagent import WAIT, DecisionContext, WithReasoning
from altruagent.mcp_transport import MCPToolError
from altruagent.models import GameState
from altruagent.runner import REALTIME_RETRY_SECONDS, DecisionError, run_game
from examples.llm import redalert
from examples.llm.base import UnsupportedStructuredAction
from examples.llm.providers import ProviderError
from examples.llm_agent import LLMAgent
from tests.test_runner import FakeMCPGameSession, make_mcp_state, result_dict, terminal_state

RED_ALERT = DecisionContext(session_id="ra", tournament_id=None, game_type="red_alert", agent_id="a")
PACING = {"mode": "realtime", "min_batch_interval_s": 0.2, "stale_seconds": 120}
CONFIG = {
    "rules": {
        "buildings": {"powr": {"cost": 300, "power": 100, "build_time_s": 7.2, "queue": "Building"}},
        "units": {"e1": {"cost": 100, "build_time_s": 2.4, "queue": "Infantry"}},
        "maps": {"singles": {"bounds": [2, 2, 108, 50], "spawns": [[12, 16], [95, 11]]}},
    }
}
ATTACK_NOW = {"cmd": "attack_move", "units": [120, 121], "to": [95, 11]}


def ra_payload(tick=100, **extra) -> dict:
    payload = {
        "session_id": "ra",
        "game_type": "red_alert",
        "status": "in_progress",
        "phase": "playing",
        "is_current_actor": True,
        "is_terminal": False,
        "state_version": tick,
        "pacing": PACING,
        "your_seat": "Multi0",
        "map": {"id": "singles", "width": 112, "height": 54, "bounds": [2, 2, 108, 50]},
        "limits": {"max_orders_per_batch": 20, "max_units_per_order": 50, "max_train_count": 10},
        "observation": {
            "tick": tick,
            "faction": "england",
            "economy": {"cash": 5000, "power_provided": 100, "power_drained": 20},
            "units": [{"id": 120, "type": "e1", "x": 14, "y": 17, "hp": 100, "idle": True},
                      {"id": 121, "type": "e1", "x": 15, "y": 17, "hp": 100, "idle": True}],
            "buildings": [{"id": 140, "type": "fact", "x": 12, "y": 16, "hp": 100}],
            "production": [],
            "available_production": ["powr", "e1"],
            "base_center": [12, 16],
            "events": [],
        },
        "legal_actions": {"build": ["powr"], "train": ["e1"], "attack_now": ATTACK_NOW, "enemy_base": [95, 11]},
    }
    payload.update(extra)
    return payload


def ra_state(tick=100, **extra) -> GameState:
    return GameState.from_mcp_state(ra_payload(tick, **extra))


def accepted(*rows) -> dict:
    """A real-time play_action answer: verdicts, no state."""
    return {"accepted": True, "session_id": "ra", "status": "in_progress", "pending": True,
            "results": [{"index": i, "cmd": c, "status": s, "reason": r} for i, (c, s, r) in enumerate(rows)]}


class RealtimeFakeGame(FakeMCPGameSession):
    """FakeMCPGameSession plus reasoning_summary and get_game_config."""

    def __init__(self, config=None):
        super().__init__(session_id="ra")
        self.config = config
        self.config_calls: list[str] = []

    def play_action(self, *, action_id=None, action=None, state_version, reasoning_summary=None) -> dict:
        self.play_action_calls.append(
            {"action_id": action_id, "action": action, "state_version": state_version,
             "reasoning_summary": reasoning_summary}
        )
        return self._pop(self._play_action_queue)

    def get_game_config(self, game_type: str) -> dict:
        self.config_calls.append(game_type)
        if isinstance(self.config, BaseException):
            raise self.config
        return self.config


class Recorder:
    """An agent object: scripted decisions, records contexts and results."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.contexts: list[DecisionContext] = []
        self.results: list[dict] = []

    def choose_action(self, state, context):
        self.contexts.append(context)
        return self.decisions.pop(0)

    def on_action_result(self, result, context):
        self.results.append(result)


ORDERS = {"type": "orders", "orders": [{"cmd": "build", "item": "powr"}]}


def _tool_error(code: str, detail: str = "nope") -> MCPToolError:
    return MCPToolError(f"{code}: {detail}", status_code=None, error_code=code)


# -- runner -------------------------------------------------------------------------------


def test_wait_in_realtime_waits_for_the_next_view_and_asks_again():
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100), ra_state(105), terminal_state())
    game.queue_play_action(accepted(("build", "pending", "pending"))).queue_result(result_dict())
    agent = Recorder(WAIT, ORDERS)
    run_game(game, RED_ALERT, agent, sleep=lambda s: None)
    assert game.wait_calls[0]["since_version"] == 100
    assert game.play_action_calls[0]["action"] == ORDERS
    assert game.play_action_calls[0]["state_version"] == 105
    # A real-time answer carries no state, so the runner re-reads it.
    assert game.get_state_calls == 2


def test_wait_in_a_turn_based_game_is_a_decision_error():
    game = RealtimeFakeGame().queue_state(make_mcp_state(legal_actions={"actions": [{"action_id": "0"}]}))
    with pytest.raises(DecisionError, match="WAIT"):
        run_game(game, RED_ALERT, Recorder(WAIT), sleep=lambda s: None)


def test_invalid_action_in_realtime_is_reported_and_play_continues():
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100), ra_state(110), terminal_state())
    game.queue_play_action(_tool_error("INVALID_ACTION", "[0] deploy: not_owned (unit 7 no longer exists)"),
                           accepted(("build", "pending", "pending")))
    game.queue_result(result_dict())
    agent = Recorder(ORDERS, ORDERS)
    run_game(game, RED_ALERT, agent, sleep=lambda s: None)
    assert agent.results[0]["error"] == "INVALID_ACTION"
    assert "not_owned" in agent.results[0]["detail"]
    assert agent.results[1]["accepted"] is True
    assert len(game.play_action_calls) == 2


def test_invalid_action_in_a_turn_based_game_still_fails_fast():
    state = make_mcp_state(legal_actions={"actions": [{"action_id": "0", "label": "0", "input": {}}]})
    game = RealtimeFakeGame().queue_state(state).queue_play_action(_tool_error("INVALID_ACTION"))
    with pytest.raises(DecisionError):
        run_game(game, RED_ALERT, Recorder("0"), sleep=lambda s: None)


def test_runtime_temporarily_unavailable_in_realtime_retries_after_a_pause():
    slept: list[float] = []
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100), ra_state(130), terminal_state())
    game.queue_play_action(_tool_error("RUNTIME_TEMPORARILY_UNAVAILABLE"), accepted(("build", "pending", "pending")))
    game.queue_result(result_dict())
    agent = Recorder(ORDERS, ORDERS)
    run_game(game, RED_ALERT, agent, sleep=slept.append)
    assert slept == [REALTIME_RETRY_SECONDS]
    assert agent.results[0]["error"] == "RUNTIME_TEMPORARILY_UNAVAILABLE"


def test_game_config_is_fetched_once_for_a_realtime_game():
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100), ra_state(105), terminal_state())
    game.queue_play_action(accepted(("build", "pending", "pending")))
    game.queue_result(result_dict())
    agent = Recorder(WAIT, ORDERS)
    run_game(game, RED_ALERT, agent, sleep=lambda s: None)
    assert game.config_calls == ["red_alert"]
    assert all(c.game_config is CONFIG for c in agent.contexts)


def test_game_config_errors_leave_it_unset():
    game = RealtimeFakeGame(_tool_error("UNKNOWN_GAME")).queue_state(ra_state(100), terminal_state())
    game.queue_play_action(accepted(("build", "pending", "pending"))).queue_result(result_dict())
    agent = Recorder(ORDERS)
    run_game(game, RED_ALERT, agent, sleep=lambda s: None)
    assert agent.contexts[0].game_config is None


def test_turn_based_games_never_fetch_the_config():
    state = make_mcp_state(legal_actions={"actions": [{"action_id": "0", "label": "0", "input": {}}]})
    game = RealtimeFakeGame(CONFIG).queue_state(state, terminal_state())
    game.queue_play_action({"accepted": True, "status": "in_progress"}).queue_result(result_dict())
    run_game(game, RED_ALERT, Recorder("0"), sleep=lambda s: None)
    assert game.config_calls == []


def _placeholder(state, context):
    return state.legal_actions[0]  # agent/agent.py's choose_action


def test_the_placeholder_in_red_alert_says_plainly_it_cant_play_it():
    # Was: "choose_action raised IndexError('list index out of range') ...".
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError) as excinfo:
        run_game(game, RED_ALERT, _placeholder, sleep=lambda s: None)

    message = str(excinfo.value)
    assert message.startswith("Your agent can't play Red Alert:")
    assert "GAMES.md" in message and "examples.llm_agent" in message
    assert "IndexError" not in message and "seat" not in message
    assert isinstance(excinfo.value.__cause__, IndexError)


def test_an_unknown_real_time_game_gets_the_same_plain_message_without_a_name():
    context = DecisionContext(session_id="rt", tournament_id=None, game_type="future_rts", agent_id="a")
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"^Your agent can't play this real-time game \(future_rts\):"):
        run_game(game, context, _placeholder, sleep=lambda s: None)


def test_other_agent_errors_in_a_real_time_game_keep_the_usual_message():
    def broken(state, context):
        return {}["missing"]

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"choose_action raised KeyError"):
        run_game(game, RED_ALERT, broken, sleep=lambda s: None)


def test_the_real_placeholder_module_in_red_alert_says_plainly_it_cant_play_it():
    from agent.agent import create_agent

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"^Your agent can't play Red Alert:"):
        run_game(game, RED_ALERT, create_agent(), sleep=lambda s: None)


def test_picking_at_random_from_the_empty_legal_actions_says_it_cant_play_either():
    import random

    def random_pick(state, context):
        return random.choice(state.legal_actions)  # IndexError raised inside random.py

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"^Your agent can't play Red Alert:"):
        run_game(game, RED_ALERT, random_pick, sleep=lambda s: None)


def test_an_unrelated_index_error_in_a_real_red_alert_agent_keeps_the_real_error():
    # Was: "Your agent can't play Red Alert: ..." for any IndexError, since
    # Red Alert's legal_actions is always empty; the real error was dropped.
    def own_bug(state, context):
        units = []
        return {"orders": [{"cmd": "stop", "units": [units[0]]}]}

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError) as excinfo:
        run_game(game, RED_ALERT, own_bug, sleep=lambda s: None)

    message = str(excinfo.value)
    assert message.startswith("choose_action raised IndexError('list index out of range')")
    assert "can't play" not in message


def test_an_index_error_in_a_helper_given_legal_actions_keeps_the_real_error():
    # The line that failed is the helper's own units[0], not a pick from
    # legal_actions, even though the caller's line names legal_actions.
    def first_unit(legal_actions, units):
        return units[0]

    def agent(state, context):
        return {"orders": [{"cmd": "stop", "units": [first_unit(state.legal_actions, [])]}]}

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"^choose_action raised IndexError"):
        run_game(game, RED_ALERT, agent, sleep=lambda s: None)


def test_only_the_expression_that_failed_counts_not_the_rest_of_its_line():
    # The line names legal_actions, but what failed was units[0].
    def fallback(state, context):
        units = []
        return state.legal_actions[0] if state.legal_actions else {"orders": [{"cmd": "stop", "units": [units[0]]}]}

    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100))

    with pytest.raises(DecisionError, match=r"^choose_action raised IndexError"):
        run_game(game, RED_ALERT, fallback, sleep=lambda s: None)


def test_an_index_error_in_a_turn_based_game_keeps_the_usual_message():
    game = RealtimeFakeGame().queue_state(make_mcp_state(legal_actions={"actions": []}))

    with pytest.raises(DecisionError, match=r"choose_action raised IndexError"):
        run_game(game, RED_ALERT, _placeholder, sleep=lambda s: None)


# -- the Red Alert player ---------------------------------------------------------------------


class FakeProvider:
    model = "fake-model"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append({"messages": messages, "schema_name": schema_name, "schema": schema})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def view(self, index=-1) -> dict:
        text = self.calls[index]["messages"][1]["content"]
        return json.loads(text.split("\n")[1])


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _player(provider, clock=None):
    return redalert.RedAlertPlayer(provider, log=lambda line: None, clock=clock or Clock())


def _ctx(config=CONFIG):
    return DecisionContext(session_id="ra", tournament_id=None, game_type="red_alert", agent_id="a", game_config=config)


def _orders(*orders, reasoning="Build power."):
    return {"orders": list(orders), "reasoning_summary": reasoning}


BUILD_POWR = {"cmd": "build", "item": "powr"}
DEPLOY_GONE = {"cmd": "deploy", "units": [7]}


def test_answer_schema_is_strict_one_shape_per_command():
    schema = redalert.answer_schema(redalert.order_fields(None))
    assert schema["required"] == ["orders", "reasoning_summary"] and schema["additionalProperties"] is False
    shapes = schema["properties"]["orders"]["items"]["anyOf"]
    assert {s["properties"]["cmd"]["enum"][0] for s in shapes} == set(redalert.ORDER_FIELDS)
    for shape in shapes:
        assert shape["required"] == list(shape["properties"]) and shape["additionalProperties"] is False
    train = next(s for s in shapes if s["properties"]["cmd"]["enum"] == ["train"])
    assert train["properties"]["count"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def test_orders_go_out_with_the_models_reasoning_and_nulls_removed():
    provider = FakeProvider(_orders({"cmd": "train", "item": "e1", "count": None}, BUILD_POWR))
    move = _player(provider).choose_action(ra_state(), _ctx())
    assert isinstance(move, WithReasoning)
    assert move.action == {"type": "orders", "orders": [{"cmd": "train", "item": "e1"}, BUILD_POWR]}
    assert move.reasoning_summary == "Build power."
    assert provider.calls[0]["schema_name"] == "red_alert_orders"


def test_the_view_shows_costs_the_enemy_spawn_and_the_ready_attack():
    provider = FakeProvider(_orders(BUILD_POWR))
    _player(provider).choose_action(ra_state(), _ctx())
    view = provider.view()
    assert {"item": "powr", "queue": "Building", "cost": 300, "power": 100, "build_s": 7.2} in view["available_production"]
    assert view["map"]["enemy_spawn_guess"] == [95, 11]
    assert view["suggested_attack"]["order"] == ATTACK_NOW
    system = provider.calls[0]["messages"][0]["content"]
    assert "Deploy it once" in system and "- attack_move: units, to, queued?" in system


def test_an_empty_batch_sends_nothing():
    assert _player(FakeProvider(_orders(reasoning="Waiting."))).choose_action(ra_state(), _ctx()) is WAIT


def test_no_observation_yet_waits_without_calling_the_model():
    provider = FakeProvider()
    assert _player(provider).choose_action(ra_state(observation=None), _ctx()) is WAIT
    assert provider.calls == []


def test_malformed_orders_are_dropped_and_an_all_malformed_answer_sends_nothing():
    player = _player(FakeProvider(_orders({"cmd": "attack", "units": [120]}, {"cmd": "fly", "units": [1]})))
    assert player.choose_action(ra_state(), _ctx()) is WAIT
    assert player.fallbacks == {"bad_answer": 1}


def test_a_provider_error_sends_nothing_and_backs_off():
    clock = Clock()
    provider = FakeProvider(ProviderError("OpenAI request failed (HTTP 429)"), _orders(BUILD_POWR))
    player = _player(provider, clock)
    assert player.choose_action(ra_state(), _ctx()) is WAIT
    clock.now += 2
    assert player.choose_action(ra_state(), _ctx()) is WAIT  # still backing off: no call
    assert len(provider.calls) == 1
    clock.now += 4
    assert isinstance(player.choose_action(ra_state(), _ctx()), WithReasoning)
    assert player.fallbacks == {"error": 1, "backoff": 1}


def test_the_call_cap_stops_model_calls():
    player = redalert.RedAlertPlayer(FakeProvider(), log=lambda line: None, max_calls=0)
    assert player.choose_action(ra_state(), _ctx()) is WAIT
    assert player.fallbacks == {"cap": 1}


def test_a_refused_batch_is_shown_to_the_model_until_a_batch_is_accepted():
    provider = FakeProvider(_orders(DEPLOY_GONE), _orders(BUILD_POWR), _orders(BUILD_POWR))
    player = _player(provider)
    player.choose_action(ra_state(), _ctx())
    player.on_action_result({"error": "INVALID_ACTION", "detail": "No order was valid: [0] deploy: not_owned (unit 7 gone)"}, _ctx())
    player.choose_action(ra_state(), _ctx())
    refused = provider.view()["your_refused_batches"]
    assert refused["in_a_row"] == 1 and "not_owned" in refused["reasons_newest_last"][0]
    player.on_action_result(accepted(("build", "pending", "pending")), _ctx())
    player.choose_action(ra_state(), _ctx())
    assert "your_refused_batches" not in provider.view()


def test_an_order_refused_three_times_is_blocked_while_the_reason_holds():
    clock = Clock()
    provider = FakeProvider(*[_orders(DEPLOY_GONE, BUILD_POWR)] * 4)
    player = _player(provider, clock)
    for _ in range(3):
        player.choose_action(ra_state(), _ctx())
        player.on_action_result({"error": "INVALID_ACTION", "detail": "[0] deploy: not_owned (unit 7)"}, _ctx())
    move = player.choose_action(ra_state(), _ctx())
    assert move.action["orders"] == [BUILD_POWR]  # the deploy was dropped
    assert player.orders_blocked == 1
    repeated = provider.view()["orders_refused_repeatedly"]["orders"][0]
    assert repeated["order"] == DEPLOY_GONE and repeated["refused"] == 3 and repeated["blocked"] is True


def test_a_rejected_order_inside_an_accepted_batch_is_counted():
    player = _player(FakeProvider(_orders({"cmd": "place", "item": "powr"}, BUILD_POWR)))
    player.choose_action(ra_state(), _ctx())
    player.on_action_result(accepted(("place", "rejected", "not_ready"), ("build", "pending", "pending")), _ctx())
    assert [e["reason"] for e in player._refusals.values()] == ["not_ready"]


def test_the_prompt_stays_under_its_size_limit():
    units = [{"id": i, "type": "e1", "x": 10, "y": 10, "hp": 100, "idle": True} for i in range(500)]
    payload = ra_payload()
    payload["observation"]["units"] = units
    text = redalert.build_user_prompt(payload, None, CONFIG)
    assert len(text) <= redalert.USER_PROMPT_MAX_CHARS
    assert json.loads(text.split("\n")[1])["note"]["entries_not_shown"]["units"] > 0


# -- the example LLM agent --------------------------------------------------------------------


def test_llm_agent_hands_red_alert_to_its_player_and_forwards_results():
    provider = FakeProvider(_orders(DEPLOY_GONE), _orders(BUILD_POWR))
    agent = LLMAgent(provider, log=lambda line: None)
    agent.choose_action(ra_state(), _ctx())
    agent.on_action_result({"error": "INVALID_ACTION", "detail": "[0] deploy: not_owned (unit 7)"}, _ctx())
    agent.choose_action(ra_state(), _ctx())
    assert "your_refused_batches" in provider.view()


def test_llm_agent_refuses_a_realtime_game_it_has_no_player_for():
    agent = LLMAgent(FakeProvider(), log=lambda line: None)
    other = DecisionContext(session_id="x", tournament_id=None, game_type="honor_of_kings", agent_id="a")
    with pytest.raises(UnsupportedStructuredAction, match="honor_of_kings"):
        agent.choose_action(ra_state(), other)


def test_llm_agent_plays_a_whole_red_alert_match_through_the_runner():
    provider = FakeProvider(_orders({"cmd": "deploy", "units": [120]}), _orders(), _orders(BUILD_POWR))
    game = RealtimeFakeGame(CONFIG).queue_state(ra_state(100), ra_state(110), ra_state(115), terminal_state())
    game.queue_play_action(accepted(("deploy", "pending", "pending")), accepted(("build", "pending", "pending")))
    game.queue_result(result_dict())
    run_game(game, RED_ALERT, LLMAgent(provider, log=lambda line: None), sleep=lambda s: None)
    assert [c["action"]["orders"][0]["cmd"] for c in game.play_action_calls] == ["deploy", "build"]
    assert game.play_action_calls[0]["reasoning_summary"] == "Build power."
    assert len(game.wait_calls) == 1  # the empty batch waited for the next view


def test_a_temporary_regrant_failure_while_reading_the_config_leaves_it_unset():
    # get_game_config is best-effort: a seat re-grant that hits a control-plane
    # hiccup (it arrives as a PlatformError, not an MCPToolError) mustn't end
    # the game.
    from altruagent.errors import PlatformError

    game = RealtimeFakeGame(PlatformError("Request failed with status 503.", status_code=503))
    game.queue_state(ra_state(100), terminal_state())
    game.queue_play_action(accepted(("build", "pending", "pending"))).queue_result(result_dict())
    agent = Recorder(ORDERS)
    run_game(game, RED_ALERT, agent, sleep=lambda s: None)
    assert agent.contexts[0].game_config is None
    assert len(game.play_action_calls) == 1


def test_a_definite_refusal_while_reading_the_config_still_stops_the_game():
    from altruagent.official import OfficialAgentError

    game = RealtimeFakeGame(OfficialAgentError("held", status_code=409, error_code="seat_busy"))
    game.queue_state(ra_state(100))
    with pytest.raises(OfficialAgentError):
        run_game(game, RED_ALERT, Recorder(ORDERS), sleep=lambda s: None)
