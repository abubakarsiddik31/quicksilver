"""Unit tests for altruagent.worker.run_worker. All dependencies are
injected fakes — no real client/network/process — except one small,
separate test that proves genuine Windows-`spawn`-compatible multiprocessing
actually works for a top-level target function (bottom of file).
"""

from __future__ import annotations

import multiprocessing
import types

import pytest

from altruagent.errors import PlatformError
from altruagent.models import GameState
from altruagent.runner import RESIGN, TERMINATE_MESSAGING, DecisionError, UnsupportedGameFlowError
from altruagent.worker import (
    EXIT_MATCH_FAILURE,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    WorkerInput,
    run_worker,
)

WORKER_INPUT = WorkerInput(
    session_id="s-1", tournament_id=None, game_type="tic_tac_toe", agent_id="agent-1"
)


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def terminal_state(**overrides) -> GameState:
    payload = {
        "session_id": "s-1",
        "game_name": "tic_tac_toe",
        "status": "finished",
        "observation": "...",
        "current_player": None,
        "legal_actions": [],
        "legal_actions_str": [],
        "is_terminal": True,
        "returns": {"Me": 1.0},
        "move_count": 3,
        "termination_reason": "completed",
        "messaging_enabled": False,
        "phase": "moving",
        "next_actions": [{"action": "game_over", "hint": "done"}],
    }
    payload.update(overrides)
    return GameState.from_dict(payload)


def make_agent_module(**attrs) -> object:
    return types.SimpleNamespace(**attrs)


def test_run_worker_happy_path_wires_client_match_and_create_agent_together():
    fake_client = FakeClient()
    create_agent_calls = []

    def choose_action(state, context):
        return 0

    def create_agent():
        create_agent_calls.append(1)
        return choose_action

    match_calls = []

    def match_factory(data, *, client):
        match_calls.append((data, client))
        return "fake-match"

    run_match_calls = []

    def run_match_fn(match, agent_id, decision_fn):
        run_match_calls.append((match, agent_id, decision_fn))
        return terminal_state()

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: fake_client,
        match_factory=match_factory,
        agent_module=make_agent_module(create_agent=create_agent),
        run_match_fn=run_match_fn,
    )

    assert exit_code == EXIT_SUCCESS
    assert len(create_agent_calls) == 1
    assert match_calls == [
        (
            {
                "session_id": "s-1",
                "tournament_id": None,
                "game_type": "tic_tac_toe",
                "status": "in_progress",
            },
            fake_client,
        )
    ]
    assert run_match_calls == [("fake-match", "agent-1", choose_action)]
    assert fake_client.closed is True


def test_run_worker_missing_create_agent_fails_clearly_without_touching_client():
    client_built = []

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: client_built.append(1) or FakeClient(),
        agent_module=make_agent_module(),  # no create_agent attribute at all
    )

    assert exit_code == EXIT_UNEXPECTED
    assert client_built == []  # failed before ever building a client


def test_run_worker_non_callable_create_agent_fails_clearly():
    exit_code = run_worker(
        WORKER_INPUT,
        agent_module=make_agent_module(create_agent="not callable"),
    )

    assert exit_code == EXIT_UNEXPECTED


@pytest.mark.parametrize(
    "error",
    [
        DecisionError("bad decision"),
        UnsupportedGameFlowError("messaging"),
        PlatformError("boom", status_code=500, error_code="server_error"),
    ],
)
def test_run_worker_match_scoped_errors_return_match_failure_and_close_client(error):
    fake_client = FakeClient()

    def run_match_fn(match, agent_id, decision_fn):
        raise error

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: fake_client,
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=lambda: (lambda s, c: 0)),
        run_match_fn=run_match_fn,
    )

    assert exit_code == EXIT_MATCH_FAILURE
    assert fake_client.closed is True


def test_run_worker_unexpected_exception_returns_unexpected_and_closes_client():
    # Includes what an AuthenticationError inside one worker looks like:
    # caught here as "unexpected" (cooldown-worthy), never propagated to
    # kill the whole supervisor — a single worker's auth trouble isn't
    # assumed to be a global problem (see altruagent.supervisor).
    fake_client = FakeClient()

    def run_match_fn(match, agent_id, decision_fn):
        raise ValueError("something else broke")

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: fake_client,
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=lambda: (lambda s, c: 0)),
        run_match_fn=run_match_fn,
    )

    assert exit_code == EXIT_UNEXPECTED
    assert fake_client.closed is True


def test_run_worker_keyboard_interrupt_exits_cleanly_as_success():
    fake_client = FakeClient()

    def run_match_fn(match, agent_id, decision_fn):
        raise KeyboardInterrupt()

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: fake_client,
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=lambda: (lambda s, c: 0)),
        run_match_fn=run_match_fn,
    )

    assert exit_code == EXIT_SUCCESS
    assert fake_client.closed is True


def test_run_worker_resign_decision_passed_through_to_run_match():
    run_match_calls = []

    def create_agent():
        return lambda state, context: RESIGN

    def run_match_fn(match, agent_id, decision_fn):
        run_match_calls.append(decision_fn(None, None))
        return terminal_state(termination_reason="resignation")

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: FakeClient(),
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=create_agent),
        run_match_fn=run_match_fn,
    )

    assert exit_code == EXIT_SUCCESS
    assert run_match_calls == [RESIGN]


# -- fresh-instance semantics ------------------------------------------------
# Two separate run_worker calls stand in for two separate worker PROCESSES —
# each would independently call create_agent() exactly once, exactly like this.


def test_two_worker_invocations_get_distinct_contestant_instances():
    created_instances = []

    class StatefulAgent:
        def __init__(self) -> None:
            self.history = []

        def choose_action(self, state, context):
            self.history.append(state)
            return 0

    def create_agent():
        instance = StatefulAgent()
        created_instances.append(instance)
        return instance

    def make_run_match_fn(marker):
        def run_match_fn(match, agent_id, decision_fn):
            # run_worker deliberately passes the raw create_agent() result
            # through unchanged (an object here, not directly callable) and
            # relies on run_match's own duck-typed resolution — mirror that
            # same one-line resolution here rather than assuming decision_fn
            # is always a plain callable.
            resolved = decision_fn if callable(decision_fn) else decision_fn.choose_action
            resolved(marker, None)
            return terminal_state()

        return run_match_fn

    run_worker(
        WorkerInput(session_id="s-1", tournament_id=None, game_type="tic_tac_toe", agent_id="agent-1"),
        client_factory=lambda: FakeClient(),
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=create_agent),
        run_match_fn=make_run_match_fn("s-1"),
    )
    run_worker(
        WorkerInput(session_id="s-2", tournament_id=None, game_type="tic_tac_toe", agent_id="agent-1"),
        client_factory=lambda: FakeClient(),
        match_factory=lambda data, *, client: "fake-match",
        agent_module=make_agent_module(create_agent=create_agent),
        run_match_fn=make_run_match_fn("s-2"),
    )

    assert len(created_instances) == 2
    assert created_instances[0] is not created_instances[1]
    assert created_instances[0].history == ["s-1"]
    assert created_instances[1].history == ["s-2"]  # no cross-contamination


# -- messaging survives the worker's object-passing path (Milestone 5) ------


def mcp_messaging_state_payload(**overrides) -> dict:
    payload = {
        "session_id": "s-1",
        "game_type": "repeated_pd",
        "status": "in_progress",
        "state_version": 0,
        "observation": "...",
        "phase": "messaging",
        "messaging_enabled": True,
        "terminated_messaging": [],
        "new_messages": [],
        "current_actor": {"agent_id": "agent-1", "position": 0},
        "is_current_actor": True,
        "is_terminal": False,
    }
    payload.update(overrides)
    return payload


class FakeMCPGameSessionForWorker:
    """Minimal MCPGameSession double so this test can exercise the REAL
    ``run_match``/``run_game`` (not a faked ``run_match_fn``) — proving a
    contestant object's ``choose_message`` survives run_worker's
    unmodified object-passing all the way through a real messaging round,
    through the MCP transport specifically (never REST).
    """

    def __init__(self) -> None:
        self._queue = [
            mcp_messaging_state_payload(),
            mcp_messaging_state_payload(phase="moving", status="completed", is_terminal=True),
        ]
        self.terminate_messaging_calls = 0

    def get_state(self) -> GameState:
        return GameState.from_mcp_state(self._queue[0])

    def get_legal_actions(self) -> dict:
        raise AssertionError("get_legal_actions should not be called during a messaging round")

    def play_action(self, *, action_id=None, action=None, state_version):
        raise AssertionError("play_action should not be called during a messaging round")

    def send_message(self, *, message_type: str, content=None, recipients=None) -> dict:
        if message_type != "terminate":
            raise AssertionError("this test's agent always terminates, never chats")
        self.terminate_messaging_calls += 1
        self._queue.pop(0)
        return {"accepted": True, "session_id": "s-1", "phase": self._queue[0]["phase"]}

    def get_messages(self, *, since: int = -1) -> dict:
        raise AssertionError("not exercised by this test")

    def resign(self) -> dict:
        raise AssertionError("resign not exercised by this test")

    def get_result(self) -> dict:
        return {
            "session_id": "s-1",
            "is_terminal": True,
            "status": "completed",
            "returns": {"Me": 0.0, "Them": 0.0},
            "your_return": 0.0,
            "termination_reason": "completed",
        }


class FakeMatchForWorker:
    def __init__(self, game_session: FakeMCPGameSessionForWorker) -> None:
        self.session_id = "s-1"
        self.tournament_id = None
        self.game_type = "repeated_pd"
        self._game_session = game_session

    def game(self) -> FakeMCPGameSessionForWorker:
        return self._game_session


def test_run_worker_supports_choose_message_through_real_run_match():
    class NegotiatingAgent:
        def __init__(self) -> None:
            self.choose_message_calls = 0

        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            self.choose_message_calls += 1
            return TERMINATE_MESSAGING

    agent = NegotiatingAgent()
    fake_game = FakeMCPGameSessionForWorker()

    exit_code = run_worker(
        WORKER_INPUT,
        client_factory=lambda: FakeClient(),
        match_factory=lambda data, *, client: FakeMatchForWorker(fake_game),
        agent_module=make_agent_module(create_agent=lambda: agent),
    )

    assert exit_code == EXIT_SUCCESS
    assert agent.choose_message_calls == 1
    assert fake_game.terminate_messaging_calls == 1


# -- real multiprocessing spawn compatibility (minimal, targeted) -----------


def _spawn_probe_target(value: int) -> None:
    """Top-level, importable-by-name target — proves the exact mechanism
    ``_process_entry`` relies on (a plain module-level function + a
    picklable argument, run under the explicit 'spawn' context, surfacing
    its result via SystemExit -> Process.exitcode) actually works, without
    touching any real SDK/network code inside the child.
    """
    raise SystemExit(value + 1)


def test_multiprocessing_spawn_compatibility_with_top_level_target():
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_spawn_probe_target, args=(41,))
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 42
