"""Unit tests for altruagent.supervisor. No real client/network/process is
involved: a FakeClient scripts sessions() responses and a FakeProcess
stands in for multiprocessing.Process, so worker lifecycle is fully
controlled and deterministic. Real spawn compatibility is proven separately
in tests/test_worker.py.
"""

from __future__ import annotations

import pytest

from altruagent.models import AgentSessions, Match
from altruagent.supervisor import (
    WorkerRegistry,
    run_forever_concurrent,
    run_once_concurrent,
)
from altruagent.worker import EXIT_MATCH_FAILURE, EXIT_SUCCESS, WorkerInput

AGENT_ID = "agent-1"


def match(session_id: str, **overrides) -> Match:
    payload = {"session_id": session_id, "status": "in_progress", "game_type": "tic_tac_toe"}
    payload.update(overrides)
    return Match.from_dict(payload)


def sessions_with_active(*matches: Match) -> AgentSessions:
    return AgentSessions(waiting=[], active=list(matches), completed=[])


class FakeClient:
    def __init__(self, *responses: AgentSessions) -> None:
        self._responses = list(responses)
        self.calls = 0

    def sessions(self) -> AgentSessions:
        self.calls += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        return self._responses[index]


class FakeProcess:
    _next_pid = 1000

    def __init__(self, target, args=(), daemon=None) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self._alive = False
        self._exitcode = None
        self.terminated = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    @property
    def exitcode(self):
        return self._exitcode

    def join(self, timeout=None) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def finish(self, exitcode: int = 0) -> None:
        """Test helper: simulate the process completing on its own."""
        self._alive = False
        self._exitcode = exitcode


class FakeProcessFactory:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []

    def __call__(self, target, args=(), daemon=None) -> FakeProcess:
        process = FakeProcess(target, args=args, daemon=daemon)
        self.processes.append(process)
        return process


def no_sleep(_seconds: float) -> None:
    pass


def make_clock(start: float = 0.0, step: float = 1.0):
    state = {"t": start}

    def now() -> float:
        state["t"] += step
        return state["t"]

    return now


# -- WorkerRegistry -----------------------------------------------------


def test_registry_reap_finished_removes_and_reports_exitcode():
    registry = WorkerRegistry()
    process = FakeProcess(target=None)
    process.start()
    registry.start("s-1", process)

    assert registry.reap_finished() == {}  # still alive, nothing to reap yet

    process.finish(exitcode=0)

    assert registry.reap_finished() == {"s-1": 0}
    assert not registry.is_active("s-1")


def test_registry_terminate_all_stops_and_clears():
    registry = WorkerRegistry()
    p1, p2 = FakeProcess(target=None), FakeProcess(target=None)
    p1.start()
    p2.start()
    registry.start("s-1", p1)
    registry.start("s-2", p2)

    registry.terminate_all(timeout=1.0)

    assert p1.terminated and p2.terminated
    assert len(registry) == 0


def test_registry_pids_reflects_distinct_live_workers():
    registry = WorkerRegistry()
    p1, p2 = FakeProcess(target=None), FakeProcess(target=None)
    p1.start()
    p2.start()
    registry.start("s-1", p1)
    registry.start("s-2", p2)

    pids = registry.pids()

    assert set(pids) == {"s-1", "s-2"}
    assert pids["s-1"] != pids["s-2"]


# -- run_once_concurrent: discovery + spawn ------------------------------


def test_run_once_concurrent_spawns_worker_for_one_active_match():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()

    run_once_concurrent(
        client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory
    )

    assert len(factory.processes) == 1
    proc = factory.processes[0]
    assert proc.args == (
        WorkerInput(session_id="s-1", tournament_id=None, game_type="tic_tac_toe", agent_id=AGENT_ID),
    )
    assert proc.daemon is True
    assert registry.is_active("s-1")


def test_run_once_concurrent_spawns_two_workers_for_two_active_matches():
    m1, m2 = match("s-1"), match("s-2")
    client = FakeClient(sessions_with_active(m1, m2))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()

    run_once_concurrent(
        client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory
    )

    assert len(factory.processes) == 2
    assert registry.is_active("s-1") and registry.is_active("s-2")


def test_run_once_concurrent_ignores_waiting_and_completed_sessions():
    sessions = AgentSessions(waiting=[match("s-w")], active=[], completed=[match("s-c")])
    client = FakeClient(sessions)
    factory = FakeProcessFactory()

    run_once_concurrent(
        client, registry=WorkerRegistry(), failed_until={}, agent_id=AGENT_ID, process_factory=factory
    )

    assert factory.processes == []


def test_run_once_concurrent_never_starts_duplicate_worker_for_live_session():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m), sessions_with_active(m))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()

    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)
    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)

    assert len(factory.processes) == 1  # second tick: s-1 already has a live worker


def test_run_once_concurrent_starts_new_worker_while_another_still_runs():
    m1, m2 = match("s-1"), match("s-2")
    # Tick 1: only s-1 active. Tick 2: s-1 (still running) + newly-active s-2.
    client = FakeClient(sessions_with_active(m1), sessions_with_active(m1, m2))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()

    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)
    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)

    assert len(factory.processes) == 2
    assert registry.is_active("s-1") and registry.is_active("s-2")
    assert not factory.processes[0].terminated  # s-1's worker was never touched/restarted


def test_run_once_concurrent_reaps_finished_worker():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m), sessions_with_active())  # s-1 gone by tick 2
    factory = FakeProcessFactory()
    registry = WorkerRegistry()

    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)
    factory.processes[0].finish(exitcode=EXIT_SUCCESS)
    run_once_concurrent(client, registry=registry, failed_until={}, agent_id=AGENT_ID, process_factory=factory)

    assert len(registry) == 0
    assert len(factory.processes) == 1  # no new worker spawned; s-1 wasn't active anymore


# -- cooldown -------------------------------------------------------------


def test_failed_worker_enters_cooldown_and_is_not_immediately_respawned():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m), sessions_with_active(m))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()
    failed_until: dict = {}
    clock = make_clock(start=0.0, step=1.0)

    run_once_concurrent(
        client, registry=registry, failed_until=failed_until, agent_id=AGENT_ID,
        now=clock, cooldown_seconds=60.0, process_factory=factory,
    )
    factory.processes[0].finish(exitcode=EXIT_MATCH_FAILURE)
    run_once_concurrent(
        client, registry=registry, failed_until=failed_until, agent_id=AGENT_ID,
        now=clock, cooldown_seconds=60.0, process_factory=factory,
    )

    assert "s-1" in failed_until
    assert len(factory.processes) == 1  # no respawn while cooling down


def test_cooldown_expiry_permits_retry():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m))
    factory = FakeProcessFactory()
    registry = WorkerRegistry()
    failed_until = {"s-1": 5.0}  # cooldown already expired as of "now" below

    run_once_concurrent(
        client, registry=registry, failed_until=failed_until, agent_id=AGENT_ID,
        now=lambda: 10.0, process_factory=factory,
    )

    assert len(factory.processes) == 1
    assert registry.is_active("s-1")


# -- run_forever_concurrent -----------------------------------------------


def test_run_forever_concurrent_sleeps_between_ticks_and_respects_max_iterations():
    client = FakeClient(sessions_with_active())
    factory = FakeProcessFactory()
    sleeps: list[float] = []

    run_forever_concurrent(
        client, agent_id=AGENT_ID, sleep=sleeps.append, now=make_clock(),
        process_factory=factory, max_iterations=3,
    )

    assert sleeps == [15.0, 15.0, 15.0]
    assert client.calls == 3


def test_run_forever_concurrent_terminates_outstanding_workers_on_exit():
    m = match("s-1")
    client = FakeClient(sessions_with_active(m))
    factory = FakeProcessFactory()

    run_forever_concurrent(
        client, agent_id=AGENT_ID, sleep=no_sleep, now=make_clock(),
        process_factory=factory, max_iterations=1,
    )

    assert factory.processes[0].terminated  # still "running" when the bounded loop ended


def test_run_forever_concurrent_propagates_discovery_failure_and_still_exits_cleanly():
    class FailingClient:
        def sessions(self):
            raise RuntimeError("control plane unreachable")

    with pytest.raises(RuntimeError):
        run_forever_concurrent(
            FailingClient(), agent_id=AGENT_ID, sleep=no_sleep, now=make_clock(),
            process_factory=FakeProcessFactory(),
        )
