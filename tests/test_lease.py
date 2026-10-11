"""Tests for the official seat execution lease, per Agent_ACP's final contract
(backend/src/routes/tournament.ts, services/officialAgentConnectionService.ts):

- ``POST .../:seatId/grant {execution_id}`` acquires the lease if absent or
  expired (``409 seat_busy`` if another execution holds it), extends it, and
  mints a SeatGrant.
- ``POST .../:seatId/lease/renew {execution_id}`` only extends a still-active
  lease this execution owns (``409 lease_not_held`` otherwise); mints nothing.
- The lease lasts 30 s and belongs to the execution_id, not the agent session.
"""

from __future__ import annotations

import json
import threading
import time
import types

import httpx
import pytest

import altruagent.mcp_game as mcp_game_module
from altruagent.errors import PlatformError
from altruagent.official import (
    LEASE_RENEW_SECONDS,
    LEASE_RETRY_SECONDS,
    OfficialAgentClient,
    OfficialAgentError,
    OfficialSeatAuth,
    SeatLeaseKeeper,
    SeatLeaseLost,
    new_execution_id,
)
from altruagent.supervisor import SEAT_BUSY_RETRY_SECONDS
from altruagent.worker import EXIT_SEAT_BUSY, EXIT_SUCCESS, TournamentWorkerInput, run_tournament_worker
from test_official import CONTROL, KEY, grant
from test_tournament import Harness, _grant, a

LEASE_TTL = 30.0
EXEC_A = "exec-a-" + "1" * 30
EXEC_B = "exec-b-" + "2" * 30
BUSY = OfficialAgentError("busy", status_code=409, error_code="seat_busy")
NOT_HELD = OfficialAgentError("not held", status_code=409, error_code="lease_not_held")


class ScriptedOfficial:
    """Scripted renew_lease/grant results (the last of each repeats)."""

    def __init__(self, renewals=(None,), grants=(None,)):
        self.renew_results, self.grant_results = list(renewals), list(grants)
        self.renew_calls: list[tuple[str, str]] = []
        self.grant_calls: list[tuple[str, str]] = []
        self.control_url = CONTROL

    def _next(self, results, calls):
        result = results[min(len(calls) - 1, len(results) - 1)]
        if isinstance(result, BaseException):
            raise result
        return result

    def renew_lease(self, seat_id, execution_id):
        self.renew_calls.append((seat_id, execution_id))
        return self._next(self.renew_results, self.renew_calls) or {"seat_id": seat_id}

    def grant(self, seat_id, execution_id):
        self.grant_calls.append((seat_id, execution_id))
        return self._next(self.grant_results, self.grant_calls) or _grant(access_token=f"minted-{len(self.grant_calls)}")

    def close(self):
        pass


def keeper(official, **kwargs):
    log = []
    return SeatLeaseKeeper(official, "seat-1", EXEC_A, log=log.append, **kwargs), log


# -- keeper -------------------------------------------------------------------------------


def test_routine_renewal_uses_lease_renew_not_grant():
    official = ScriptedOfficial()
    k, log = keeper(official)

    assert [k.renew_once() for _ in range(3)] == [LEASE_RENEW_SECONDS] * 3
    assert official.renew_calls == [("seat-1", EXEC_A)] * 3
    assert official.grant_calls == []
    assert k.renewals == 3 and log == []


def test_renewal_schedule_stays_inside_the_lease_ttl_even_with_failures():
    assert LEASE_RENEW_SECONDS == 10.0
    assert LEASE_RENEW_SECONDS + 3 * LEASE_RETRY_SECONDS < LEASE_TTL


def test_transient_failure_retries_sooner_logs_once_and_recovers():
    official = ScriptedOfficial(renewals=(PlatformError("503", status_code=503), httpx.ConnectError("down"), None))
    k, log = keeper(official)

    assert [k.renew_once() for _ in range(3)] == [LEASE_RETRY_SECONDS, LEASE_RETRY_SECONDS, LEASE_RENEW_SECONDS]
    assert not k.lost.is_set() and official.grant_calls == []
    assert len([line for line in log if "renewal failed" in line]) == 1
    assert log[-1] == "Seat lease renewal recovered."


def test_lapsed_lease_is_reacquired_with_the_same_execution_id():
    official = ScriptedOfficial(renewals=(NOT_HELD,))
    k, log = keeper(official)

    assert k.renew_once() == LEASE_RENEW_SECONDS
    assert official.grant_calls == [("seat-1", EXEC_A)]
    assert not k.lost.is_set() and log == ["Seat lease re-acquired."]


def test_lease_taken_by_another_runtime_is_lost():
    official = ScriptedOfficial(renewals=(NOT_HELD,), grants=(BUSY,))
    k, log = keeper(official)

    assert k.renew_once() is None
    assert k.lost.is_set() and "Lost this seat" in log[0]


def test_seat_busy_from_renew_is_lost():
    k, _ = keeper(ScriptedOfficial(renewals=(BUSY,)))

    assert k.renew_once() is None and k.lost.is_set()


def test_failed_reacquire_is_transient():
    k, _ = keeper(ScriptedOfficial(renewals=(NOT_HELD,), grants=(PlatformError("500", status_code=500),)))

    assert k.renew_once() == LEASE_RETRY_SECONDS and not k.lost.is_set()


@pytest.mark.parametrize("code", ["assignment_not_grantable", "assignment_not_found"])
def test_ended_assignment_stops_renewing_without_losing(code):
    k, _ = keeper(ScriptedOfficial(renewals=(OfficialAgentError("ended", status_code=409, error_code=code),)))

    assert k.renew_once() is None and not k.lost.is_set()


def test_background_thread_renews_periodically_and_stops_on_shutdown():
    official = ScriptedOfficial()
    k, _ = keeper(official, renew_seconds=0.01)

    k.start()
    deadline = time.monotonic() + 2
    while len(official.renew_calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    k.stop()
    calls_at_stop = len(official.renew_calls)
    time.sleep(0.05)

    assert calls_at_stop >= 3
    assert len(official.renew_calls) == calls_at_stop
    assert not k._thread.is_alive()
    assert EXEC_A not in repr(k)


def test_execution_ids_are_random_opaque_and_in_bounds():
    ids = {new_execution_id() for _ in range(50)}
    assert len(ids) == 50 and all(16 <= len(i) <= 256 for i in ids)


# -- against a mock backend with the real lease rule ------------------------------------------


class LeaseBackend:
    def __init__(self):
        self.now = 0.0
        self.sessions = 0
        self.valid: set[str] = set()
        self.owner: str | None = None
        self.expires = 0.0
        self.minted = 0
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, request):
        with self.lock:
            path = request.url.path
            if path == "/tournament/agent/authenticate":
                self.sessions += 1
                token = f"sess-{self.sessions}"
                self.valid.add(token)
                return httpx.Response(200, json={"access_token": token, "expires_at": "x"})
            if (request.headers.get("authorization") or "").removeprefix("Bearer ") not in self.valid:
                return httpx.Response(401, json={"error": "invalid_agent_session"})
            execution_id = (json.loads(request.content) or {}).get("execution_id")
            active = self.now < self.expires
            if path.endswith("/lease/renew"):
                self.calls.append(("renew", execution_id))
                if not active or self.owner != execution_id:
                    return httpx.Response(409, json={"error": "lease_not_held"})
                self.expires = self.now + LEASE_TTL
                return httpx.Response(200, json={"seat_id": "seat-1", "execution_lease_expires_at": "x"})
            self.calls.append(("grant", execution_id))
            if active and self.owner != execution_id:
                return httpx.Response(409, json={"error": "seat_busy"})
            self.owner, self.expires = execution_id, self.now + LEASE_TTL
            self.minted += 1
            return httpx.Response(200, json=grant(f"seat-jwt-{self.minted}"))


def client(backend):
    return OfficialAgentClient(CONTROL, KEY, load_env_file=False, transport=httpx.MockTransport(backend))


def test_renewal_keeps_the_lease_without_minting_and_blocks_other_runtimes():
    backend = LeaseBackend()
    runtime_a, runtime_b = client(backend), client(backend)
    runtime_a.grant("seat-1", EXEC_A)
    keeper_a = SeatLeaseKeeper(runtime_a, "seat-1", EXEC_A, log=lambda m: None)

    for _ in range(6):  # a minute of healthy 10 s renewals
        backend.now += LEASE_RENEW_SECONDS
        assert keeper_a.renew_once() == LEASE_RENEW_SECONDS
    assert backend.minted == 1  # keepalive never minted a SeatGrant

    with pytest.raises(OfficialAgentError) as exc_info:
        runtime_b.grant("seat-1", EXEC_B)
    assert exc_info.value.error_code == "seat_busy"


def test_session_reauthentication_does_not_self_evict():
    backend = LeaseBackend()
    runtime = client(backend)
    runtime.grant("seat-1", EXEC_A)
    k = SeatLeaseKeeper(runtime, "seat-1", EXEC_A, log=lambda m: None)

    backend.valid.clear()  # the agent session expires
    backend.now += LEASE_RENEW_SECONDS

    assert k.renew_once() == LEASE_RENEW_SECONDS and not k.lost.is_set()
    assert backend.sessions == 2 and backend.owner == EXEC_A


def test_other_runtime_takes_over_only_after_the_lease_lapses_then_the_first_loses_it():
    backend = LeaseBackend()
    runtime_a, runtime_b = client(backend), client(backend)
    runtime_a.grant("seat-1", EXEC_A)
    keeper_a = SeatLeaseKeeper(runtime_a, "seat-1", EXEC_A, log=lambda m: None)

    backend.now += LEASE_TTL + 1  # A went silent long enough to lapse
    runtime_b.grant("seat-1", EXEC_B)
    backend.now += 1

    assert keeper_a.renew_once() is None
    assert keeper_a.lost.is_set()
    assert backend.owner == EXEC_B


def test_lapsed_lease_nobody_took_is_reacquired_in_place():
    backend = LeaseBackend()
    runtime = client(backend)
    runtime.grant("seat-1", EXEC_A)
    k = SeatLeaseKeeper(runtime, "seat-1", EXEC_A, log=lambda m: None)

    backend.now += LEASE_TTL + 5  # e.g. a long network outage

    assert k.renew_once() == LEASE_RENEW_SECONDS and not k.lost.is_set()
    assert backend.calls[-2:] == [("renew", EXEC_A), ("grant", EXEC_A)]


def test_gameapi_401_regrant_uses_the_same_execution_id():
    backend = LeaseBackend()
    seat_auth = OfficialSeatAuth(client(backend), "seat-1", EXEC_A)

    seat_auth.login(None, CONTROL)
    backend.now += 5
    seat_auth.login(None, CONTROL)  # what a GameAPI 401 triggers

    assert backend.calls == [("grant", EXEC_A), ("grant", EXEC_A)]
    assert seat_auth.grant.access_token == "seat-jwt-2"


def test_concurrent_renewal_and_regrant_share_one_session():
    backend = LeaseBackend()
    runtime = client(backend)
    runtime.grant("seat-1", EXEC_A)
    backend.valid.clear()
    errors = []

    def call(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(lambda: runtime.renew_lease("seat-1", EXEC_A),)),
               threading.Thread(target=call, args=(lambda: runtime.grant("seat-1", EXEC_A),))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == [] and backend.sessions == 2  # exactly one re-authentication


# -- worker ----------------------------------------------------------------------------------------


def _run_worker(official, run_game_fn, **kwargs):
    return run_tournament_worker(
        TournamentWorkerInput("seat-1", "match-1", "pokemon_vgc_doubles_draft", "agent.agent:create_agent", EXEC_A),
        official_factory=lambda: official, run_game_fn=run_game_fn, **kwargs,
    )


def _finished():
    return types.SimpleNamespace(termination_reason="normal", returns={})


def test_worker_grants_once_renews_periodically_and_keeps_its_gameapi_token():
    official = ScriptedOfficial()
    seen = {}

    def playing(game, context, contestant):
        deadline = time.monotonic() + 2
        while len(official.renew_calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        seen["token"] = game._client._access_token
        return _finished()

    code = _run_worker(official, playing, lease_renew_seconds=0.01)
    renewals_at_exit = len(official.renew_calls)
    time.sleep(0.05)

    assert code == EXIT_SUCCESS
    assert official.grant_calls == [("seat-1", EXEC_A)]  # only the initial acquisition
    assert renewals_at_exit >= 3 and set(official.renew_calls) == {("seat-1", EXEC_A)}
    assert seen["token"] == "minted-1"  # renewal never replaced the GameAPI SeatGrant
    assert len(official.renew_calls) == renewals_at_exit  # renewal stopped with the worker


def test_worker_that_loses_its_seat_stops_without_further_gameplay_calls(monkeypatch):
    official = ScriptedOfficial(renewals=(NOT_HELD,), grants=(None, BUSY))
    gameplay_calls = []
    monkeypatch.setattr(mcp_game_module, "call_tool", lambda *args: gameplay_calls.append(args) or {})

    def playing(game, context, contestant):
        deadline = time.monotonic() + 2
        while not game._lost.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        game.get_state()
        return _finished()

    assert _run_worker(official, playing, lease_renew_seconds=0.01) == EXIT_SEAT_BUSY
    assert gameplay_calls == []


def test_guarded_session_raises_seat_lease_lost():
    raised = {}

    def playing(game, context, contestant):
        game._lost.set()
        try:
            game.play_action(action_id="x", state_version=1)
        except SeatLeaseLost as exc:
            raised["exc"] = exc
            raise

    assert _run_worker(ScriptedOfficial(), playing) == EXIT_SEAT_BUSY
    assert isinstance(raised["exc"], SeatLeaseLost)


def test_worker_whose_seat_is_already_held_does_not_play():
    runs = []

    assert _run_worker(ScriptedOfficial(grants=(BUSY,)), lambda *args: runs.append(args)) == EXIT_SEAT_BUSY
    assert runs == []


# -- supervisor ------------------------------------------------------------------------------------


def test_seat_busy_stops_only_that_seat_and_retries_after_the_lease_can_lapse():
    h = Harness([a("seat-1"), a("seat-2")], cooldown=60.0)
    h.tick()
    busy, other = h.factory.processes
    busy.finish(EXIT_SEAT_BUSY)

    h.tick(advance=1.0)
    assert h.state.registry.is_active("seat-2") and not other.terminated
    assert any("Another runtime is playing this match" in line for line in h.log)

    h.tick(advance=SEAT_BUSY_RETRY_SECONDS)
    assert h.started() == ["seat-1", "seat-2", "seat-1"]
    assert h.state.registry.is_active("seat-2")
    assert SEAT_BUSY_RETRY_SECONDS > LEASE_TTL
