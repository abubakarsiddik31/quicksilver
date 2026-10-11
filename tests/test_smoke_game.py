"""Focused unit tests for scripts/smoke_game.py's helper logic.

Only the bounded-polling loops (`wait_for_in_progress`,
`wait_for_tournament_in_progress`) are unit-tested here — they have real
logic worth verifying (return as soon as ready, time out without looping
forever) and are easy to test with injected clock/sleep functions. The rest
of smoke_game.py is orchestration against the real platform, exercised by
running the script itself, not by mocking it here.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import smoke_game  # noqa: E402


def _clock(values):
    it = iter(values)
    return lambda: next(it)


def test_wait_for_in_progress_returns_as_soon_as_ready():
    poll_results = iter(
        [
            {"status": "waiting", "game_server_url": None},
            {"status": "in_progress", "game_server_url": "host:8000"},
        ]
    )
    sleep_calls = []

    result = smoke_game.wait_for_in_progress(
        lambda: next(poll_results),
        timeout_seconds=10.0,
        interval_seconds=1.0,
        sleep=sleep_calls.append,
        now=_clock([0.0, 1.0]),
    )

    assert result == {"status": "in_progress", "game_server_url": "host:8000"}
    assert sleep_calls == [1.0]


def test_wait_for_in_progress_returns_immediately_without_sleeping():
    # Mirrors the real platform's behavior: the join that fills the last
    # slot auto-starts the competition synchronously, so the very first
    # poll is often already in_progress.
    poll_results = iter([{"status": "in_progress", "game_server_url": "host:8000"}])
    sleep_calls = []

    result = smoke_game.wait_for_in_progress(
        lambda: next(poll_results),
        timeout_seconds=10.0,
        interval_seconds=1.0,
        sleep=sleep_calls.append,
        now=_clock([0.0]),
    )

    assert result["status"] == "in_progress"
    assert sleep_calls == []


def test_wait_for_in_progress_times_out_without_looping_forever():
    poll_results = itertools.repeat({"status": "waiting", "game_server_url": None})
    sleep_calls = []
    # now() is called once for the deadline, then once per loop iteration.
    clock = _clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    with pytest.raises(smoke_game.SmokeTestError):
        smoke_game.wait_for_in_progress(
            lambda: next(poll_results),
            timeout_seconds=5.0,
            interval_seconds=1.0,
            sleep=sleep_calls.append,
            now=clock,
        )

    # Bounded: stops once now() has advanced past the deadline, not looping
    # until the injected clock values (or poll_results) are exhausted.
    assert len(sleep_calls) <= 6


def test_wait_for_in_progress_ignores_in_progress_without_game_server_url():
    # A response that says in_progress but hasn't populated game_server_url
    # yet (e.g. a transient read-your-writes lag) must not be treated as ready.
    poll_results = iter(
        [
            {"status": "in_progress", "game_server_url": None},
            {"status": "in_progress", "game_server_url": "host:8000"},
        ]
    )
    sleep_calls = []

    result = smoke_game.wait_for_in_progress(
        lambda: next(poll_results),
        timeout_seconds=10.0,
        interval_seconds=1.0,
        sleep=sleep_calls.append,
        now=_clock([0.0, 1.0]),
    )

    assert result["game_server_url"] == "host:8000"
    assert sleep_calls == [1.0]


def test_wait_for_tournament_in_progress_returns_as_soon_as_ready():
    poll_results = iter([{"status": "waiting"}, {"status": "in_progress"}])
    sleep_calls = []

    result = smoke_game.wait_for_tournament_in_progress(
        lambda: next(poll_results),
        timeout_seconds=10.0,
        interval_seconds=1.0,
        sleep=sleep_calls.append,
        now=_clock([0.0, 1.0]),
    )

    assert result == {"status": "in_progress"}
    assert sleep_calls == [1.0]


def test_wait_for_tournament_in_progress_does_not_require_game_server_url():
    # Unlike wait_for_in_progress, a tournament dict with no game_server_url
    # at all is still "ready" once status flips — the tournament smoke flow
    # resolves the child match's URL separately, via client.sessions().
    poll_results = iter([{"status": "in_progress"}])
    sleep_calls = []

    result = smoke_game.wait_for_tournament_in_progress(
        lambda: next(poll_results),
        timeout_seconds=10.0,
        interval_seconds=1.0,
        sleep=sleep_calls.append,
        now=_clock([0.0]),
    )

    assert result["status"] == "in_progress"
    assert sleep_calls == []


def test_wait_for_tournament_in_progress_times_out_without_looping_forever():
    poll_results = itertools.repeat({"status": "waiting"})
    sleep_calls = []
    clock = _clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    with pytest.raises(smoke_game.SmokeTestError):
        smoke_game.wait_for_tournament_in_progress(
            lambda: next(poll_results),
            timeout_seconds=5.0,
            interval_seconds=1.0,
            sleep=sleep_calls.append,
            now=clock,
        )

    assert len(sleep_calls) <= 6
