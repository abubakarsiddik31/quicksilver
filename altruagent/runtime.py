"""Legacy sequential local discovery + execution loop.

Owns exactly what Milestone 4A's single-match runner (``altruagent.runner``)
deliberately didn't: repeatedly discovering which match to play via
``client.sessions()``, and feeding matches into ``run_match()`` one at a
time, forever (or until interrupted). It does not implement concurrency —
only one match is ever being played at any moment.

No new game-loop logic lives here — every actual decision/state/step call
still goes through the committed ``run_match()``. This module is only
responsible for *which* match to hand it next, and what to do when that
match finishes or fails. It remains exported as a bounded/manual primitive;
the normal ``python -m agent`` entry point uses ``altruagent.supervisor``'s
concurrent process-per-match runtime instead.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from .errors import PlatformError
from .runner import DecisionError, UnsupportedGameFlowError
from .runner import run_match as _default_run_match

if TYPE_CHECKING:
    from .client import AltruAgentClient
    from .models import GameState

DEFAULT_DISCOVERY_INTERVAL_SECONDS = 15.0
# No platform-imposed number exists for this — chosen to be comfortably
# longer than the default discovery interval, so a deterministically
# broken match isn't retried on essentially every tick.
DEFAULT_COOLDOWN_SECONDS = 60.0

# Errors scoped to the one match that was being played — a contestant logic
# bug (DecisionError), a match this runner can't drive yet
# (UnsupportedGameFlowError), or a GameAPI-level issue for that one session
# (PlatformError). None of these mean the runtime itself is broken, so they
# don't propagate — the match is just skipped for a cooldown period.
# AuthenticationError (and anything else unrecognized) is deliberately NOT
# caught here: a broken JWT/API key affects every match identically, so
# that's a clearly global problem the process should exit on, not loop past.
_MATCH_SCOPED_ERRORS = (DecisionError, UnsupportedGameFlowError, PlatformError)


def _log(message: str) -> None:
    print(f"[agent] {message}")


def run_once(
    client: "AltruAgentClient",
    choose_action,
    *,
    agent_id: str,
    failed_until: dict,
    now: Callable[[], float] = time.monotonic,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    run_match_fn: Callable[..., "GameState"] = _default_run_match,
) -> bool:
    """One discovery tick: fetch this agent's sessions, pick at most one
    eligible active match, and play it to completion via ``run_match_fn``.

    Only ``sessions.active`` is ever considered — ``waiting`` matches have
    no GameAPI session to open yet (so ``Match.game()`` is never called for
    them), and ``completed`` matches are just history. Both are skipped by
    construction, not by an explicit check.

    Matches are considered in the order ``client.sessions().active`` already
    returns them — currently the backend's own ``joined_at DESC`` ordering
    (most recently joined first, see ``db/competitions.ts``'s
    ``getCompetitionsForAgent``) — no additional sorting is applied. The
    first eligible match not currently in cooldown is serviced; this makes
    the choice deterministic for a given sessions() response.

    ``failed_until`` is a plain ``dict[str, float]`` the *caller* owns and
    threads across repeated calls (``run_forever`` does this) — mapping
    ``session_id`` to the monotonic timestamp before which that match
    should be skipped, following a previous failure. Purely in-memory,
    never persisted.

    Returns ``True`` if a match was serviced this call (whether it
    succeeded or failed), ``False`` if nothing was eligible right now.
    """
    sessions = client.sessions()
    current_time = now()

    match = None
    for candidate in sessions.active:
        retry_at = failed_until.get(candidate.session_id)
        if retry_at is not None and current_time < retry_at:
            continue
        match = candidate
        break

    if match is None:
        return False

    _log(
        f"starting match session_id={match.session_id} game_type={match.game_type} "
        f"tournament_id={match.tournament_id}"
    )
    try:
        final_state = run_match_fn(match, agent_id, choose_action)
    except _MATCH_SCOPED_ERRORS as exc:
        failed_until[match.session_id] = current_time + cooldown_seconds
        _log(
            f"match failed, skipping session_id={match.session_id} for "
            f"{cooldown_seconds:.0f}s: {exc}"
        )
        return True

    failed_until.pop(match.session_id, None)
    _log(
        f"match finished session_id={match.session_id} "
        f"termination_reason={final_state.termination_reason}"
    )
    return True


def run_forever(
    client: "AltruAgentClient",
    choose_action,
    *,
    agent_id: str,
    discovery_interval: float = DEFAULT_DISCOVERY_INTERVAL_SECONDS,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    run_match_fn: Callable[..., "GameState"] = _default_run_match,
    max_iterations: int | None = None,
) -> None:
    """Repeatedly discover and play matches, one at a time, until
    interrupted (e.g. Ctrl+C — this function does not catch
    ``KeyboardInterrupt``; that's the caller's job, see ``agent/__main__.py``).

    Each tick calls ``run_once``. If it serviced a match, the next tick
    starts immediately (there may be another active match waiting); if
    nothing was eligible, sleeps ``discovery_interval`` before checking
    again. Sequential only — this never plays more than one match at a time.

    ``max_iterations`` bounds the loop to that many discovery ticks instead
    of running forever — the same parameter a production caller would just
    never pass, used by tests to make this loop terminate deterministically.
    """
    failed_until: dict = {}
    ticks = 0
    while max_iterations is None or ticks < max_iterations:
        serviced = run_once(
            client,
            choose_action,
            agent_id=agent_id,
            failed_until=failed_until,
            now=now,
            cooldown_seconds=cooldown_seconds,
            run_match_fn=run_match_fn,
        )
        ticks += 1
        if not serviced:
            _log(f"no active matches — checking again in {discovery_interval:.0f}s")
            sleep(discovery_interval)
