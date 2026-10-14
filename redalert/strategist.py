"""Event-driven LLM strategist harness for Red Alert.

The fast loop (``RedAlertAgent.choose_action``) answers in milliseconds and
never waits on a model. This class runs the model in a background thread and
publishes *plans* (:mod:`redalert.plan`) the loop then executes. When the model
is consulted is decided here, by events rather than a fixed timer:

- the match start, and every ``review_in_s`` the model asked for itself;
- being attacked, losing a unit or building, a first enemy sighting, low power,
  a new tech building finishing, the endgame clock;
- the harness itself reporting it is stuck or idle (a wanted item that is not
  available, everything satisfied, cash floating) or an order being refused.

Calls never overlap. Triggers that arrive while one is running are coalesced
and fire the instant it finishes, against a fresh state. A minimum gap, a
per-kind cooldown, an error backoff and a total call cap bound the cost. Every
failure leaves the previous plan in force; with no model the default plan runs
for the whole match.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from common.llm import GeminiClient
from redalert import trace
from redalert.plan import DEFAULT_PLAN, Plan, parse_plan
from redalert.prompt import build_system, build_user

MIN_GAP_S = 1.5
MODEL_TIMEOUT_S = 20.0
MAX_CALLS = 800
BACKOFF_FIRST_S, BACKOFF_MAX_S = 5.0, 60.0
ENDGAME_S = 150
HISTORY = 5

# Per-kind cooldown before the same kind of event can trigger again.
EVENT_COOLDOWN_S = {
    "under_attack": 8.0,
    "unit_lost": 15.0,
    "building_lost": 3.0,
    "enemy_spotted": 10.0,
    "low_power": 10.0,
}
# Only buildings that unlock something new are worth a consult; the rest is progress narration.
TECH_BUILDINGS = {"weap", "fix", "dome", "atek", "stek", "afld", "hpad", "syrd", "spen"}


class Strategist:
    def __init__(
        self,
        client: GeminiClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        min_gap: float = MIN_GAP_S,
        max_calls: int = MAX_CALLS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self._clock = clock
        self._min_gap = min_gap
        self._max_calls = max_calls
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self._plan = DEFAULT_PLAN
        self._version = 0
        self._plan_at = clock()
        self._busy = False
        self._last_start = -1e9
        self._calls = 0
        self._errors = 0
        self._backoff_until = 0.0
        self._pending: list[str] = []
        self._started = False
        self._last_seq = -1
        self._kind_at: dict[str, float] = {}
        self._seen_diag: set[str] = set()
        self._seen_refusals: set[str] = set()
        self._cap_logged = False
        self._endgame_sent = False
        self._system: str | None = None
        self._history: list[str] = []
        self._latency: float | None = None

    # -- read side (called every decision) ---------------------------------------

    def current(self) -> Plan:
        with self._lock:
            return self._plan

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def notify(self, reason: str) -> None:
        """Ask for a consult for an outside reason (e.g. Jev is unsure or sees a threat)."""
        with self._lock:
            self._add(reason)

    # -- triggers ----------------------------------------------------------------

    def _add(self, reason: str) -> None:
        if reason not in self._pending and len(self._pending) < 8:
            self._pending.append(reason)

    def _detect(
        self, raw: dict[str, Any], diagnostics: list[str], refusals: list[str], now: float
    ) -> None:
        obs = raw.get("observation") or {}
        if not self._started:
            self._started = True
            self._add("match_start")
        events = [e for e in (obs.get("events") or []) if isinstance(e, dict)]
        seqs = [e["seq"] for e in events if isinstance(e.get("seq"), int)]
        first_read = self._last_seq < 0
        for e in events:
            seq = e.get("seq")
            if not isinstance(seq, int) or seq <= self._last_seq or first_read:
                continue
            kind = str(e.get("kind"))
            if kind == "building_created":
                typ = str(e.get("type", "")).lower()
                if typ in TECH_BUILDINGS:
                    self._add(f"finished_building:{typ}")
                continue
            cooldown = EVENT_COOLDOWN_S.get(kind)
            if cooldown is None:
                continue
            if now - self._kind_at.get(kind, -1e9) >= cooldown:
                self._kind_at[kind] = now
                self._add(f"event:{kind}" + (f":{e['type']}" if e.get("type") else ""))
        if seqs:
            self._last_seq = max(self._last_seq, max(seqs))
        elif first_read:
            self._last_seq = 0
        fresh = [d for d in diagnostics if d not in self._seen_diag]
        self._seen_diag = set(diagnostics)
        if fresh and now - self._kind_at.get("harness", -1e9) >= 15.0:
            self._kind_at["harness"] = now
            for d in fresh[:2]:
                self._add(f"harness:{d}")
        new_refusals = [r for r in refusals if r not in self._seen_refusals]
        self._seen_refusals = set(refusals)
        if new_refusals and now - self._kind_at.get("refusal", -1e9) >= 10.0:
            self._kind_at["refusal"] = now
            self._add("refusals")
        remaining = (raw.get("time") or {}).get("remaining_seconds")
        if (
            isinstance(remaining, (int, float))
            and remaining <= ENDGAME_S
            and not self._endgame_sent
        ):
            self._endgame_sent = True
            self._add(f"endgame:{int(remaining)}s_left")
        if not self._busy and now >= self._plan_at + self._plan.review_in_s:
            self._add("scheduled_review")

    # -- consult -----------------------------------------------------------------

    def maybe_update(
        self,
        raw: dict[str, Any],
        config: dict[str, Any] | None,
        diagnostics: list[str] | None = None,
        refusals: list[str] | None = None,
        *,
        jev: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        sync: bool = False,
    ) -> None:
        """Note triggers and, if one is due, start a background consult. Never blocks."""
        if self.client is None or not isinstance(raw.get("observation"), dict):
            return
        diagnostics, refusals = diagnostics or [], refusals or []
        now = self._clock()
        with self._lock:
            self._detect(raw, diagnostics, refusals, now)
            if self._calls >= self._max_calls and not self._cap_logged:
                self._cap_logged = True
                self._log(
                    f"[strategist] call cap {self._max_calls} reached; the last plan stays in force"
                )
            if (
                not self._pending
                or self._busy
                or self._calls >= self._max_calls
                or now < self._backoff_until
                or now - self._last_start < self._min_gap
            ):
                return
            self._busy = True
            self._last_start = now
            self._calls += 1
            triggers, self._pending = self._pending, []
            previous = self._plan
            history = list(self._history)
            age = now - self._plan_at
            latency = self._latency
        try:
            if self._system is None:
                self._system = build_system(config, (raw.get("map") or {}).get("id"))
            user = build_user(
                raw,
                triggers=triggers,
                diagnostics=diagnostics,
                refusals=refusals,
                plan=previous.as_json(),
                plan_age_s=age,
                history=history,
                last_latency_s=latency,
                jev_advice=jev,
                extra=extra,
            )
        except Exception as exc:  # noqa: BLE001 - never break the loop
            with self._lock:
                self._busy = False
            self._log(f"[strategist] could not build prompt: {exc!r}")
            return
        if sync:
            self._run(user, previous, triggers)
        else:
            threading.Thread(target=self._run, args=(user, previous, triggers), daemon=True).start()

    def _run(self, user: str, previous: Plan, triggers: list[str]) -> None:
        started = self._clock()
        parsed: Plan | None = None
        try:
            text = self.client.generate(  # type: ignore[union-attr]
                user,
                system=self._system,
                temperature=0.5,
                max_output_tokens=2200,
                timeout=MODEL_TIMEOUT_S,
                json_output=True,
            )
            parsed = parse_plan(text, previous)
        except Exception as exc:  # noqa: BLE001 - the strategist must never break the loop
            self._log(f"[strategist] consult failed: {exc!r}")
        done = self._clock()
        with self._lock:
            self._latency = round(done - started, 1)
            if parsed is not None:
                self._plan, self._plan_at = parsed, done
                self._version += 1
                self._errors = 0
                self._pending = [r for r in self._pending if r != "scheduled_review"]
                self._history = [*self._history, parsed.assessment][-HISTORY:]
            else:
                self._errors += 1
                self._backoff_until = done + min(
                    BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** min(self._errors - 1, 4)
                )
                # a failed consult must not lose the reasons it was for
                for t in triggers[:4]:
                    self._add(t)
            self._busy = False
        trace.write(
            "plan",
            ok=parsed is not None,
            triggers=triggers,
            latency_s=self._latency,
            version=self.version,
            plan=parsed.as_json() if parsed else None,
        )
        if parsed is not None:
            self._log(
                f"[strategist] plan v{self.version} ({self._latency}s, {','.join(triggers)[:80]}): "
                f"{parsed.assessment[:200]}"
            )
        else:
            self._log(
                f"[strategist] no usable plan ({self._errors} in a row); keeping the previous one"
            )
