"""Measured income: the number the plans never had.

``economy.cash`` only shows cash after spending, so income is recovered as
(change in cash + what the queues drew in the same window) / seconds, using each
queue entry's ``progress`` times its cost. Smoothed with an EMA so a single
refinery drop-off doesn't whipsaw it. Used by the LLM and Jev prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

WINDOW_S = 1.0
ALPHA = 0.25


class EconTracker:
    def __init__(self) -> None:
        self.income = 0.0  # credits per second, smoothed
        self._cash: float | None = None
        self._t: float | None = None
        self._spend = 0.0
        self._prog: dict[tuple[str, int], float] = {}

    def update(self, obs: dict[str, Any], cost_of: Callable[[str], int], now: float) -> float:
        econ = obs.get("economy") or {}
        cash = econ.get("cash")
        if not isinstance(cash, (int, float)):
            return self.income
        seen: dict[str, int] = {}
        prog: dict[tuple[str, int], float] = {}
        for p in obs.get("production") or []:
            if not isinstance(p, dict):
                continue
            item = str(p.get("item")).lower()
            nth = seen.get(item, 0)
            seen[item] = nth + 1
            value = p.get("progress")
            if not isinstance(value, (int, float)):
                continue
            key = (item, nth)
            prog[key] = float(value)
            before = self._prog.get(key, 0.0)
            if value > before:  # a new entry that is already underway counts from zero
                self._spend += (float(value) - before) * cost_of(item)
        self._prog = prog
        if self._cash is None or self._t is None:
            self._cash, self._t, self._spend = float(cash), now, 0.0
            return self.income
        dt = now - self._t
        if dt >= WINDOW_S:
            gained = max(0.0, (float(cash) - self._cash + self._spend) / dt)
            self.income = gained if self.income == 0.0 else ALPHA * gained + (1 - ALPHA) * self.income
            self._cash, self._t, self._spend = float(cash), now, 0.0
        return self.income
