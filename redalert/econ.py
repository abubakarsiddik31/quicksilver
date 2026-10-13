"""Measured income: the number the plans never had.

``economy.cash`` + ``economy.ore`` only show money after spending, so income is recovered as
(change in cash + what the queues drew in the same window) / seconds, using each
queue entry's ``progress`` times its cost. Smoothed with an EMA so a single
refinery drop-off doesn't whipsaw it. Used by the LLM and Jev prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

WINDOW_S = 1.0


def funds(econ: Any) -> float:
    """What we can spend: ``cash`` plus refined ``ore`` not yet turned into cash.

    Both pay for production. Reading ``cash`` alone showed $0 for minutes while the ore store filled up;
    once it is full (``resource_capacity``), harvesters cannot unload and income stops.
    """
    if not isinstance(econ, dict):
        return 0.0
    total = 0.0
    for key in ("cash", "ore"):
        v = econ.get(key)
        if isinstance(v, (int, float)) and v > 0:
            total += float(v)
    return total
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
        if not isinstance(econ.get("cash"), (int, float)):
            return self.income
        cash = funds(econ)
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
