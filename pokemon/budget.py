"""Spending the battle clock.

The event game runs Showdown's VGC timer: **55 seconds per battle decision and
a 7-minute (420 s) bank for the whole battle**, and an empty bank does not mean
a bad move — it means a **forfeit**. Thinking time is therefore a resource, and
this module is where the policy decides how much of it a decision may spend.

Two numbers matter, both reported by the server in ``observation.clock``:

``decision_seconds_left``
    When the platform takes the decision out of your hands and plays a
    **random** legal choice (50 s in, or 85 s at Team Preview). It already
    folds in the bank: when the bank holds less time than Showdown's timer,
    this *is* the bank, and hitting 0 forfeits.
``bank_seconds_left``
    The whole battle's remaining budget. Spending it all early leaves nothing
    for the turns that matter.

The rule encoded here: never start something that cannot finish before the
deadline, and never let one decision eat the battle. When either is in doubt,
answer from the script. A mediocre move delivered in milliseconds beats a
brilliant one that arrives after the clock has run out — and the platform's
random auto-play is worse than either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pokemon import observation as obs

# Round-trip and overhead margin: never plan to finish at the deadline itself.
SAFETY_SECONDS = 3.0

# The most of the remaining bank any single decision may spend. Turns are
# simultaneous and a battle runs to roughly 10-25 decisions, so ~1/5 is already
# generous for one turn; the point is that a slow call can never drain the
# battle.
MAX_BANK_FRACTION = 0.2


@dataclass(frozen=True)
class Budget:
    """What a decision may spend, in seconds."""

    deadline: float | None
    """Seconds until the platform acts for us (or the bank empties). ``None``
    when the server hasn't reported a clock yet."""

    bank: float | None
    """Seconds left in the whole-battle bank. ``None`` when unreported."""

    @property
    def affordable(self) -> float:
        """How long this decision may take. ``0.0`` means: do not call out to
        anything — answer from the script."""
        if self.deadline is None:
            # No clock yet: be conservative rather than gamble the bank.
            return 0.0
        limit = self.deadline - SAFETY_SECONDS
        if self.bank is not None:
            limit = min(limit, self.bank * MAX_BANK_FRACTION)
        return max(0.0, limit)

    def affords(self, seconds: float) -> bool:
        return seconds <= self.affordable


def budget_of(observation: dict) -> Budget:
    deadline = obs.decision_seconds_left(observation)
    bank = obs.bank_seconds_left(observation)
    return Budget(
        deadline=None if deadline is None else float(deadline),
        bank=None if bank is None else float(bank),
    )


def budget_for(state: Any) -> Budget:
    """The budget for a decision, from the state or a raw observation."""
    if isinstance(state, dict):
        return budget_of(state)
    return budget_of(obs.observation_of(state))


def affords(state: Any, seconds: float) -> bool:
    """Whether there is time to do something that takes ``seconds``.

    The single question the policy asks before reaching for a model: a cheap
    script choice costs nothing and is always affordable; an LLM call is only
    worth starting if it can finish.
    """
    return budget_for(state).affords(seconds)


# A rough ceiling for one model call round trip. The starter's own LLM example
# budgets 40 s per battle decision and 10 s per draft pick, and refuses to make
# a single request that could run over 25 s — these are deliberately inside
# those, leaving room for a retry.
LLM_SECONDS_DECISION = 12.0
LLM_SECONDS_DRAFT = 4.0


def choose_source(state: Any, *, game_phase: str = "moving") -> str:
    """``"llm"`` or ``"script"`` for this decision.

    ``"script"`` is the default and the safe answer; ``"llm"`` is only returned
    when the call comfortably fits both the decision deadline and its share of
    the bank.
    """
    estimate = LLM_SECONDS_DRAFT if game_phase == "draft" else LLM_SECONDS_DECISION
    return "llm" if affords(state, estimate) else "script"
