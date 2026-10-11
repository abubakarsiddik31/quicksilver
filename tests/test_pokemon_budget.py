"""Tests for the battle-clock budget guard.

This is the code that enforces "latency is a resource": an LLM call may only
start when it can finish before the platform plays a random move for us, and
without draining the whole-battle bank.
"""

from __future__ import annotations

from types import SimpleNamespace

from pokemon import budget


def state(decision=None, bank=None, *, no_clock=False):
    observation = {} if no_clock else {"clock": {}}
    if decision is not None:
        observation["clock"]["decision_seconds_left"] = decision
    if bank is not None:
        observation["clock"]["bank_seconds_left"] = bank
    return SimpleNamespace(raw={"observation": observation}, observation="", phase="moving")


def test_affordable_is_capped_by_the_decision_deadline():
    """A large bank must not license a long call when this decision's own
    deadline is near — the platform plays a *random* move at 0."""
    b = budget.budget_for(state(decision=20, bank=400))
    assert b.affordable == 20 - budget.SAFETY_SECONDS


def test_affordable_is_capped_by_the_banks_share():
    """Early in a battle the bank is the binding constraint, so one slow
    decision cannot spend the battle."""
    b = budget.budget_for(state(decision=50, bank=60))
    assert b.affordable == 60 * budget.MAX_BANK_FRACTION
    assert b.affordable < 50 - budget.SAFETY_SECONDS


def test_the_deadline_already_folds_in_the_bank():
    """The platform reports the bank as the deadline when the bank is shorter,
    so a nearly-empty bank shows up as a nearly-zero deadline. The bank-share
    cap then binds harder than the safety margin, which is the point: a
    nearly-empty bank must not be spent on one decision."""
    b = budget.budget_for(state(decision=4, bank=4))
    assert b.deadline == 4
    assert b.affordable == min(4 - budget.SAFETY_SECONDS, 4 * budget.MAX_BANK_FRACTION)
    assert b.affordable == 4 * budget.MAX_BANK_FRACTION


def test_a_missing_clock_is_conservative():
    """The clock is absent until Showdown reports it. Gambling the bank on an
    unknown deadline is not a trade worth making."""
    b = budget.budget_for(state(no_clock=True))
    assert b.deadline is None
    assert b.affordable == 0.0
    assert b.affords(0.1) is False
    assert budget.choose_source(state(no_clock=True)) == "script"


def test_affordable_never_goes_negative():
    for decision, bank in ((0, 0), (1, 10), (2, None)):
        assert budget.budget_for(state(decision=decision, bank=bank)).affordable >= 0.0


def test_affords_boundary():
    b = budget.budget_for(state(decision=20, bank=400))
    assert b.affords(b.affordable) is True
    assert b.affords(b.affordable + 0.001) is False


def test_choose_source_prefers_the_script_when_time_is_short():
    assert budget.choose_source(state(decision=6, bank=400)) == "script"
    assert budget.choose_source(state(decision=50, bank=400)) == "llm"


def test_a_draft_pick_gets_a_tighter_estimate_than_a_battle_turn():
    """15 s per pick, and the 15 s includes the model call."""
    tight = state(decision=8, bank=None)
    assert budget.choose_source(tight, game_phase="draft") == "llm"
    # 8 s is enough for a 4 s draft call but not a 12 s battle call.
    assert budget.choose_source(tight, game_phase="moving") == "script"


def test_a_full_bank_alone_never_licenses_an_unbounded_call():
    """Even with 7 minutes banked, one decision may not spend a large slice."""
    b = budget.budget_for(state(decision=50, bank=420))
    assert b.affordable <= 420 * budget.MAX_BANK_FRACTION
    assert b.affordable < 50


def test_budget_accepts_a_raw_observation_dict():
    assert budget.affords({"clock": {"decision_seconds_left": 50, "bank_seconds_left": 400}}, 10)
    assert not budget.affords({"clock": {"decision_seconds_left": 5, "bank_seconds_left": 400}}, 10)
