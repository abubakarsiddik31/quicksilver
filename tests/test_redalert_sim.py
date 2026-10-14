"""The default (model-free) opening, run through a small economy simulator.

Targets come from the lost-match trace: war factory at ~150 s, vehicle production
idle for 62 s, cash at ~0 for minutes, one harvester at 3 minutes, no tanks and no
defence before the attack. These bounds are deliberately loose (the simulator is a
model, not the game) but each one failed badly before the build fixes.
"""

from __future__ import annotations

import os

import pytest

from redalert.agent import RedAlertAgent
from redalert.strategist import Strategist
from redalert.tactics import TacticalAdvisor
from tests.ra_sim import run

os.environ["RA_TRACE"] = "0"


@pytest.mark.parametrize("faction", ["russia", "england"])
def test_default_opening_is_fast_and_never_starves(faction):
    result = run(
        lambda clock: RedAlertAgent(Strategist(None), TacticalAdvisor(None), clock=clock),
        340,
        faction=faction,
    )
    # Barracks come second (a guard against the enemy's opening scout), so the war factory lands ~20 s later than
    # a pure economy opening; the army floor then spends cash as it comes in (cash near zero is the army being built).
    assert result["placed"]["barr"] <= 40, result
    assert 0 < result["war_factory_s"] <= 130, result  # lost match: 150
    assert result["harvesters_at_180s"] >= 2, result  # lost match: 1
    assert result["zero_cash_s"] <= 110, result  # lost match: ~170, with nothing to show for it
    assert result["tanks_at_330s"] >= 2, result  # lost match: 0
    assert 0 < result["first_defence_s"] <= 260, result  # lost match: never
    assert result["placed"]["proc"] <= 80 and result["placed"]["fix"] <= 270, (
        result
    )  # the second (baseline)
    # refinery delays the depot a few seconds; tanks_at_330s is the number that shows the trade was worth it


def test_the_same_run_is_deterministic():
    make = lambda clock: RedAlertAgent(Strategist(None), TacticalAdvisor(None), clock=clock)  # noqa: E731
    assert run(make, 200) == run(make, 200)
