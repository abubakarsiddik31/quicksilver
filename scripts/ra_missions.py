"""Mission-driven offline battle: a scripted General (fixed missions) against the war simulator's scripted opponent.

    python -m scripts.ra_missions --seeds 6 --scale 1.3 --engage 1.2 --abort 0.7

Exercises the officers and the fight model (no API calls). The simulator is a model, not the game: use it to check
that decisions are sane and that predictions calibrate, not to chase its score.
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import redalert
from redalert.agent import RedAlertAgent
from redalert.officers import HOLD_LINE, SCREEN, STRIKE, Mission
from redalert.plan import Plan
from redalert.strategist import Strategist
from redalert.tactics import TacticalAdvisor
from tests.ra_battle import play


class Scripted(Strategist):
    def __init__(self, engage, abort):
        super().__init__(None)
        self._plan = Plan(
            missions=(
                Mission("line", HOLD_LINE, share=0.25),
                Mission("screen", SCREEN, count=1),
                Mission("main", STRIKE, target="base", engage_ratio=engage, abort_ratio=abort),
            )
        )

    def current(self):
        return self._plan

    @property
    def enabled(self):
        return True

    def maybe_update(self, *a, **k):
        return None

    def notify(self, reason):
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--seconds", type=float, default=900.0)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--engage", type=float, default=1.2)
ap.add_argument("--abort", type=float, default=0.7)
a = ap.parse_args()
print("agent from", redalert.__file__)
rows = []
for seed in range(1, a.seeds + 1):
    r = play(
        lambda clock: RedAlertAgent(
            Scripted(a.engage, a.abort), TacticalAdvisor(None), clock=clock
        ),
        a.seconds,
        "russia",
        seed,
        a.scale,
    )
    rows.append(r)
    print(
        f"seed{seed}: {r['outcome']:7s} t={r['t']:4d}s kills={r['kills']:6d} lost={r['deaths']:6d} units {r['units_killed']}/{r['units_lost']} bldg_lost={r['buildings_lost']} foe_bld_killed={r['enemy_buildings_killed']} waves={r['foe_waves']}"
    )
print(
    "SUMMARY won=%d lost=%d trade=%+d kills=%d lost=%d"
    % (
        sum(r["outcome"] == "won" for r in rows),
        sum(r["outcome"] == "lost" for r in rows),
        sum(r["kills"] for r in rows) - sum(r["deaths"] for r in rows),
        sum(r["kills"] for r in rows),
        sum(r["deaths"] for r in rows),
    )
)
