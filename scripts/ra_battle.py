"""Play the Red Alert agent against the scripted opponent in the local war simulator.

    python -m scripts.ra_battle                 # this checkout, three seeds, two factions
    python -m scripts.ra_battle --seeds 5 --scale 1.3

Run it from another checkout (e.g. a worktree of an older commit) to compare two agent versions on the
same scenarios; ``redalert`` is imported from the current directory.
"""

from __future__ import annotations

import argparse
import statistics

import redalert
from redalert.agent import RedAlertAgent
from redalert.strategist import Strategist
from redalert.tactics import TacticalAdvisor
from tests.ra_battle import play


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="enemy income multiplier")
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print(f"agent from {redalert.__file__}")
    rows = []
    for faction in ("russia",):
        for seed in range(1, args.seeds + 1):
            r = play(lambda clock: RedAlertAgent(Strategist(None), TacticalAdvisor(None), clock=clock),
                     args.seconds, faction, seed, args.scale)
            rows.append(r)
            print(f"{faction} seed{seed}: {r['outcome']:7s} t={r['t']:4d}s kills={r['kills']:6d} lost={r['deaths']:6d} "
                  f"units {r['units_killed']}/{r['units_lost']} bldg_lost={r['buildings_lost']} "
                  f"enemy_bldg_killed={r['enemy_buildings_killed']} waves={r['foe_waves']}")
            if args.verbose:
                for row in r["timeline"]:
                    print("   ", row)
    wins = sum(r["outcome"] == "won" for r in rows)
    lost = sum(r["outcome"] == "lost" for r in rows)
    print(f"SUMMARY won={wins} lost={lost} timeout={len(rows) - wins - lost} "
          f"avg_survival={statistics.mean(r['t'] for r in rows):.0f}s "
          f"trade={sum(r['kills'] for r in rows) - sum(r['deaths'] for r in rows):+d}")


if __name__ == "__main__":
    main()
