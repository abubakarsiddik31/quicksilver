"""Compare Red Alert matches across traces: what the winners did that the losers did not.

    python scripts/ra_report.py                 # every traces/ra-*.jsonl
    python scripts/ra_report.py traces/ra-1.jsonl traces/ra-2.jsonl

A match is a win when we destroyed 5+ enemy buildings and lost none, a loss for the reverse; anything else
is listed but left out of the comparison. Each match reports the numbers that separated the two groups in the
traces (income and army by 300 s, harvesters, time at zero cash) and flags the ones below the winners' floor.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

# Floors taken from the won matches (median of 10): below these the economy has already lost the game.
FLOORS = {"inc300": 30.0, "h180": 2, "h300": 3, "army300": 2500, "z300": 0.25}  # z300 is a ceiling
CEILINGS = {"z300"}


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") == "state" and isinstance(row.get("time"), dict) and "income" in row:
                rows.append(row)
    return rows


def metrics(states: list[dict]) -> dict | None:
    if len(states) < 8:
        return None
    units = lambda d: d.get("units") or {}  # noqa: E731
    at = lambda t: min(states, key=lambda d: abs(d["time"]["game_seconds"] - t))  # noqa: E731
    early = [d for d in states if d["time"]["game_seconds"] <= 300] or states
    mil = states[-1].get("military") or {}
    bk, bl = mil.get("buildings_killed") or 0, mil.get("buildings_lost") or 0
    result = "win" if bk >= 5 and bl == 0 else "loss" if bl >= 5 and bk == 0 else "?"
    return {
        "result": result,
        "gs": round(states[-1]["time"]["game_seconds"]),
        "inc300": round(statistics.mean(d["income"] for d in early), 1),
        "z300": round(sum(d["cash"] < 50 for d in early) / len(early), 2),
        "h180": units(at(180)).get("harv", 0),
        "h300": units(at(300)).get("harv", 0),
        "army300": at(300).get("army_value") or 0,
        "weap_s": next((round(d["time"]["game_seconds"]) for d in states if units(d).get("weap")), None),
        "kills/lost": f"{mil.get('units_killed')}/{mil.get('units_lost')}",
    }


def below_floor(m: dict) -> list[str]:
    return [
        k
        for k, floor in FLOORS.items()
        if (m[k] > floor if k in CEILINGS else m[k] < floor)
    ]


def main(argv: list[str]) -> int:
    paths = argv or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "traces", "ra-*.jsonl")), key=os.path.getmtime)
    games = [(os.path.basename(p), metrics(load(p))) for p in paths]
    games = [(n, m) for n, m in games if m]
    if not games:
        print("no Red Alert state traces found")
        return 1
    cols = ["result", "gs", "inc300", "z300", "h180", "h300", "army300", "weap_s", "kills/lost"]
    print(f"{'trace':26}" + "".join(f"{c:>11}" for c in cols) + "  below floor")
    for name, m in games:
        print(f"{name[:25]:26}" + "".join(f"{str(m[c]):>11}" for c in cols) + "  " + ",".join(below_floor(m)))
    print()
    for label in ("win", "loss"):
        group = [m for _, m in games if m["result"] == label]
        if not group:
            continue
        print(f"{label.upper()} (n={len(group)}) median: " + "  ".join(
            f"{k}={statistics.median(m[k] for m in group if m[k] is not None):.1f}"
            for k in ("inc300", "z300", "h180", "h300", "army300", "weap_s")
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
