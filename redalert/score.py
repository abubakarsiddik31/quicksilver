"""Score a Red Alert match trace against the build-speed targets.

    python -m redalert.score                      # newest file in traces/
    python -m redalert.score traces/ra-123-45.jsonl [--json]

Reads the ``state`` samples (about every 5 s), the ``plan`` and ``scout`` events
written by :mod:`redalert.trace`, and reports the numbers that decided the match
we lost: how soon the war factory stood, how long vehicle production sat idle,
how long cash sat at zero, whether harvesters, tanks, defences and a scout came
in time, and how much of the LLM's time went on narration.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

VEHICLES = {"1tnk", "2tnk", "3tnk", "4tnk", "apc", "ftrk", "jeep", "harv", "v2rl", "arty", "mgg", "mrj", "ttnk", "mcv"}
TANKS = {"1tnk", "2tnk", "3tnk", "4tnk"}
UNIT_TYPES = VEHICLES | {"e1", "e2", "e3", "e4", "e6", "e7", "dog", "medi", "mech", "spy", "thf", "shok"}
DEFENSES = {"gun", "pbox", "hbox", "ftur", "tsla", "sam", "agun"}

# metric -> (target, True if lower is better)
TARGETS: dict[str, tuple[float, bool]] = {
    "war_factory_s": (110, True),
    "vehicle_idle_s": (10, True),
    "harvesters_at_180s": (2, False),
    "zero_cash_s": (20, True),
    "tanks_at_330s": (6, False),
    "first_defence_s": (240, True),
    "first_scout_s": (100, True),
    "first_attack_s": (360, True),
    "no_army_s": (0, True),
}
NON_COMBAT = {"mcv", "harv", "medi", "mech", "e6", "spy", "thf", "truk"}


def load(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _clock(row: dict[str, Any], t0: float) -> float:
    g = (row.get("time") or {}).get("game_seconds") if isinstance(row.get("time"), dict) else None
    return float(g) if isinstance(g, (int, float)) else row["t"] - t0


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    t0 = rows[0]["t"]
    states = [r for r in rows if r["kind"] == "state"]
    plans = [r for r in rows if r["kind"] == "plan"]
    m: dict[str, Any] = {k: None for k in TARGETS}
    prev = None
    idle = zero = 0.0
    for r in states:
        s = _clock(r, t0)
        u, queue = r.get("units") or {}, r.get("queue") or []
        if m["war_factory_s"] is None and u.get("weap"):
            m["war_factory_s"] = round(s)
        if m["first_defence_s"] is None and any(u.get(d) for d in DEFENSES):
            m["first_defence_s"] = round(s)
        if prev is not None:
            dt = max(0.0, s - prev[0])
            if prev[1].get("weap") and not any(q in VEHICLES for q in prev[2]) and (prev[3] or 0) >= 300:
                idle += dt
            if (prev[3] or 0) < 50:
                zero += dt
        prev = (s, u, queue, r.get("cash"))
    m["vehicle_idle_s"], m["zero_cash_s"] = round(idle), round(zero)
    # Seconds with a war factory but not one combat unit alive (the army was never rebuilt).
    no_army, prev = 0.0, None
    for r in states:
        s = _clock(r, t0)
        u = r.get("units") or {}
        if prev is not None and prev[1]:
            no_army += max(0.0, s - prev[0])
        fighters = sum(v for k, v in u.items() if k in UNIT_TYPES and k not in NON_COMBAT)
        prev = (s, bool(u.get("weap")) and fighters == 0)
    m["no_army_s"] = round(no_army)
    # An offensive attack-move (3+ units) that is not the base-defence reflex.
    defends = [r["t"] for r in rows if r["kind"] == "defend"]
    state_clock = [(r["t"], _clock(r, t0)) for r in states]
    for r in rows:
        if r["kind"] != "orders" or any(abs(r["t"] - d) < 0.5 for d in defends):
            continue
        if any(o.get("cmd") == "attack_move" and (o.get("n") or 0) >= 3 for o in r.get("orders") or []):
            near = min(state_clock, key=lambda x: abs(x[0] - r["t"]), default=(r["t"], r["t"] - t0))
            m["first_attack_s"] = round(near[1] + (r["t"] - near[0]))
            break
    rates = [r["decisions_per_s"] for r in states if isinstance(r.get("decisions_per_s"), (int, float))]
    m["decisions_per_s"] = round(sum(rates) / len(rates), 1) if rates else None

    def at(sec: float) -> dict[str, int]:
        best = min(states, key=lambda r: abs(_clock(r, t0) - sec), default=None)
        return (best or {}).get("units") or {}

    if states:
        m["harvesters_at_180s"] = at(180).get("harv", 0)
        m["tanks_at_330s"] = sum(v for k, v in at(330).items() if k in TANKS)
    scouts = [r for r in rows if r["kind"] == "scout"]
    if scouts:
        m["first_scout_s"] = round(scouts[0].get("game_s") or scouts[0]["t"] - t0)
    lat = [p["latency_s"] for p in plans if isinstance(p.get("latency_s"), (int, float))]
    narration = [p for p in plans if all(t.startswith(("scheduled_review", "harness:", "jev_uncertain")) for t in p.get("triggers", []))]
    m["consults"] = len(plans)
    m["consult_latency_mean_s"] = round(sum(lat) / len(lat), 1) if lat else None
    m["narration_consults"] = len(narration)
    m["duration_s"] = round(_clock(states[-1], t0)) if states else None
    m["final_buildings_units"] = (states[-1].get("units") if states else None)
    return m


def report(m: dict[str, Any]) -> str:
    lines = [f"{'metric':<24}{'value':>10}  target"]
    for k, (target, lower) in TARGETS.items():
        v = m.get(k)
        ok = "?" if v is None else ("ok" if (v <= target if lower else v >= target) else "MISS")
        lines.append(f"{k:<24}{'-' if v is None else v:>10}  {'<=' if lower else '>='} {target:g}   {ok}")
    lines.append("")
    for k in ("decisions_per_s", "consults", "consult_latency_mean_s", "narration_consults", "duration_s"):
        lines.append(f"{k:<24}{m.get(k)!s:>10}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    path = paths[0] if paths else max(glob.glob("traces/ra-*.jsonl"), key=os.path.getmtime, default=None)
    if not path:
        print("no trace found (play a match first; traces are written to traces/)")
        return 1
    m = score(load(path))
    print(json.dumps(m, indent=1) if "--json" in argv else f"{path}\n{report(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
