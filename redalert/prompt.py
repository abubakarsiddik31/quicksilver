"""What the strategist model is told: everything the platform documents, plus the live state.

The system prompt (built once per match) carries the standing context: the
platform's own Red Alert guide, this match's rules table (cost, build time,
power, prerequisites, queue for every item), the map, the harness contract and
the plan schema. The user message is the live state, the harness's own
diagnostics and what it has already tried. Opponent-controlled text never
appears in either: names are not sent, only engine type strings and numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GUIDE_PATH = Path(__file__).resolve().parents[1] / "docs" / "altruagent-skill" / "redalert.md"
GUIDE_CUTOFF = "## 10."  # sections 10+ are MCP call mechanics the harness already handles
MAX_UNITS = 100
MAX_ENEMIES = 60
MAX_USER_CHARS = 60000


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


GUIDE_SECTIONS = {1, 4, 5, 7, 8, 9}  # rules, state, orders, scoring, build primer, pitfalls (not check-in/waiting)


def load_guide() -> str:
    """The platform guide minus the sections about connecting and waiting, which the harness handles."""
    try:
        text = GUIDE_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    cut = text.find(GUIDE_CUTOFF)
    text = text[:cut] if cut > 0 else text
    parts = re.split(r"(?m)^(?=## \d+\. )", text)
    keep = [parts[0]]
    for part in parts[1:]:
        m = re.match(r"## (\d+)\. ", part)
        if m and int(m.group(1)) in GUIDE_SECTIONS:
            keep.append(part)
    return "".join(keep)


def rules_table(config: dict[str, Any] | None) -> str:
    rules = _dict(_dict(config).get("rules"))
    lines: list[str] = []
    for group in ("buildings", "units"):
        for name, info in sorted(_dict(rules.get(group)).items()):
            info = _dict(info)
            bits = [f"{name} ({group[:-1]})"]
            for key, label in (
                ("cost", "cost"), ("build_time_s", "build_s"), ("power", "power"),
                ("queue", "queue"), ("prerequisites", "needs"), ("factions", "only"),
            ):
                if info.get(key) not in (None, [], ""):
                    bits.append(f"{label}={json.dumps(info[key], separators=(',', ':'))}")
            weapon = info.get("weapon")
            if weapon:
                bits.append(f"weapon={json.dumps(weapon, separators=(',', ':'))[:120]}")
            lines.append(" ".join(bits))
    return "\n".join(lines)


def map_block(config: dict[str, Any] | None, map_id: str | None) -> str:
    entry = _dict(_dict(_dict(_dict(config).get("rules")).get("maps")).get(map_id or ""))
    if not entry:
        return ""
    layout = _dict(entry.get("layout"))
    out = [f"MAP {map_id}: bounds={entry.get('bounds')} spawns={entry.get('spawns')}"]
    if layout.get("legend"):
        out.append("(the ore/terrain picture is omitted for speed: scout to learn the ground)")
    return "\n".join(out)


HARNESS = """\
YOU ARE THE STRATEGIST of a Red Alert bot. A fast harness (5 decisions/second) executes your PLAN; you do not
issue moves one by one, and your latency (seconds) does not stall the base. You are consulted when something
happens (see "triggers"), never to narrate progress: do not describe what the harness is already doing.

WHAT THE HARNESS DOES ON ITS OWN (do not order these)
- deploys the MCV; places every finished building immediately; repairs buildings under 75% hp; queues an
  emergency power plant if the base is actually in low power.
- STRUCTURES: walks your `buildings` list in priority order; an item is built when its owned+queued count is below
  its count and it is in available_production. Structures and Defenses are SEPARATE queues and run in parallel.
  The next structure is queued once the current one is 40% done, so the Building queue never waits for a decision.
- UNITS: walks your `units` list in priority order the same way (counts are TOTAL alive+queued), one new order
  per queue per decision, queues kept shallow. HARVESTERS are automatic: once a war factory exists it keeps
  2 per refinery (max 6) with priority over everything; you only need to list more to want more.
- SPENDING GOVERNOR: only one purchase costing 800+ is in production at a time unless cash covers the next one
  outright, and units hold back the price of your first unmet structure (unless `army_first`). Income is the
  real limit: look at `harness_measurements.measured_income_per_s`. Several expensive items at once do NOT go
  faster, they all starve (one match was lost with cash at 0 for two minutes).
- ALIASES you may use as item names: "barracks" (your faction's tent/barr), "tower" (cheapest defensive tower
  you can build; its count is the TOTAL of towers of any kind), "aa" (anti-air site; total of AA sites).
- BASELINE DEFENCE: one tower by 200 s and two by 300 s once a war factory exists, even if you do not list any.
- ANTI-AIR-ONLY units (ftrk, sam, agun) are filtered out of your unit wants until enemy aircraft is seen.
- SCOUT: from ~70 s one cheap fast unit is kept sent toward the enemy (via the midpoint, then their base), so
  `visible_enemies` / `visible_enemy_buildings` fill in. Re-read them; you were blind until the attack landed
  in the match that was lost.
- ARMY: sets idle fighters to attack_anything; "hold" gathers idle fighters at `gather` (default your base);
  "attack" sends idle fighters as one attack-move wave once `wave_min` are idle; "all_in" attack-moves EVERY
  combat unit repeatedly. Targets: "nearest", "base", "building" (nearest visible enemy structure) or [x, y].
- `orders`: optional ONE-OFF raw orders (see the order table in the guide) for anything the standing plan cannot
  express (a specific scouting path, selling, rally points, a focused attack on an enemy id). Ids/cells must come
  from the CURRENT state; each is re-checked against the live state and dropped if the id is gone.

REACTIONS (your instant reflexes: the harness evaluates them every decision, in milliseconds, while you think)
  "reactions": [{"name": "...", "when": {...}, "do": [...]}]   // at most 8; `when` keys are ANDed
  when: {"enemy_near_base": {"at_least": 2, "radius": 30, "types": []}}, {"under_attack": true},
        {"power_low": true}, {"enemy_air": true}, {"cash_below": N}, {"cash_above": N}, {"time_over_s": N},
        {"army_at_least": N}, {"enemy_building_seen": ["tsla", ...]}
        (cash_above / cash_below are automatically held until 150 s so they cannot jump the opening)
  do:   {"want_building": {"item": "tower", "count": 3}}, {"want_unit": {"item": "e3", "count": 8}},
        {"army": "hold" | "retreat" | "attack"}, {"army_first": true}
  Their wants go to the FRONT of your lists while the condition holds. Use them for everything you would
  otherwise only be able to do after a 5-second consult: answering an armour push, going anti-air, switching to
  an army when cash floats. Restate reactions you still want: the default set (enemy at the gates / enemy
  aircraft / cash floating) is replaced by yours.

YOUR PLAN REPLACES THE PREVIOUS ONE ENTIRELY. Restate everything you still want. Plans persist until you send the
next one, so a good plan is robust for the next 20-60 seconds: express intent as counts, ordered priorities and
reactions, not as a single moment's reaction.

JEV, YOUR FAST TACTICAL PARTNER (if "jev_latest_advice" is present). Jev is a separate decision model that
answers bounded questions about the live state about once a second: which structure/unit to build next (among
your wants that are buildable now), attack/hold/retreat, whether a wave is worth sending, the threat level
(0-4), whether another harvester or power plant is needed, and the stage ("expand_economy" / "build_army" /
"defend_now"). The harness obeys a Jev answer only if it is fresh, its slot is in your `tactics.jev_slots`, and
its confidence reaches your `tactics.min_confidence`; a committed "all_in" and your reactions are never
overridden. YOU stay the strategist: drop a slot from `jev_slots` to pin it to your plan, raise min_confidence
to trust it less, and write `tactics.hint`, a short brief shown to Jev with the state. You are woken with
"jev_threat:<level>" when Jev sees a serious attack and "jev_uncertain:<slot>" for posture/stage doubts.

BUDGET REALITY (learned the hard way). A match starts with $5,000 and ~10-15 credits/second per early
harvester. Everything costs: power plant 300, barracks 500, refinery 1400, war factory 2000, depot (fix) 1200,
tesla 1500, a second refinery 1400, heavy tank ~950. Tech and economy bought before an army leave nothing to
defend with; the opponent is another AI that may attack from about 5 minutes with medium tanks + infantry.
Do the arithmetic every time: (cash + income x seconds) against the total cost of what you want, and put
cheap, effective defence (towers, rocket soldiers e3 against tanks) in place BEFORE the push you expect.
Tanks beat infantry; rocket infantry/towers beat tanks; never buy anti-air without seeing aircraft.

HOW TO DECIDE
- Think about economy (income, harvesters), tech path, army composition against what the enemy actually has
  (visible_enemies, enemy_buildings), base defence, the clock (time.remaining_seconds) and the tiebreak. The
  rules table (costs, prerequisites, queues) is authoritative for this match; the guide is background.
- Don't feed the enemy: avoid small waves into a stronger army or defences; trading efficiently matters at the
  time limit. Keep units alive when a fight is lost (switch to "hold" and rebuild).
- Look at `harness_diagnostics` and `refusals`: they say what is blocked and why. Fix the cause instead of
  repeating it. Be decisive: every second you spend is 25 ticks.

ANSWER FORMAT: ONE JSON object, nothing else. Write "assessment" FIRST (<= 400 chars: situation, enemy
tendencies, your decision and why, with the money arithmetic) so you reason before you commit, then:
{
  "assessment": "...",
  "buildings": [{"item": "powr", "count": 1}, ...],          // priority order, TOTAL desired
  "units": [{"item": "e1", "count": 6}, ...],                // priority order, TOTAL desired alive+queued
  "army": {"mode": "hold|attack|all_in", "wave_min": 3-30, "target": "nearest|base|building|[x,y]", "gather": [x,y] or null},
  "army_first": false,
  "reactions": [ ... as above ... ],
  "orders": [ {"cmd": "move", "units": [id], "to": [x, y]}, ... ],   // optional, at most 8, one-off
  "tactics": {"hint": "<=300 chars brief for Jev", "min_confidence": 0.5-0.95,
              "jev_slots": ["structure","unit","posture","wave","target","threat","harvester","power","stage"]},
  "review_in_s": 10-90                                        // when you want to be consulted next if nothing happens
}
"""


def build_system(config: dict[str, Any] | None, map_id: str | None) -> str:
    config = _dict(config)
    parts = [
        "You command one side of a 1v1 Command & Conquer: Red Alert match (OpenRA engine) against another AI agent.",
        HARNESS,
    ]
    guide = load_guide()
    if guide:
        parts.append("=== PLATFORM GUIDE (authoritative on rules, orders, limits, pitfalls) ===\n" + guide)
    schema = _dict(config.get("order_schema"))
    if schema:
        parts.append("=== ORDER SCHEMA ===\n" + json.dumps(schema, separators=(",", ":"))[:8000])
    table = rules_table(config)
    if table:
        parts.append("=== RULES TABLE (this server) ===\n" + table)
    for key in ("results_policy", "match_rules"):
        if config.get(key):
            parts.append(f"=== {key.upper()} ===\n" + json.dumps(config[key], separators=(",", ":"))[:3000])
    mp = map_block(config, map_id)
    if mp:
        parts.append("=== " + mp)
    return "\n\n".join(parts)


def _units(obs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for u in (obs.get("units") or [])[:MAX_UNITS]:
        if isinstance(u, dict) and not u.get("wreck"):
            rows.append({k: u.get(k) for k in ("id", "type", "x", "y", "hp", "idle", "stance") if k in u})
    return rows


def _enemies(items: Any) -> list[dict[str, Any]]:
    return [
        {k: e.get(k) for k in ("id", "type", "x", "y", "hp") if k in e}
        for e in (items or [])[:MAX_ENEMIES]
        if isinstance(e, dict) and not e.get("wreck")
    ]


def build_user(
    raw: dict[str, Any],
    *,
    triggers: list[str],
    diagnostics: list[str],
    refusals: list[str],
    plan: dict[str, Any],
    plan_age_s: float,
    history: list[str],
    last_latency_s: float | None,
    jev_advice: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    obs = _dict(raw.get("observation"))
    legal = _dict(raw.get("legal_actions"))
    state = {
        "triggers": triggers,
        "tick": obs.get("tick"),
        "time": raw.get("time"),
        "match_rules": raw.get("match_rules"),
        "map": raw.get("map"),
        "faction": obs.get("faction"),
        "enemy_faction": obs.get("enemy_faction"),
        "base_center": obs.get("base_center"),
        "explored_percent": obs.get("explored_percent"),
        "economy": obs.get("economy"),
        "military": obs.get("military"),
        "buildings": [
            {k: b.get(k) for k in ("id", "type", "x", "y", "hp", "powered", "producing", "repairing") if k in b}
            for b in (obs.get("buildings") or [])
            if isinstance(b, dict)
        ],
        "units": _units(obs),
        "production": obs.get("production"),
        "available_production": obs.get("available_production"),
        "visible_enemies": _enemies(obs.get("visible_enemies")),
        "visible_enemy_buildings": _enemies(obs.get("visible_enemy_buildings")),
        "enemy_base_guess": legal.get("enemy_base"),
        "events": (obs.get("events") or [])[-12:],
        "last_orders": raw.get("last_orders"),
        "recent_order_problems": raw.get("recent_order_problems"),
        "harness_diagnostics": diagnostics,
        "refusals": refusals,
        "your_current_plan": plan,
        "plan_age_s": round(plan_age_s, 1),
        "your_last_answer_took_s": last_latency_s,
        "your_previous_assessments": history,
        "jev_latest_advice": jev_advice,
        "harness_measurements": extra,
    }
    text = json.dumps(state, separators=(",", ":"), default=str)
    if len(text) > MAX_USER_CHARS:  # shrink the biggest lists before giving up detail
        for key in ("units", "visible_enemies", "events", "last_orders"):
            state[key] = (state[key] or [])[: max(10, len(state[key] or []) // 3)] if isinstance(state[key], list) else None
        text = json.dumps(state, separators=(",", ":"), default=str)[:MAX_USER_CHARS]
    return "CURRENT STATE (answer with your plan JSON):\n" + text
