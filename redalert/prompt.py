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
from pathlib import Path
from typing import Any

GUIDE_PATH = Path(__file__).resolve().parents[1] / "docs" / "altruagent-skill" / "redalert.md"
GUIDE_CUTOFF = "## 10."  # sections 10+ are MCP call mechanics the harness already handles
MAX_UNITS = 150
MAX_ENEMIES = 80
MAX_USER_CHARS = 60000


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def load_guide() -> str:
    try:
        text = GUIDE_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    cut = text.find(GUIDE_CUTOFF)
    return text[:cut] if cut > 0 else text


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
                ("hp", "hp"), ("speed", "speed"),
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
    rows = layout.get("rows")
    if isinstance(rows, list):
        out.append(
            f"Layout origin={layout.get('origin')} legend={json.dumps(layout.get('legend'))}\n"
            f"ruler: {layout.get('ruler')}\n" + "\n".join(str(r) for r in rows)
        )
    return "\n".join(out)


HARNESS = """\
YOU ARE THE STRATEGIST of a Red Alert bot. A fast harness (5 decisions/second) executes your PLAN; you do not
issue moves one by one, and your latency does not stall the base. You are consulted when something happens
(see "triggers" in each message) and at least every `review_in_s` seconds you choose yourself.

WHAT THE HARNESS DOES ON ITS OWN (you do not need to order these)
- deploys the MCV at the start; places every finished building immediately; repairs buildings under 75% hp;
  queues an emergency power plant if the base is actually in low power.
- builds structures: walks your `buildings` list in order; the first item whose owned+queued count is below its
  count and that is in available_production is built next. Structures and Defenses are SEPARATE queues and
  build in parallel; within one queue only one item builds at a time. Nothing is built before it is in
  available_production, so list what you will want even if prerequisites are missing now.
- trains units: walks your `units` list in order the same way, keeping each unit queue shallow. Unless
  `army_first` is true it holds back the cost of your first two unmet structure wants, so units cannot starve
  the build order. The count is TOTAL alive + queued (e.g. {"item":"3tnk","count":10} keeps 10 heavy tanks).
- army: sets idle fighters to attack_anything; mode "hold" gathers idle fighters at `gather` (default: your
  base); "attack" sends idle fighters as one attack-move wave once `wave_min` are idle; "all_in" attack-moves
  EVERY combat unit repeatedly. Targets: "nearest" (nearest visible enemy, else the enemy base), "base",
  "building" (nearest visible enemy structure) or an explicit [x, y] cell.
- `orders` are optional ONE-OFF raw orders (same format as the order table in the guide) for anything the
  standing plan cannot express: scouting with a specific unit, selling, rally points, guarding, a focused attack
  on a specific enemy id, power_toggle. Ids/cells must come from the CURRENT state below; each is re-checked
  against the live state when sent and dropped if the id has died. They run once per plan you send.

YOUR PLAN REPLACES THE PREVIOUS ONE ENTIRELY. Restate everything you still want. Plans persist until you send
the next one, so a good plan is robust for the next 10-30 seconds: express intent as counts and ordered
priorities, not as a single moment's reaction.

HOW TO DECIDE
- Think about economy (refineries/harvesters), tech path, army composition against what the enemy actually has
  (visible_enemies, enemy_buildings), base defence against what is attacking, the clock (time.remaining_seconds)
  and the tiebreak. Use the rules table: costs, prerequisites and queues there are authoritative for this
  match; the guide text is background. If you are unsure whether an item exists for your faction, check `only`.
- Respect cash: income comes from harvesters. A plan that wants far more than you can afford just stalls.
- Don't feed the enemy: avoid attacking with tiny groups into a stronger army or defences; trading efficiently
  matters at the time limit. Keep units alive when a fight is lost (switch to "hold" and rebuild).
- Look at `harness_diagnostics` and `refusals`: they tell you what is blocked and why. Fix the cause (e.g. a
  missing prerequisite) instead of repeating it.
- Be decisive: every second you spend is 25 ticks.

ANSWER FORMAT: ONE JSON object, nothing else. Write "assessment" FIRST (<= 400 chars: situation, enemy
tendencies, your decision and why) so you reason before you commit, then:
{
  "assessment": "...",
  "buildings": [{"item": "powr", "count": 1}, ...],          // priority order, TOTAL desired; "barracks" = your faction's tent/barr
  "units": [{"item": "e1", "count": 6}, ...],                // priority order, TOTAL desired alive+queued
  "army": {"mode": "hold|attack|all_in", "wave_min": 3-30, "target": "nearest|base|building|[x,y]", "gather": [x,y] or null},
  "army_first": false,
  "orders": [ {"cmd": "move", "units": [id], "to": [x, y]}, ... ],   // optional, at most 8, one-off
  "review_in_s": 4-60                                         // when you want to be consulted next if nothing happens
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
        "events": obs.get("events"),
        "last_orders": raw.get("last_orders"),
        "recent_order_problems": raw.get("recent_order_problems"),
        "harness_diagnostics": diagnostics,
        "refusals": refusals,
        "your_current_plan": plan,
        "plan_age_s": round(plan_age_s, 1),
        "your_last_answer_took_s": last_latency_s,
        "your_previous_assessments": history,
    }
    text = json.dumps(state, separators=(",", ":"), default=str)
    if len(text) > MAX_USER_CHARS:  # shrink the biggest lists before giving up detail
        for key in ("units", "visible_enemies", "events", "last_orders"):
            state[key] = (state[key] or [])[: max(10, len(state[key] or []) // 3)] if isinstance(state[key], list) else None
        text = json.dumps(state, separators=(",", ":"), default=str)[:MAX_USER_CHARS]
    return "CURRENT STATE (answer with your plan JSON):\n" + text
