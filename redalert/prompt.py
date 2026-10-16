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


GUIDE_SECTIONS = {
    1,
    4,
    5,
    7,
    8,
    9,
}  # rules, state, orders, scoring, build primer, pitfalls (not check-in/waiting)


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
                ("cost", "cost"),
                ("build_time_s", "build_s"),
                ("power", "power"),
                ("queue", "queue"),
                ("prerequisites", "needs"),
                ("factions", "only"),
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


WAR_DOCTRINE = """\
HOW TO FIGHT THIS WAR (principles, not rules: reason from them and adapt; nothing here is a threshold)

WHAT KIND OF WAR THIS IS. A 20-minute war of attrition against a capable opponent. It ends in one of two ways:
a base is destroyed, or the clock runs out and it is decided on trades (what you killed minus what you lost),
then assets. So the question behind every decision is: "will this action improve the exchange of value, or end
their ability to fight?" Pushing when it will not is a loss that looks like courage; sitting when you are the
stronger side is a loss that looks like caution. You are the GENERAL: you set intent, your officers (the
harness's guard line, strike force, flank packs and scouts) carry it out, and your soldiers fight.

KNOW THE ENEMY BEFORE YOU DECIDE. What you can see is a fraction of what they have. An enemy base you have not
seen recently means their army is bigger than the visible count; estimate it from the time elapsed, their
economy (harvesters, refineries) and how fast their units keep arriving. Scouting is the war's eyes: a scout
should observe from a safe distance and survive, not walk into the base. Ask each review: what is their army
made of, where is it, what are they trying to do (raid my harvesters, bait me, mass for one push, expand)?
Read their intent from behaviour: a lone unit that retreats when you chase is bait; two units hitting your
harvesters are a diversion; a growing mass near the middle is a coming push.

CONCENTRATION AND LOCAL SUPERIORITY. Wars are won by being stronger at the place and moment of the fight, not
by being stronger overall. Do not feed pieces of the army into a bigger one: losses grow faster than the gap in
numbers, so a force that is half as strong does not lose half, it loses nearly everything while killing little.
Fight with the army together, close to your reinforcements and your towers; let the enemy cross the ground and
meet you there. Count a fight before you take it: their units in range plus what can arrive in the next half
minute, against yours. If you cannot win it, do not take it; if you cannot avoid it, take it where you are
strongest. Defence is cheaper than attack: a defender with towers, repair and reinforcements next door trades
better than the same army walking into their half.

COMBINED ARMS: EVERY UNIT HAS A JOB. Read each unit's cost and weapon in the rules table and give it the job it is
good at; nothing is good at everything.
- Tanks are the front: they take the hits and kill vehicles and buildings, and soldiers stand and fire behind them.
- Anti-armour infantry (rocket soldiers) kill tanks cheaply and are weak against infantry: keep them behind the
  line and in numbers when armour is coming.
- Cheap infantry screens, holds ground and is the efficient body against light attackers; it is slaughtered by
  tanks and by anything that kills groups of it, so it must not meet those in the open without cover.
- Fast cheap units (dogs, scouts, light vehicles) have specialist jobs: hunting lone infantry and scouts,
  running down raiders, killing exposed harvesters, forcing the enemy to keep guards at home. They lose to a
  combined force: use them to hunt what is weaker and avoid what is stronger.
- Long-range artillery wins fights from behind a line and dies if it is reached: it needs a screen.
- Mobile anti-air is a specialist: only worth building when there is something in the air.
Composition answers the enemy: when what they field changes, what you build changes, but not on every sighting;
a trend that holds for a minute is information, a single unit is noise.

MANOEUVRE: FLANKS, RAIDS, TRAPS. Hit the enemy where it is weak, not where it is strong.
- A flank is a fast detachment that arrives from the side or behind while the main force holds their attention in
  front; it hits harvesters, refineries, power and production, and leaves before their army turns on it. It needs
  the front to be held and an idea of where their army is.
- Raiding their economy is worth more than trading armies: an enemy with fewer harvesters cannot replace losses.
- The enemy uses the same tools: units that retreat when you chase are leading you into the rest of their force.
  Chase only as far as your own support reaches, then return and take the ground you want.
- Be hard to flank: a thin guard that sees them coming and calls the army is better than the whole army standing
  still or the whole army away. Never leave the base empty while their army is alive and unseen; the strike force
  and the base are one problem.

TEMPO AND INITIATIVE. After you break an enemy wave, their army is dead and their reinforcements are far away:
that is the moment to counter-attack, hard, before they rebuild; hesitating hands the initiative back. After you
lose a fight, do the opposite: pull back to where you are strong, repair, rebuild, and come again bigger. A long
wait for a perfect army only helps the side that is out-producing you.

RETREAT IS A SKILL. Withdraw early, from a position you chose, toward your towers and reinforcements, and keep
firing while you do it: a unit ordered to plain-move does not shoot, so a retreat across open ground with the enemy
in contact is a way of dying without trading. Pull back only the group that is losing; a fight that is going well
is not a retreat. Retreating does not mean giving up the war, it means choosing the next fight.

THE CLOCK AND THE SCORE. In the last minutes the score matters, but never trade it for a collapse: if you are
behind on trades, ask whether a push can kill more than it loses. A push that cannot win its fight only deepens
the deficit and empties the base; fighting at home with your towers and reinforcements usually improves the
exchange. If you are ahead, protect it by winning the fights that come to you, not by hiding. Assets count too:
keep what you have alive.

ECONOMY IS THE ARMY. Income pays for every unit, so protect your harvesters and raid theirs. Spend what you earn
on what the fight needs; money held back is army you do not have, money spent on the wrong things is the same.

ADAPT. If the same approach has failed twice, it is the wrong approach: change something real (composition, where
you fight, how you scout, whether you attack) rather than repeating it with a bigger number. Say in your
assessment what you believe the enemy is doing, what you intend, and what would change your mind.
"""


MEMORY_GUIDE = """\
YOUR MEMORY OF THIS MATCH (`harness_measurements.match_memory`: read it before every plan)
The harness keeps the whole game for you and never resets it. It is the long-run picture a commander carries:
- `clock`: how the match stands on the tiebreak (trade balance now), and how long is left.
- `enemy`: what they have been seen to have for the whole match (buildings, production, tech, defences, harvesters,
  the peak army ever seen and where it was last seen), and a BELIEF about their army right now: what is in view,
  what was seen and is out of view, and an ESTIMATE of the unseen rest worked out from the economy both sides share
  (they started with the same cash; their income follows their harvesters). The longer no enemy army unit has been
  in view, the more that estimate matters and the wider `army_belief_range` is. An army you cannot see is not gone.
- `fights_so_far`: every engagement: what you had, what they had at its peak, what each side lost, the exchange
  (their value destroyed per value you lost), the exchange the fight model EXPECTED beforehand, and how it ended.
  `model_calibration` below 1 means fights have gone worse than the model said: trust a plan less, not more.
- `plans_so_far`: your own earlier plans and what each achieved (exchange gained, army and income change). If a
  plan's result was poor, do not restate it with a bigger number: find out why from the fights and change it.
- `trend`: where economy, your army and their estimated army are heading.
Use it to STRATEGISE, not to react: set the fights you take from what the memory says you can win, rebuild and
re-scout after a fight that went badly, and attack when the picture (not the clock alone) says the exchange will
be in your favour. The harness's fight model does the arithmetic of a single engagement from this same memory
(unit matchups, defences, the unseen force that can reach the fight, how past fights compared with its
predictions). Your officers' `odds` in the reports are its expected exchange.
"""

HARNESS = """\
YOU ARE THE STRATEGIST of a Red Alert bot. A fast harness (5 decisions/second) executes your PLAN; you do not
issue moves one by one, and your latency (seconds) does not stall the base. You are consulted when something
happens (see "triggers"), never to narrate progress: do not describe what the harness is already doing.

WHAT THE HARNESS DOES ON ITS OWN (do not order these)
- deploys the MCV; places every finished building immediately; repairs buildings under 75% hp; queues an
  emergency power plant if the base is actually in low power.
- LOCKED OPENING: power plant -> barracks -> refinery -> war factory, in that order, whatever your list says
  (a plan that put barracks or tech ahead of the war factory lost a match). Before the war factory the barracks
  trains only a 3-rifleman guard. Plan from the war factory onward.
- STRUCTURES: walks your `buildings` list in priority order; an item is built when its owned+queued count is below
  its count and it is in available_production. Structures and Defenses are SEPARATE queues and run in parallel.
  The next structure is queued once the current one is 40% done, so the Building queue never waits for a decision.
- UNITS: walks your `units` list in priority order the same way (counts are TOTAL alive+queued), one new order
  per queue per decision, queues kept shallow. HARVESTERS are automatic: once a war factory exists it keeps
  2 per refinery (max 6) with priority over everything; you only need to list more to want more.
- SPENDING GOVERNOR: structures and units share ONE draw ledger - the total credits-per-second committed by every
  queue must fit the measured income, or cash above a small floor must cover every remaining commitment outright.
  Below that cash floor only harvesters and power commit (income is rebuilt first, always); if cash pins at zero
  the harness cancels the worst non-essential queued item itself so the queues can drain. Income is the real
  limit: look at `harness_measurements.measured_income_per_s`. Several expensive items at once do NOT go faster,
  they all starve (a long stretch at cash 0 stalls the whole build).
- ALIASES you may use as item names: "barracks" (your faction's tent/barr), "tower" (cheapest defensive tower
  you can build; its count is the TOTAL of towers of any kind), "aa" (anti-air site; total of AA sites).
- ARMY FLOOR: from 100 s, combat value must be at least 20 credits per second of game time (4,000 at 300 s).
  Below it, units get the money first and non-essential structures (2nd refinery, radar, tech) wait; power,
  barracks, towers and whatever unlocks a unit you want (e.g. fix for 3tnk) still go ahead.
- KEEP BUSY: once your unit counts are met, an idle barracks / war factory keeps training (riflemen with rocket
  soldiers, more e3 when enemy vehicles outnumber infantry; the war factory your listed vehicles first).
- BATTLE PLAN (doctrine): after the war factory, Jev picks one of expand / mass_army / pressure / defend /
  rebuild / all_in about once a second from both sides' numbers (rules pick when Jev is unsure). It sets the
  army mode, wave size, army_first and adds its units/towers in front of your lists. Keep "doctrine" in
  `tactics.jev_slots` to let it adapt; drop it to pin your own army mode. Switching needs a sure Jev answer or
  two agreeing ones (emergencies skip the wait but not the agreement), and an all_in with no enemy army ever
  seen is refused - no information means assume the enemy is at least as strong as you.
- BASE DEFENCE REFLEX: enemy units within 24 cells of the base (or 10 of a building) pull every fighter onto them
  with attack-move, re-aimed every 1.5 s (a raid of 1-2 is answered by the home line alone and never recalls the strike force: treat a lone intruder as a probe or a diversion). You do not need to order it.
- BASELINE DEFENCE: one tower by 200 s and two by 300 s once a war factory exists, even if you do not list any.
- ANTI-AIR-ONLY units (ftrk, sam, agun) are filtered out of your unit wants until enemy aircraft is seen.
- SCOUT: at ~70 s ONE cheap fast unit is sent toward the enemy (via the midpoint, then their base); a second is
  allowed after 180 s when cash is flush (re-sending lone scouts early fed the enemy five free kills, but
  fighting blind lost matches), so `visible_enemies` / `visible_enemy_buildings` fill in.
- ARMY: sets idle fighters to attack_anything; "hold" gathers idle fighters at `gather` (default your base);
  "attack" strikes ONLY with the whole idle army at once, once `wave_min` are idle (Jev's yes never halves a
  wave, and for ~25 s after each strike the next one assembles at a staging point toward the enemy instead of
  trickling out); "all_in" attack-moves EVERY combat unit repeatedly. Targets: "nearest", "base", "building"
  (nearest visible enemy structure) or [x, y]. The time limit is decided on trades first and assets second (see the doctrine: weigh the exchange, not the score). NEVER order a `move` for units that are in contact with the enemy and winning (a moving unit does not
  shoot; the harness will drop it): keep them on attack-move and let them finish the enemy. Pull back only a group
  that is being outweighed, and then to the nearest tower line, not across the map.
  A lead is also safest when the enemy cannot fight back: when the army you have is several times the enemy army and
  towers you have seen, ATTACK THE BASE and destroy it (that ends the match; do not set wave_min above ~12 then).
  A big army idling behind a large wave_min, while the enemy has almost nothing left, only wins on the tiebreak.
- BATTALIONS (executor-run, on top of the mode): while being raided or holding, the cheapest 2-3 fighters
  form a defence line in front of the base on `defend` stance (they engage in range, never chase); with 6+
  fighters and the opening done, packs of up to 3 run WIDE around the target on alternating sides every
  ~45-70 s and raid from a different direction; the rest is the wave. Jev's `enemy_style` read (mass_infantry
  / mass_armor / mixed / turtle / harass) paces the flank runs and the guard line.
- CHAIN OF COMMAND: you are the general — your plan sets what gets built and how the army fights; Jev is the
  tactician re-deciding doctrine/stage/posture/waves twice a second; the executor is the officer running the
  battalions (report in state as `my_formation`). You are woken OUT OF TURN when the enemy moves: their style
  changes, their army surges by ~1500+ credits, new enemy production structures appear, threat level hits
  "army", or Jev is unsure of the battle plan — use the wake reason to rewrite the plan for THAT enemy.
- `orders`: optional ONE-OFF raw orders (see the order table in the guide) for anything the standing plan cannot
  express (a specific scouting path, selling, rally points, a focused attack on an enemy id). Ids/cells must come
  from the CURRENT state; each is re-checked against the live state and dropped if the id is gone.

MISSIONS: HOW YOU COMMAND THE ARMY (this is your main lever in a fight)
You do not set a single army mode. You assign MISSIONS to OFFICERS; each officer is a named group of soldiers that
keeps its soldiers between plans (restate the same `id` to keep a group together). The harness carries each
mission out at five decisions a second and REPORTS BACK in `harness_measurements.officers` (soldiers, value, where,
state, the enemy value near it, and the odds). You are woken when an officer is broken, withdraws or spots the
enemy: read what happened and change the plan. Always send `missions`; the older `army` block is only a fallback.
  {"id": "main", "kind": "STRIKE", "share": 0.6, "target": "base", "engage_ratio": 1.2, "abort_ratio": 0.7, "regroup_s": 30}
kinds (all fields except id and kind are optional):
- HOLD_LINE  the home guard. Meets anything that comes near the base, otherwise holds a line in front of it on
             defend stance (engages in range, does not chase). `share` or `count`.
- STRIKE     the main body. It musters at a staging point, advances together when the fight model's EXPECTED
             EXCHANGE (their value destroyed per value you lose, counting what is in view, the unseen force from your
             memory that can reach the fight, unit matchups and their defences) is at least `engage_ratio`, fights
             on attack-move, and BREAKS OFF to the tower line when the expected exchange from where it stands, or the
             exchange it is actually getting, falls below `abort_ratio`, then regroups for `regroup_s` seconds and
             comes again. `target`: "base",
             "nearest", "building", "enemy_army", a cell [x, y], or an economic target (below). With no `share` it
             takes whoever is left. Raise `engage_ratio` to demand a safer fight; lower it to accept a riskier one (the model already adds the unseen force).
- FLANK      a detachment that swings wide of the target (`side`: left | right | alternate), hits it from the side
             and withdraws after `duration_s` or when a bigger force meets it. Use it while the main body holds
             their attention in front.
- RAID       a fast group that goes straight for an economic target ("harvesters", "economy", "power",
             "refinery", "production") and withdraws when something bigger arrives. The cheapest, fastest units.
- SCREEN     a small forward picket `leash` cells toward the enemy: it gives vision and gives way when the enemy
             comes, reporting "spotted". It is how you see their army without feeding scouts into their base.
- FALL_BACK  pull a group back to the tower line ("towers") or the base ("base"), fighting as it goes.
`share` is a fraction of all fighters (0-1) or `count` is an exact number. At most 6 missions. The base always
keeps a guard. Typical shapes: a line + a strike + a screen early; add a raid when their harvesters are exposed
and a flank when the front is stuck; after a broken fight, FALL_BACK the broken group and rebuild; after you
break their wave, send the strike at once while their army is dead. Choose what the situation needs: this is
war, not a checklist.

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
answers bounded questions about the live state about twice a second, from the full battle state: which
structure/unit to build next (among your wants that are buildable now), the battle plan, the money's stage
("expand_economy" / "build_army" / "defend_now"), attack/hold posture, whether a wave is worth sending and
where, which unit type should dominate the army ("focus"), the threat level (0-4), and whether another
harvester or power plant is needed. The harness obeys a Jev answer only if it is fresh, its slot is in your
`tactics.jev_slots`, and its confidence reaches your `tactics.min_confidence`; otherwise the whole menu of
options is re-ranked by Jev's per-option probabilities. A committed "all_in" and your reactions are never
overridden. YOU stay the strategist: drop a slot from `jev_slots` to pin it to your plan, raise
min_confidence to trust it less, and write `tactics.hint`, a short brief shown to Jev with the state. You are
woken with "jev_threat:<level>" when Jev sees a serious attack and "jev_uncertain:<slot>" for doubts.

BUDGET REALITY (learned the hard way). A match starts with $5,000 and ~10-15 credits/second per early
harvester. Everything costs: power plant 300, barracks 500, refinery 1400, war factory 2000, depot (fix) 1200,
tesla 1500, a second refinery 1400, heavy tank ~950. Tech and economy bought before an army leave nothing to
defend with; the opponent is another AI that may attack from about 5 minutes with medium tanks + infantry.
Do the arithmetic every time: (cash + income x seconds) against the total cost of what you want, and put
cheap, effective defence (towers, rocket soldiers e3 against tanks) in place BEFORE the push you expect.
Tanks beat infantry; rocket infantry/towers beat tanks; never buy anti-air without seeing aircraft.
ONE BODY: the army fights as one group. Do not issue attack orders for slices of it into the open (a group of 3-10
walking out alone is killed by an enemy that kept its force together, which can throw away a large lead); when ahead
with a clear front, keep producing, regroup at a staging point and move out together, or leave it to the harness.
COST EFFICIENCY: judge every purchase by the army it costs you. Towers cannot move and only help what walks into
their range; cheap units can be everywhere, trade in the open and win the time-limit tiebreak. Keep static defence
to a couple of towers until the army is at its floor, and spend the rest on units, workers and the power they need.
COMPOSITION IS THE FIGHT: pick units that counter what the enemy actually fields (read visible_enemies before every
unit list): anti-armour infantry is poor against infantry, tanks and flame are strong against it, and vehicles are
best answered by anti-armour infantry and towers. Keep tanks in front of soldiers so the soldiers fire from behind them.

HOW TO DECIDE
- Think about economy (income, harvesters), tech path, army composition against what the enemy actually has
  (visible_enemies, enemy_buildings), base defence, the clock (time.remaining_seconds) and the tiebreak. The
  rules table (costs, prerequisites, queues) is authoritative for this match; the guide is background.
- The enemy AI wins by positioning: it keeps training, holds ground behind defences, baits with a lone unit,
  then strikes from the hills with tanks and infantry together. Match that. Never attack a held position at
  parity (the harness will not wave below 1.2x the army plus defences seen); send flank packs on the sides
  while the main force fixes the front (a push that meets enemy units and towers worth as much as itself pulls
  back to the staging point for 45 s: rebuild, then come back bigger); scout the base before committing a wave; keep one doctrine until the
  numbers really change.
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
  "missions": [ {"id": "line", "kind": "HOLD_LINE", "share": 0.25}, {"id": "main", "kind": "STRIKE", "target": "base", "engage_ratio": 1.2}, ... ],   // see MISSIONS
  "army": {"mode": "hold|attack|all_in", "wave_min": 3-30, "target": "nearest|base|building|[x,y]", "gather": [x,y] or null},   // fallback only
  "army_first": false,
  "reactions": [ ... as above ... ],
  "orders": [ {"cmd": "move", "units": [id], "to": [x, y]}, ... ],   // optional, at most 8, one-off
  "tactics": {"hint": "<=300 chars brief for Jev", "min_confidence": 0.5-0.95,
              "jev_slots": ["doctrine","structure","unit","stage","posture","wave","target","focus","threat","harvester","power","style"]},
  "review_in_s": 10-90                                        // when you want to be consulted next if nothing happens
}
"""


def build_system(config: dict[str, Any] | None, map_id: str | None) -> str:
    config = _dict(config)
    parts = [
        "You command one side of a 1v1 Command & Conquer: Red Alert match (OpenRA engine) against another AI agent.",
        WAR_DOCTRINE,
        MEMORY_GUIDE,
        HARNESS,
    ]
    guide = load_guide()
    if guide:
        parts.append(
            "=== PLATFORM GUIDE (authoritative on rules, orders, limits, pitfalls) ===\n" + guide
        )
    schema = _dict(config.get("order_schema"))
    if schema:
        parts.append("=== ORDER SCHEMA ===\n" + json.dumps(schema, separators=(",", ":"))[:8000])
    table = rules_table(config)
    if table:
        parts.append("=== RULES TABLE (this server) ===\n" + table)
    for key in ("results_policy", "match_rules"):
        if config.get(key):
            parts.append(
                f"=== {key.upper()} ===\n" + json.dumps(config[key], separators=(",", ":"))[:3000]
            )
    mp = map_block(config, map_id)
    if mp:
        parts.append("=== " + mp)
    return "\n\n".join(parts)


def _units(obs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for u in (obs.get("units") or [])[:MAX_UNITS]:
        if isinstance(u, dict) and not u.get("wreck"):
            rows.append(
                {k: u.get(k) for k in ("id", "type", "x", "y", "hp", "idle", "stance") if k in u}
            )
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
            {
                k: b.get(k)
                for k in ("id", "type", "x", "y", "hp", "powered", "producing", "repairing")
                if k in b
            }
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
            state[key] = (
                (state[key] or [])[: max(10, len(state[key] or []) // 3)]
                if isinstance(state[key], list)
                else None
            )
        text = json.dumps(state, separators=(",", ":"), default=str)[:MAX_USER_CHARS]
    return "CURRENT STATE (answer with your plan JSON):\n" + text
