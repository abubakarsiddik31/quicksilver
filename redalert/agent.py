"""Real-time Red Alert player: an LLM strategist, a Jev tactician and a fast executor.

Red Alert never waits, so the parts run at different speeds:

* ``redalert.strategist.Strategist`` consults an LLM in a background thread,
  woken by game events, and publishes a *plan* (``redalert.plan.Plan``): ordered
  structure and unit wants, army posture, if-then *reactions*, optional one-off
  raw orders. Every strategic decision comes from the model; with no model the
  built-in default opening runs instead.
* ``redalert.tactics.TacticalAdvisor`` (TypeSafe Jev) answers bounded tactical
  questions about the live state about once a second: next structure or unit,
  attack/hold/retreat, wave timing, target, threat, economy-vs-army stage. Its
  answers are used only when fresh, allowed by the plan and confident enough.
* ``RedAlertAgent`` (here) runs ~5 times a second and executes all of that
  against the latest state in milliseconds. It owns the reflexes that must not
  wait for a model: deploy the MCV, place finished buildings at once, repair, an
  emergency power plant, automatic harvesters, scouting, a spending governor
  (one big purchase at a time, so queues don't starve each other) and
  pre-queueing the next structure so the Building queue never waits on us.

The executor never invents strategy and the models never touch the wire: raw
orders and Jev picks are re-validated against the live state before they are sent.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, WithReasoning
from common.jev import JevClient
from common.llm import GeminiClient
from redalert import trace
from redalert import doctrine as doctrines
from redalert.econ import EconTracker, funds
from redalert.intel import Intel
from redalert.plan import DEFAULT_PLAN, Plan, Reaction, sanitize_order
from redalert.strategist import Strategist
from redalert.tactics import ADVICE_TTL_S, ESCALATE_LEVEL, Candidates, TacticalAdvisor

CORE_STRUCTURES = {"fact", "powr", "apwr", "proc", "weap", "tent", "barr"}
NON_COMBAT = {"mcv", "harv", "medi", "mech", "e6", "spy", "thf", "truk"}
DEFENSES = {"gun", "pbox", "hbox", "ftur", "tsla", "sam", "agun"}
STRUCTURES = {
    "powr", "apwr", "tent", "barr", "proc", "weap", "fix", "dome", "atek", "stek",
    "silo", "hpad", "afld", "syrd", "spen", "kenn", "fcom", "iron", "pdox", "mslo",
}
INFANTRY = {"e1", "e2", "e3", "e4", "e6", "e7", "dog", "medi", "mech", "spy", "thf", "shok"}
SHIPS = {"ss", "dd", "ca", "pt", "lst", "msub"}
AIRCRAFT = {"mig", "yak", "hind", "heli", "tran"}
ENEMY_AIR = {"mig", "yak", "hind", "heli", "badr", "u2", "mh60", "tran"}
AA_ONLY = {"ftrk", "sam", "agun"}  # useless without enemy aircraft: filtered out until some is seen
ALLIED_FACTIONS = {"allies", "allied", "england", "france", "germany"}
SOVIET_FACTIONS = {"soviet", "soviets", "russia", "ukraine"}
TOWER_ORDER = ("pbox", "ftur", "gun", "tsla")  # cheapest first
AA_ORDER = ("agun", "sam")
SCOUT_PREF = ("dog", "jeep", "e1")
FALLBACK_SECS = {"powr": 14, "barr": 20, "proc": 42, "weap": 45, "fix": 34, "harv": 26, "e1": 3, "e3": 5, "3tnk": 30}
FALLBACK_COST = {
    "powr": 300, "tent": 500, "barr": 500, "proc": 1400, "weap": 2000, "harv": 1100, "e1": 100, "e3": 300,
    "fix": 1200, "tsla": 1500, "pbox": 400, "ftur": 600, "gun": 600, "3tnk": 950, "2tnk": 800, "1tnk": 700,
    "apc": 850, "ftrk": 600, "jeep": 500, "dog": 200,
}

TTL = {"build": 2.5, "train": 1.5, "place": 1.0, "burst": 3.0}
REJECT_TTL = {"not_available": 5.0, "not_ready": 2.0, "invalid": 5.0}
DIAG_AFTER_S = 12.0
REFUSAL_KEEP_S = 60.0
MAX_BATCH = 20
YES = 0.65  # a Jev yes/no needs at least this to act on
WAVE_YES = 0.55
MAX_HARV, MAX_POWR, MAX_DEFENSES = 6, 8, 4
DEFENSE_ORDER = ("tsla", "gun", "pbox", "ftur")
TARGET_NAMES = {"nearest", "building", "base"}
# Spending governor. Every queue draws cost/build-time credits per second while it works, so several queues
# running at once starve each other (a harvester alone burns ~42/s against ~20/s income per harvester). A new
# purchase starts only if the burn of what is already in flight stays within the budget, or cash covers it outright.
MIN_BUDGET = 35.0  # credits/s allowed before income is known (a refinery burns ~33/s)
BUDGET_FACTOR = 1.5  # of measured income: queues may run slightly ahead of income, cash cushions the rest
CASH_MARGIN = 300
PRE_QUEUE_PROGRESS = 0.4  # queue the next structure once the current one is this far along
SCOUT_AT_S = 70
SCOUT_MAX = 1  # one scout per match: re-sending lone units fed the enemy five free kills
# Locked until the war factory is up; no plan may reorder it. Barracks second: the enemy's opening scout
# walked through our empty base in live matches, and a few riflemen kill it (and anything that follows).
OPENING = (("powr", 1), ("barracks", 1), ("proc", 1), ("weap", 1))
EARLY_GUARD = 3  # infantry trained before the war factory; the rest of the money is the war factory's
BARRACKS = ("barr", "tent")
# Army floor: after this many seconds, combat value (credits) must keep pace with the clock or units go first
# and non-essential structures (2nd refinery, radar, tech) wait.
ARMY_FLOOR_FROM_S, ARMY_FLOOR_PER_S = 100.0, 20.0
DEFEND_RADIUS = 24  # enemy units this close to the base centre (or 10 to any building) trigger the defence reflex
DEFEND_EVERY_S = 1.5
POWER_MARGIN = 40  # after the opening, keep this much spare power: low power slows every queue
ORE_FULL = 0.75  # ore store this full: spend it now (a full store stops the harvesters unloading)
NOT_A_THREAT = {"harv", "mcv", "truk"}  # enemy units that never pull the army home
CRITICAL = {"fact", "proc", "weap"}  # always worth repairing; the rest only with money to spare
EXPECTED = {  # documented observation fields we rely on; a missing one is reported once (the "ore" lesson)
    "observation": ("tick", "economy", "military", "units", "buildings", "production", "available_production",
                    "visible_enemies", "visible_enemy_buildings", "base_center", "events"),
    "economy": ("cash", "ore", "power_provided", "power_drained", "resource_capacity", "harvesters"),
    "legal_actions": ("deploy", "combat_units", "idle_units", "harvesters", "enemy_base"),
}
ATTACK_VALUE = 5000  # an army worth this much (credits) attacks even if the plan or Jev says hold
FORWARD_RADIUS = 36  # idle fighters farther than this from home are already in the field: keep them fighting
FILLER_VEHICLES = ("3tnk", "2tnk", "1tnk", "4tnk", "v2rl", "arty", "apc", "jeep")
OPENING_S = 150  # until then (or a war factory) defences wait unless there is an emergency
TOWER_BY_S = ((200, 1), (300, 2))  # baseline defence timeline: (game seconds, towers)
ATTACK_MEMORY_S = 8.0
ALIAS_ITEMS = {"barracks", "tower", "aa"}


def _cell(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
        return int(value[0]), int(value[1])
    if isinstance(value, dict) and isinstance(value.get("x"), (int, float)) and isinstance(value.get("y"), (int, float)):
        return int(value["x"]), int(value["y"])
    return None


class RedAlertAgent:
    """Executes the strategist's plan at real-time speed."""

    def __init__(
        self,
        strategist: Strategist | None = None,
        advisor: TacticalAdvisor | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self.advisor = advisor if advisor is not None else TacticalAdvisor(
            JevClient.from_env(), telemetry_path=os.environ.get("RA_TELEMETRY") or None
        )
        if strategist is None:
            # Opt-in: in live matches the LLM kept the army on "hold" for ten minutes at ~6.5 s a consult, while the
            # built-in playbook attacks on its own. RA_LLM=1 turns the Gemini strategist back on.
            on = os.environ.get("RA_LLM", "0").strip().lower() in ("1", "true", "yes", "on")
            strategist = Strategist(GeminiClient.from_env() if on else None)
        self.strategist = strategist
        if not self.strategist.enabled:
            print("[redalert] playing the built-in playbook (Gemini off; RA_LLM=1 turns it on)", file=sys.stderr, flush=True)
        self.econ = EconTracker()
        self._until: dict[str, float] = {}
        self._stanced: set[int] = set()
        self._gathered: set[int] = set()
        self.intel = Intel()
        self.doctrines = doctrines.DoctrineState()
        self._battle: dict[str, Any] = {}
        self._scouts: set[int] = set()
        self._scouts_sent = 0
        self._exec_version = -1
        self._diag_since: dict[str, float] = {}
        self._refusals: list[tuple[float, str]] = []
        self._last_sent: list[dict[str, Any]] = []
        self._rules_for: int | None = None
        self._cost: dict[str, int] = {}
        self._queue: dict[str, str] = {}
        self._needs: dict[str, list[str]] = {}
        self._secs: dict[str, float] = {}
        self._errors_logged: set[str] = set()
        self._state_at = 0.0
        self._decisions = 0
        self._schema_checked = False  # choose_action calls since the last state trace: our real decision rate
        self._attack_seq = -1
        self._attack_at = -1e9
        self._fired: set[str] = set()

    # -- throttling and learning -------------------------------------------------

    def _fresh(self, key: str, ttl: float) -> bool:
        """True (and arms the cooldown) unless ``key`` was sent or refused too recently."""
        now = self._clock()
        if now < self._until.get(key, 0.0):
            return False
        self._until[key] = now + ttl
        return True

    def _cooling(self, key: str) -> bool:
        """True while ``key`` is inside its resend cooldown (does not arm it)."""
        return self._clock() < self._until.get(key, 0.0)

    def _learn(self, config: dict[str, Any] | None) -> None:
        if not isinstance(config, dict) or self._rules_for == id(config):
            return
        self._rules_for = id(config)
        rules = config.get("rules") if isinstance(config.get("rules"), dict) else {}
        for group in ("buildings", "units"):
            for name, info in (rules.get(group) or {}).items():
                if not isinstance(info, dict):
                    continue
                key = str(name).lower()
                if isinstance(info.get("cost"), (int, float)):
                    self._cost[key] = int(info["cost"])
                if isinstance(info.get("queue"), str):
                    self._queue[key] = info["queue"].lower()
                if isinstance(info.get("build_time_s"), (int, float)) and info["build_time_s"] > 0:
                    self._secs[key] = float(info["build_time_s"])
                if isinstance(info.get("prerequisites"), list):
                    self._needs[key] = [str(p).lower() for p in info["prerequisites"]]

    def queue_of(self, item: str) -> str:
        if item in self._queue:
            return self._queue[item]
        if item in DEFENSES:
            return "defense"
        if item in STRUCTURES:
            return "building"
        if item in INFANTRY:
            return "infantry"
        if item in SHIPS:
            return "ship"
        if item in AIRCRAFT:
            return "aircraft"
        return "vehicle"

    def cost_of(self, item: str) -> int:
        return self._cost.get(item, FALLBACK_COST.get(item, 100))

    def rate_of(self, item: str) -> float:
        """Credits per second an item draws while it builds."""
        cost = self.cost_of(item)
        secs = self._secs.get(item) or FALLBACK_SECS.get(item) or max(3.0, cost / 30.0)
        return cost / max(1.0, secs)

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """Learn which orders were refused: back off them and tell the strategist why."""
        sent = self._last_sent
        now = self._clock()
        if result.get("error") == "INVALID_ACTION":
            self._refusals.append((now, str(result.get("detail") or "batch refused")[:200]))
            return
        if "error" in result:
            return
        for row in result.get("results") or []:
            if not isinstance(row, dict) or row.get("status") != "rejected":
                continue
            index = row.get("index")
            if not isinstance(index, int) or not 0 <= index < len(sent):
                continue
            order = sent[index]
            reason = str(row.get("reason") or "rejected")
            item = order.get("item")
            if item and order.get("cmd") in ("build", "train", "place"):
                key = f"{order['cmd']}:{str(item).lower()}"
                self._until[key] = max(self._until.get(key, 0.0), now + REJECT_TTL.get(reason, 3.0))
            self._refusals.append(
                (now, f"{order.get('cmd')} {item or ''}: {reason} {row.get('detail') or ''}".strip()[:200])
            )
            trace.write("refused", order=order, reason=reason, detail=row.get("detail"))
        self._refusals = [(t, m) for t, m in self._refusals if now - t < REFUSAL_KEEP_S][-8:]

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _barracks(item: str, faction: str, avail: set[str], owned: dict[str, int]) -> str:
        """The plan may say "barracks": resolve it to this faction's tent/barr."""
        if item != "barracks":
            return item
        order = ("barr", "tent") if faction in SOVIET_FACTIONS else ("tent", "barr")
        for name in order:
            if name in avail or owned.get(name):
                return name
        return order[0]

    def _resolve(self, item: str, faction: str, avail: set[str], owned: dict[str, int]) -> str:
        """Aliases -> this faction's real item: barracks, tower (cheapest defence), aa."""
        if item == "barracks":
            return self._barracks(item, faction, avail, owned)
        group = TOWER_ORDER if item == "tower" else AA_ORDER if item == "aa" else None
        if group is None:
            return item
        for name in group:
            if name in avail:
                return name
        soviet = faction in SOVIET_FACTIONS
        return ("ftur" if soviet else "pbox") if item == "tower" else ("sam" if soviet else "agun")

    def _group_want(self, item: str, count: int, faction: str, avail: set[str], owned: dict[str, int],
                    queued: dict[str, int]) -> tuple[str, int] | None:
        """A want for an alias: ``count`` is the TOTAL across the group (towers of any kind)."""
        res = self._resolve(item, faction, avail, owned)
        group = TOWER_ORDER if item == "tower" else AA_ORDER if item == "aa" else (res,)
        total = sum(owned.get(g, 0) + queued.get(g, 0) for g in group)
        if total >= count:
            return None
        return res, owned.get(res, 0) + queued.get(res, 0) + (count - total)

    # -- reactions ---------------------------------------------------------------

    def _holds(self, key: str, arg: Any, c: dict[str, Any]) -> bool:
        if key == "enemy_near_base":
            home, radius = c["home"], arg["radius"]
            types = set(arg["types"])
            n = sum(
                1 for e in c["enemies"]
                if home and (cell := _cell(e)) and math.dist(cell, home) <= radius
                and (not types or str(e.get("type")).lower() in types)
            )
            return n >= arg["at_least"]
        if key == "under_attack":
            return self._clock() - self._attack_at <= ATTACK_MEMORY_S
        if key == "power_low":
            return c["drained"] > c["provided"]
        if key == "enemy_air":
            return c["air_seen"]
        if key == "cash_below":
            return c["cash"] < arg
        if key == "cash_above":
            return c["cash"] > arg
        if key == "time_over_s":
            return c["game_s"] >= arg
        if key == "army_at_least":
            return c["army"] >= arg
        if key == "enemy_building_seen":
            return bool(c["enemy_building_types"] & set(arg))
        return False

    def _react(self, plan: Plan, c: dict[str, Any]) -> list[Reaction]:
        fired = [r for r in plan.reactions if all(self._holds(k, v, c) for k, v in r.when.items())]
        names = {r.name for r in fired}
        for name in names - self._fired:
            trace.write("reaction", name=name, tick=c.get("tick"))
        self._fired = names
        return fired

    # -- the fast loop -----------------------------------------------------------

    def _safe(self, name: str, fn: Callable[[], Any], default: Any) -> Any:
        """Run an optional subsystem; a bug in it must never stop the core build loop."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            key = f"{name}:{type(exc).__name__}:{exc}"[:120]
            if key not in self._errors_logged:
                self._errors_logged.add(key)
                print(f"[redalert] {name} failed, continuing without it: {key}", file=sys.stderr, flush=True)
            return default

    def choose_action(self, state: GameState, context: DecisionContext):
        """Never raises: an exception here would forfeit the match, so a bad state just waits."""
        try:
            return self._decide(state, context)
        except Exception as exc:  # noqa: BLE001
            key = f"{type(exc).__name__}:{exc}"[:120]
            if key not in self._errors_logged:
                self._errors_logged.add(key)
                print(f"[redalert] decision error, waiting: {key}", file=sys.stderr, flush=True)
            return WAIT

    def _decide(self, state: GameState, context: DecisionContext):
        raw = getattr(state, "raw", None)
        obs = raw.get("observation") if isinstance(raw, dict) else None
        if not isinstance(obs, dict):
            return WAIT
        config = getattr(context, "game_config", None)
        self._learn(config)
        mono = self._clock()

        plan = self.strategist.current() if self.strategist.enabled else DEFAULT_PLAN
        legal = raw.get("legal_actions") or {}
        production = [p for p in (obs.get("production") or []) if isinstance(p, dict)]
        avail = {str(x).lower() for x in (obs.get("available_production") or [])}
        econ = obs.get("economy") or {}
        cash = funds(econ)  # cash + refined ore: both are spendable
        if not self._schema_checked:
            self._schema_checked = True
            self._safe("schema", lambda: self._check_schema(obs, econ, legal), None)
        provided, drained = econ.get("power_provided", 0) or 0, econ.get("power_drained", 0) or 0
        faction = str(obs.get("faction") or "").lower()
        units = [u for u in (obs.get("units") or []) if isinstance(u, dict) and not u.get("wreck")]
        buildings = [b for b in (obs.get("buildings") or []) if isinstance(b, dict)]
        enemies = [e for e in (obs.get("visible_enemies") or []) if isinstance(e, dict) and not e.get("wreck")]
        enemy_buildings = [e for e in (obs.get("visible_enemy_buildings") or []) if isinstance(e, dict)]
        home = _cell(obs.get("base_center"))
        game_s = (raw.get("time") or {}).get("game_seconds")
        game_s = float(game_s) if isinstance(game_s, (int, float)) else float(obs.get("tick") or 0) / 25.0
        income = self._safe("economy", lambda: self.econ.update(obs, self.cost_of, mono), 0.0)

        owned: dict[str, int] = {}
        for row in (*units, *buildings):
            t = str(row.get("type")).lower()
            owned[t] = owned.get(t, 0) + 1
        queued: dict[str, int] = {}
        entries: dict[str, list[dict[str, Any]]] = {}
        burn_struct = burn_units = 0.0
        for p in production:
            item = str(p.get("item")).lower()
            q = str(p.get("queue") or self.queue_of(item)).lower()
            queued[item] = queued.get(item, 0) + 1
            progress = p.get("progress")
            entries.setdefault(q, []).append({
                "item": item, "ready": bool(p.get("ready")),
                "progress": float(progress) if isinstance(progress, (int, float)) else 0.0,
            })
            if not p.get("ready"):
                if q in ("building", "defense"):
                    burn_struct += self.rate_of(item)
                else:
                    burn_units += self.rate_of(item)
        budget = max(MIN_BUDGET, income * BUDGET_FACTOR)
        # Cash already promised to what is under way (cost still to be drawn): only the rest is really free.
        committed_s = committed_u = 0.0
        for q_name, es in entries.items():
            for e in es:
                if not e["ready"]:
                    owed = self.cost_of(e["item"]) * (1.0 - e["progress"])
                    if q_name in ("building", "defense"):
                        committed_s += owed
                    else:
                        committed_u += owed

        # Attack memory for the "under_attack" reaction condition.
        for e in obs.get("events") or []:
            seq = e.get("seq") if isinstance(e, dict) else None
            if isinstance(seq, int) and seq > self._attack_seq:
                self._attack_seq = seq
                if str(e.get("kind")) == "under_attack":
                    self._attack_at = mono

        idle_combat = [u for u in (legal.get("combat_units") or []) if isinstance(u, int) and u not in self._scouts]
        all_combat = [
            u for u in units if isinstance(u.get("id"), int) and str(u.get("type")).lower() not in NON_COMBAT
            and u["id"] not in self._scouts
        ]
        air_seen = any(str(e.get("type")).lower() in ENEMY_AIR for e in enemies)

        # The battle picture (both sides, with memory through the fog) and the battle plan in force.
        self._safe("intel", lambda: self.intel.update(obs, mono), None)
        refineries = owned.get("proc", 0)
        army_value = sum(self.cost_of(str(u.get("type")).lower()) for u in all_combat)
        enemy_seen = self._safe("intel", lambda: self.intel.enemy_army_value(self.cost_of), 0.0)
        near_n, near_v = self._safe("intel", lambda: self.intel.enemy_near(home, self.cost_of, mono), (0, 0.0))
        towers_up = sum(n for t, n in owned.items() if t in DEFENSES or t in TOWER_ORDER)
        army_floor = max(0.0, (game_s - ARMY_FLOOR_FROM_S) * ARMY_FLOOR_PER_S)
        military = obs.get("military") or {}
        numbers = doctrines.battle_state(
            game_s=game_s, remaining_s=(raw.get("time") or {}).get("remaining_seconds"), my_army=army_value,
            enemy_army_seen=enemy_seen, enemy_near_base=near_n, enemy_army_near_base=near_v,
            my_losses=military.get("deaths_cost") or 0, my_kills=military.get("kills_cost") or 0, income=income,
            harvesters=owned.get("harv", 0), refineries=refineries, towers=towers_up, army_floor=army_floor,
        )
        if owned.get("weap") and "doctrine" in plan.jev_slots:  # the opening is locked; a plan may pin its army
            plan = self._safe("doctrine", lambda: self._choose_doctrine(plan, numbers, mono), plan)
        doc = self.doctrines.doctrine
        army_floor *= doc.floor_scale

        # Reactions: the model's if-then rules, evaluated in microseconds.
        fired = self._safe("reactions", lambda: self._react(plan, {
            "home": home, "enemies": enemies, "drained": drained, "provided": provided, "air_seen": air_seen,
            "cash": cash, "game_s": game_s, "army": len(all_combat), "tick": obs.get("tick"),
            "enemy_building_types": {str(e.get("type")).lower() for e in enemy_buildings},
        }), [])
        react_wants: list[tuple[str, int]] = []
        react_units: list[tuple[str, int]] = []
        forced_mode: str | None = None
        army_first = plan.army_first
        for r in fired:
            for action in r.do:
                if "want_building" in action:
                    react_wants.append((action["want_building"]["item"], action["want_building"]["count"]))
                elif "want_unit" in action:
                    react_units.append((action["want_unit"]["item"], action["want_unit"]["count"]))
                elif "army" in action and forced_mode is None:
                    forced_mode = action["army"]
                elif action.get("army_first"):
                    army_first = True

        orders: list[dict[str, Any]] = []
        diag: list[str] = []

        # Reflex 1: deploy the MCV. Reflex 2: place finished structures at once.
        deploy = legal.get("deploy") or []
        if deploy:
            orders.append({"cmd": "deploy", "units": [deploy[0]]})
        placed: set[str] = set()
        for p in production:
            item = str(p.get("item")).lower()
            if (
                p.get("ready") and item not in placed
                and self.queue_of(item) in ("building", "defense")
                and self._fresh(f"place:{item}", TTL["place"])
            ):
                orders.append({"cmd": "place", "item": p.get("item")})
                placed.add(item)

        orders += self._raw_orders(plan, raw, obs, units, buildings, avail)

        # Wants: the plan's, with reactions and baseline rules in front.
        wants: list[tuple[str, int]] = []
        for i, c in plan.buildings:
            if i in ("tower", "aa"):
                w = self._group_want(i, c, faction, avail, owned, queued)
                if w:
                    wants.append(w)
            else:
                wants.append((self._resolve(i, faction, avail, owned), c))
        unit_wants = [(i, c) for i, c in plan.units if i not in AA_ONLY or air_seen]
        for item, count in reversed(react_wants):
            w = self._group_want(item, count, faction, avail, owned, queued) if item in ("tower", "aa") \
                else (self._resolve(item, faction, avail, owned), count)
            if w:
                wants.insert(0, w)
        for item, count in reversed(react_units):
            if item not in AA_ONLY or air_seen:
                unit_wants.insert(0, (item, count))

        # Baseline defence timeline and automatic harvesters (the model may ask for more).
        for at_s, n in reversed(TOWER_BY_S):
            if game_s >= at_s:
                w = self._group_want("tower", n, faction, avail, owned, queued)
                if w and owned.get("weap"):  # the Defense queue is separate: this never delays structures
                    wants.append(w)
                break
        if doc.extra_towers and (owned.get("weap") or near_n):
            w = self._group_want("tower", min(MAX_DEFENSES, towers_up + doc.extra_towers), faction, avail, owned, queued)
            if w:
                wants.insert(0, w)
        harv_goal = min(MAX_HARV, doc.harv_per_proc * refineries) if owned.get("weap") else 0
        army_short = bool(owned.get("weap")) and army_value < army_floor
        if army_short and owned.get("harv", 0) >= refineries:
            # One per refinery keeps income alive; rebuying 1,100 harvesters with no army left lost a match.
            harv_goal = min(harv_goal, refineries)
        harvs = owned.get("harv", 0) + queued.get("harv", 0)
        if harv_goal and harvs < harv_goal and "harv" in avail:
            unit_wants.insert(0, ("harv", harv_goal))

        def short(item: str, count: int) -> bool:
            return owned.get(item, 0) + queued.get(item, 0) < count

        def slot_free(q: str) -> bool:
            """Can another structure be queued? Yes once the current one is well along (or finished)."""
            es = entries.get(q, [])
            return not es or (len(es) == 1 and (es[0]["ready"] or es[0]["progress"] >= PRE_QUEUE_PROGRESS))

        def unit_slot_free(item: str) -> bool:
            q = self.queue_of(item)
            return len(entries.get(q, [])) < (3 if q == "infantry" else 2)

        struct_cands = tuple(dict.fromkeys(
            i for i, c in wants if short(i, c) and i in avail and slot_free(self.queue_of(i))
        ))
        unit_cands = tuple(dict.fromkeys(
            i for i, c in unit_wants
            if short(i, c) and i in avail and self.cost_of(i) <= cash and unit_slot_free(i)
        ))

        advice = None
        if self.advisor.enabled:
            self._safe("jev", lambda: self.advisor.maybe_refresh(raw, plan, Candidates(
                structures=struct_cands, units=unit_cands, idle_combat=len(idle_combat),
                all_combat=len(all_combat), has_harvester_item="harv" in avail, has_power_item="powr" in avail,
                describe={i: f"cost {self.cost_of(i)}" for i in (*struct_cands, *unit_cands)},
                income=income, scout_alive=bool(self._scouts),
                battle=self._battle_summary(obs, mono, home, legal, idle_combat, numbers),
                doctrine=self.doctrines.name, doctrine_for_s=mono - self.doctrines.since,
            )), None)
            advice = self.advisor.current()
        now_adv = self.advisor.now()

        def adv(slot: str) -> Any:
            return advice.get(slot, plan, now=now_adv) if advice else None

        pick = adv("structure")
        if pick in struct_cands:
            wants.sort(key=lambda w: w[0] != pick)  # stable: Jev's pick first, the rest keep plan order
        pick = adv("unit")
        if pick in unit_cands:
            unit_wants.sort(key=lambda w: w[0] != pick)
        need = adv("harvester")
        harvs = owned.get("harv", 0) + queued.get("harv", 0)
        if isinstance(need, float) and need >= YES and "harv" in avail and harvs < MAX_HARV and not army_short:
            unit_wants.insert(0, ("harv", harvs + 1))
        need = adv("power")
        powrs = owned.get("powr", 0) + queued.get("powr", 0)
        if isinstance(need, float) and need >= YES and "powr" in avail and powrs < MAX_POWR:
            wants.insert(0, ("powr", powrs + 1))
        threat = adv("threat")
        stage = adv("stage")
        defend = (isinstance(threat, int) and threat >= ESCALATE_LEVEL) or stage == "defend_now"
        if defend:
            tower = next((d for d in DEFENSE_ORDER if d in avail), None)
            have_d = sum(owned.get(d, 0) + queued.get(d, 0) for d in DEFENSE_ORDER)
            if tower and have_d < MAX_DEFENSES:
                wants.insert(0, (tower, owned.get(tower, 0) + queued.get(tower, 0) + 1))
        if stage in ("build_army", "defend_now"):
            army_first = True
        if not owned.get("weap"):
            # The opening is fixed: power, refinery, war factory. A plan that puts barracks or tech first
            # cost 40-60 s of war factory (and so of harvesters) in live matches.
            opening = [(self._resolve(i, faction, avail, owned), c) for i, c in OPENING]
            rest = [w for w in wants if w[0] not in {i for i, _ in opening} | set(BARRACKS)]
            wants = [*opening, *rest]
        if drained > provided and "powr" in avail:  # emergency power: really in low power
            wants.insert(0, ("powr", owned.get("powr", 0) + queued.get("powr", 0) + 1))
        elif owned.get("weap") and provided - drained < POWER_MARGIN and "powr" in avail and not queued.get("powr") \
                and owned.get("powr", 0) < MAX_POWR:
            # Before the next building tips us into low power (40 s of it in one match), not after.
            wants.insert(0, ("powr", owned.get("powr", 0) + 1))
        ore, capacity = econ.get("ore"), econ.get("resource_capacity")
        ore_full = isinstance(ore, (int, float)) and isinstance(capacity, (int, float)) and capacity > 0 \
            and ore >= ORE_FULL * capacity
        if ore_full and "silo" in avail and owned.get("silo", 0) + queued.get("silo", 0) < 3:
            wants.append(("silo", owned.get("silo", 0) + queued.get("silo", 0) + 1))

        # Army floor (computed above): once the war factory is up, an army that lags the clock gets the money first.
        essential: set[str] = {"powr", "weap", *BARRACKS}
        if not owned.get("proc"):
            essential.add("proc")
        unlocks: set[str] = set()
        for item, count in unit_wants:  # what unlocks a wanted unit is essential too (e.g. fix for heavy tanks)
            if item not in avail:
                unlocks.update(n for n in self._needs.get(item) or [] if not owned.get(n))
        essential |= unlocks
        if army_short:
            army_first = True

        # Whoever has priority (structures, or units when army_first) sees the other side's commitments as free.
        free_s = max(0.0, cash - committed_s - (committed_u if army_first else 0.0))
        free_u = max(0.0, cash - committed_u - (0.0 if army_first else committed_s))
        spent_s = 0.0  # structures ordered this decision

        # Structures: priority order; one per queue, pre-queued, within the spending budget.
        held: set[str] = set()  # queues where a higher-priority want is still waiting on a prerequisite
        cooling: set[str] = set()  # queues whose top want is inside its resend cooldown: nothing may slip past it
        emergency = bool(fired) or defend or drained > provided
        for item, count in wants:
            if not short(item, count):
                continue
            q = self.queue_of(item)
            if item not in avail:
                needs = self._needs.get(item) or []
                if not (needs and all(owned.get(n) or queued.get(n) for n in needs) and any(queued.get(n) for n in needs)):
                    diag.append(f"wanted {item} x{count} is not in available_production (prerequisites?)")
                held.add(q)
                continue
            cost = self.cost_of(item)
            if not slot_free(q):
                continue
            if q in held and entries.get(q):
                continue  # no queue-jumping: the higher-priority item needs this queue to clear first
            if q in cooling:
                continue  # a higher-priority want of this queue is waiting out its resend cooldown
            if q == "defense" and not owned.get("weap") and game_s < OPENING_S and not emergency:
                continue  # towers must not compete with the opening for the war factory's 2,000
            if army_short and item not in essential and q != "defense":
                continue  # 2nd refinery, radar, tech: after the army has caught up with the clock
            rate = self.rate_of(item)
            # Structures have priority over units unless army_first; what unlocks a wanted unit always does.
            units_first = army_first and item not in unlocks
            fs = free_s if units_first else max(0.0, cash - committed_s - spent_s)
            if burn_struct + (burn_units if units_first else 0.0) + rate > budget and fs < cost + CASH_MARGIN:
                continue
            if self._cooling(f"build:{item}"):
                cooling.add(q)
                continue
            if self._fresh(f"build:{item}", TTL["build"]):
                orders.append({"cmd": "build", "item": item})
                entries.setdefault(q, []).append({"item": item, "ready": False, "progress": 0.0})
                burn_struct += rate
                spent_s += cost
                queued[item] = queued.get(item, 0) + 1
                free_s = max(0.0, free_s - cost)
                if not army_first:
                    free_u = max(0.0, free_u - cost)

        # Units: priority order, one new order per queue per decision. Harvesters come first and are never held back.
        spendable = free_u
        saving_for = 0.0  # price of the top-priority unit we are waiting to afford: cheaper wants must not eat it
        used: set[str] = set()
        trained = False
        for item, count in unit_wants:
            have = owned.get(item, 0) + queued.get(item, 0)
            if have >= count:
                continue
            if item not in avail:
                needs = self._needs.get(item) or []
                if not (needs and all(owned.get(n) or queued.get(n) for n in needs) and any(queued.get(n) for n in needs)):
                    diag.append(f"wanted unit {item} x{count} is not in available_production")
                continue
            q = self.queue_of(item)
            if q in used or not unit_slot_free(item):
                continue
            if q == "infantry" and not owned.get("weap") and not fired:
                continue  # before the war factory the barracks only trains the guard (below)
            unit_cost = max(1, self.cost_of(item))
            n = min(count - have, 3, max(1, int(spendable // unit_cost))) if q == "infantry" else 1
            cost = unit_cost * n
            rate = self.rate_of(item)
            if item != "harv":  # harvesters are the economy: never held back, though they count toward the burn
                if burn_units + (0.0 if army_first else burn_struct) + rate > budget and spendable < cost + CASH_MARGIN:
                    continue
                if spendable - saving_for < cost:
                    if not saving_for and spendable < cost <= 2 * (spendable + budget * 8):
                        saving_for = cost  # within reach (~8 s of income): save for it instead of buying cheaper things
                        used.add(q)
                    continue
            if self._cooling(f"train:{item}"):
                used.add(q)  # hold the queue: a lower-priority unit must not take the slot this tick
                continue
            if not self._fresh(f"train:{item}", TTL["train"]):
                continue
            orders.append({"cmd": "train", "item": item, "count": n})
            entries.setdefault(q, []).append({"item": item, "ready": False, "progress": 0.0})
            used.add(q)
            if item != "harv":
                spendable -= cost
                if army_first:
                    free_s = max(0.0, free_s - cost)
            burn_units += rate
            trained = True

        # Keep busy: a barracks or war factory with nothing queued is wasted. Once the plan's counts are met,
        # keep training (the opponent that beat us never let its barracks idle).
        waiting = any(short(i, c) and i in avail and i in essential and not queued.get(i) for i, c in wants)
        if not saving_for and not waiting:  # an unfunded war factory / depot / barracks gets the money first
            enemy_vehicles = sum(1 for e in enemies if self.queue_of(str(e.get("type")).lower()) == "vehicle")
            enemy_infantry = sum(1 for e in enemies if self.queue_of(str(e.get("type")).lower()) == "infantry")
            for q in ("infantry", "vehicle"):
                if q in used or entries.get(q):
                    continue
                guard_left = EARLY_GUARD - sum(owned.get(i, 0) + queued.get(i, 0) for i in INFANTRY)
                if not owned.get("weap") and guard_left <= 0:
                    continue  # a guard, not an army: the war factory comes first
                item = self._filler(q, avail, owned, plan, air_seen, enemy_vehicles > enemy_infantry)
                if item is None or self._cooling(f"train:{item}"):
                    continue
                unit_cost = max(1, self.cost_of(item))
                # Floating cash (or an army behind the clock) is what this spends; the plan's wants came first.
                margin = 0 if (army_short or ore_full) else CASH_MARGIN
                if spendable < unit_cost + margin or not self._fresh(f"train:{item}", TTL["train"]):
                    continue
                n = min(3, max(1, int(spendable // unit_cost))) if q == "infantry" else 1
                if not owned.get("weap"):
                    n = min(n, guard_left)
                orders.append({"cmd": "train", "item": item, "count": n})
                entries.setdefault(q, []).append({"item": item, "ready": False, "progress": 0.0})
                spendable -= unit_cost * n
                burn_units += self.rate_of(item)
                trained = True

        orders += self._safe("scout", lambda: self._scout(units, legal, avail, entries, game_s, home, raw), [])

        unmet_structure = any(short(i, c) for i, c in wants)
        if not unmet_structure and not trained and not production and cash > 1500:
            diag.append(f"{cash} cash unspent and nothing is queued or wanted")
        if not diag and not unmet_structure and not production and all(
            owned.get(i, 0) + queued.get(i, 0) >= c for i, c in (*plan.buildings, *plan.units) if i not in ALIAS_ITEMS
        ):
            diag.append("every structure and unit want in the plan is satisfied")

        # Reflex 3: repairs. Repairing costs money, so only the core buildings when cash is short.
        for b in buildings:
            if (
                (b.get("hp", 100) or 100) < 75 and b.get("id") is not None and not b.get("repairing")
                and (str(b.get("type")).lower() in CRITICAL or cash >= 600)
                and self._fresh(f"repair:{b['id']}", 2.0)
            ):
                orders.append({"cmd": "repair", "building": b["id"], "on": True})

        # Reflex 4: idle harvesters go back to work (a depleted patch or a retreat leaves them standing).
        idle_ids = set(legal.get("idle_units") or [])
        lazy = [
            u["id"] for u in units if str(u.get("type")).lower() == "harv" and isinstance(u.get("id"), int)
            and (u.get("idle") is True or u["id"] in idle_ids)
        ]
        lazy = [h for h in lazy if self._fresh(f"harvest:{h}", 8.0)]
        if lazy:
            orders.append({"cmd": "harvest", "units": lazy[:50]})

        idle_combat = [u for u in idle_combat if u not in self._scouts]  # a scout picked this tick is not a fighter
        guard = None if plan.army_mode == "all_in" else self._safe(
            "defend", lambda: self._defend(home, buildings, enemies, all_combat), None
        )
        if guard is None:
            orders += self._army(plan, raw, obs, legal, units, adv, defend, forced_mode, idle_combat, army_value,
                                 enemy_seen)
        else:
            orders += guard

        now = self._clock()
        self._diag_since = {d: self._diag_since.get(d, now) for d in diag}
        stuck = [d for d, t in self._diag_since.items() if now - t >= DIAG_AFTER_S]
        self._refusals = [(t, m) for t, m in self._refusals if now - t < REFUSAL_KEEP_S]
        if self.strategist.enabled:
            for reason in self.advisor.pop_escalations():
                self.strategist.notify(reason)
            jev = None
            if advice is not None:
                jev = {"age_s": round(now_adv - advice.at, 1), "values": advice.values, "confidence": advice.confidence}
            self.strategist.maybe_update(
                raw, config, stuck, [m for _, m in self._refusals], jev=jev,
                extra={
                    "game_seconds": round(game_s), "measured_income_per_s": round(income, 1),
                    "harvesters": owned.get("harv", 0), "refineries": refineries,
                    "queue_eta_s": [
                        {"item": p.get("item"), "remaining_s": p.get("remaining_s")} for p in production[:8]
                    ],
                    "reactions_fired": [r.name for r in fired], "scout_alive": bool(self._scouts),
                    "anti_air_filtered_until_air_seen": sorted(AA_ONLY) if not air_seen else [],
                },
            )

        self._decisions += 1
        if now - self._state_at >= 5.0:
            rate = self._decisions / max(1e-6, now - self._state_at) if self._state_at else None
            self._state_at, self._decisions = now, 0
            trace.write(
                "state", tick=obs.get("tick"), cash=cash, ore=econ.get("ore"), ore_capacity=econ.get("resource_capacity"), power=[provided, drained], income=round(income, 1),
                time=raw.get("time"), units=dict(owned), queue=[p.get("item") for p in production],
                idle_combat=len(idle_combat), enemies=len(enemies), enemy_buildings=len(enemy_buildings),
                military=obs.get("military"), plan_version=self.strategist.version,
                jev=(advice.values if advice else None), reactions=[r.name for r in fired],
                burn=[round(burn_struct, 1), round(burn_units, 1)], budget=round(budget, 1), scout=sorted(self._scouts),
                decisions_per_s=round(rate, 2) if rate is not None else None, army_value=army_value,
            )
        if not orders:
            return WAIT
        batch = orders[:MAX_BATCH]
        trace.write("orders", tick=obs.get("tick"), orders=[
            {k: v for k, v in o.items() if k != "units"} | ({"n": len(o["units"])} if "units" in o else {})
            for o in batch
        ])
        self._last_sent = batch
        return WithReasoning({"type": "orders", "orders": batch}, self._label(batch, guard is not None))

    @staticmethod
    def _label(batch: list[dict[str, Any]], defending: bool) -> str:
        """A few words on what this batch does (shown to spectators); the plan's essay stays in the trace."""
        if defending:
            return "defending the base"
        cmds = {o.get("cmd") for o in batch}
        if cmds & {"attack_move", "attack"}:
            return "attacking"
        made = [str(o.get("item")) for o in batch if o.get("cmd") in ("build", "train")]
        if made:
            return "producing " + ", ".join(dict.fromkeys(made))
        if "place" in cmds:
            return "placing " + ", ".join(str(o.get("item")) for o in batch if o.get("cmd") == "place")
        if "move" in cmds:
            return "moving"
        return str(next(iter(cmds), "orders"))

    # -- scouting ----------------------------------------------------------------

    def _scout(
        self, units: list[dict[str, Any]], legal: dict[str, Any], avail: set[str],
        entries: dict[str, list[dict[str, Any]]], game_s: float, home: tuple[int, int] | None, raw: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Keep one cheap fast unit sent toward the enemy so the model and Jev can see it coming."""
        alive = {u["id"] for u in units if isinstance(u.get("id"), int)}
        self._scouts &= alive
        enemy = _cell(legal.get("enemy_base"))
        if self._scouts or self._scouts_sent >= SCOUT_MAX or game_s < SCOUT_AT_S or enemy is None or home is None:
            return []
        idle = set(legal.get("idle_units") or [])
        kinds = {u["id"]: str(u.get("type")).lower() for u in units if isinstance(u.get("id"), int)}
        for kind in SCOUT_PREF:
            for uid, k in kinds.items():
                if k == kind and uid in idle and self._fresh("scout", 40.0):
                    mid = self._clamp((home[0] + (enemy[0] - home[0]) * 0.55, home[1] + (enemy[1] - home[1]) * 0.55),
                                      (raw.get("map") or {}).get("bounds"))
                    far = self._clamp(enemy, (raw.get("map") or {}).get("bounds"))
                    self._scouts.add(uid)
                    self._scouts_sent += 1
                    trace.write("scout", unit=uid, unit_type=kind, via=mid, to=far, game_s=round(game_s))
                    return [
                        {"cmd": "move", "units": [uid], "to": mid},
                        {"cmd": "move", "units": [uid], "to": far, "queued": True},
                    ]
        for kind in ("dog", "e1"):
            if kind in avail and len(entries.get("infantry", [])) < 3 and self._fresh("scout_train", 25.0):
                entries.setdefault("infantry", []).append({"item": kind, "ready": False, "progress": 0.0})
                return [{"cmd": "train", "item": kind, "count": 1}]
        return []

    def _check_schema(self, obs: dict[str, Any], econ: dict[str, Any], legal: dict[str, Any]) -> None:
        """Record the real observation's shape once, and say loudly if a field we rely on is missing."""
        missing = [f"observation.{k}" for k in EXPECTED["observation"] if k not in obs]
        missing += [f"economy.{k}" for k in EXPECTED["economy"] if k not in econ]
        missing += [f"legal_actions.{k}" for k in EXPECTED["legal_actions"] if legal and k not in legal]
        unit = next((u for u in obs.get("units") or [] if isinstance(u, dict)), {})
        trace.write("schema", observation=sorted(obs), economy=econ, unit_fields=sorted(unit),
                    legal_actions=sorted(legal), missing=missing)
        if missing:
            print(f"[redalert] observation is missing fields the agent relies on: {missing}", file=sys.stderr, flush=True)

    def _choose_doctrine(self, plan: Plan, numbers: dict[str, Any], now: float) -> Plan:
        """Jev's battle-plan pick (fresh, allowed, confident) or the rules' pick, with hysteresis; then apply it."""
        pick = conf = answer_id = None
        advice = self.advisor.current() if self.advisor.enabled else None
        if advice is not None and "doctrine" in plan.jev_slots:
            if self.advisor.now() - advice.at <= ADVICE_TTL_S:
                pick, conf, answer_id = advice.values.get("doctrine"), advice.confidence.get("doctrine"), advice.at
        rule = doctrines.fallback(numbers, self.doctrines.name)
        before = self.doctrines.name
        if not self.doctrines.since:
            self.doctrines.since = now  # the dwell clock starts when the war factory does
        if self.doctrines.update(now, pick, conf, rule, answer_id):
            trace.write("doctrine", name=self.doctrines.name, was=before, source=self.doctrines.source,
                        jev=pick, jev_confidence=conf, rules=rule, numbers=numbers)
        return doctrines.apply(self.doctrines.doctrine, plan)

    def _battle_summary(
        self, obs: dict[str, Any], now: float, home: tuple[int, int] | None, legal: dict[str, Any],
        idle: list[int], numbers: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._safe("intel", lambda: self.intel.summary(
            obs=obs, now=now, home=home, enemy_base=_cell(legal.get("enemy_base")), cost_of=self.cost_of,
            idle=set(idle),
        ), {})
        out["key_numbers"] = numbers
        self._battle = out
        return out

    def _filler(
        self, q: str, avail: set[str], owned: dict[str, int], plan: Plan, air_seen: bool, vs_armor: bool,
    ) -> str | None:
        """What an idle barracks / war factory should train next when the plan's counts are already met."""
        ok = [
            i for i in avail
            if self.queue_of(i) == q and i not in NON_COMBAT and (i not in AA_ONLY or air_seen)
        ]
        if not ok:
            return None
        if q == "infantry":
            if vs_armor and "e3" in ok:
                return "e3"  # rockets against tanks
            mix = [i for i in ("e1", "e3") if i in ok]
            if mix:  # roughly two riflemen per rocket soldier: cheap bodies plus anti-armour
                return "e3" if "e3" in mix and owned.get("e3", 0) * 2 < owned.get("e1", 0) else mix[0]
        for item, _ in plan.units:  # the plan's own unit choices first, in its order
            if item in ok:
                return item
        for item in FILLER_VEHICLES:
            if item in ok:
                return item
        return None

    # -- army --------------------------------------------------------------------

    def _defend(
        self, home: tuple[int, int] | None, buildings: list[dict[str, Any]], enemies: list[dict[str, Any]],
        army: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Reflex: enemy units in the base pull every fighter onto them, re-aimed every 1.5 s.

        ``None`` when the base is clear (the plan's army mode applies); a list (maybe empty while the
        re-aim timer runs) while it is not, so nothing else moves the army away mid-fight.
        """
        if home is None or not army:
            return None
        spots = [c for c in (_cell(b) for b in buildings) if c]
        intruders = [
            c for c in (_cell(e) for e in enemies if str(e.get("type")).lower() not in NOT_A_THREAT)
            if c and (math.dist(c, home) <= DEFEND_RADIUS or any(math.dist(c, s) <= 10 for s in spots))
        ]
        if not intruders:
            return None
        # A raid of one or two (a scout, a lone harasser) is handled by whoever is home; a push recalls everyone.
        fighters = [
            u["id"] for u in army if u["id"] not in self._scouts
            and (len(intruders) > 2 or (_cell(u) is not None and math.dist(_cell(u), home) <= 2 * DEFEND_RADIUS))
        ]
        if not fighters:
            return None
        out: list[dict[str, Any]] = []
        fresh = [u for u in fighters if u not in self._stanced]
        if fresh:
            self._stanced.update(fresh[:20])
            out.append({"cmd": "set_stance", "units": fresh[:20], "stance": "attack_anything"})
        if self._fresh("defend", DEFEND_EVERY_S):
            target = min(intruders, key=lambda c: math.dist(c, home))
            self._gathered.clear()  # after the fight, hold mode regroups everyone
            for i in range(0, min(len(fighters), 150), 50):
                out.append({"cmd": "attack_move", "units": fighters[i:i + 50], "to": [int(target[0]), int(target[1])]})
            trace.write("defend", intruders=len(intruders), fighters=len(fighters), to=list(target))
        return out

    def _target(self, target: Any, raw: dict[str, Any], obs: dict[str, Any], legal: dict[str, Any]) -> list[int] | None:
        bounds = (raw.get("map") or {}).get("bounds")
        base = _cell(legal.get("enemy_base"))
        cell: tuple[int, int] | None
        if isinstance(target, tuple):
            cell = target
        elif target == "building":
            mine = _cell(obs.get("base_center")) or (0, 0)
            seen = [
                c for c in (
                    _cell(e) for e in (obs.get("visible_enemy_buildings") or [])
                    if isinstance(e, dict) and not e.get("wreck")
                ) if c
            ]
            cell = min(seen, key=lambda c: math.dist(c, mine)) if seen else base
        elif target == "nearest":
            nearest = legal.get("attack_now")
            cell = (_cell(nearest.get("to")) if isinstance(nearest, dict) else None) or base
        else:
            cell = base
        return self._clamp(cell, bounds)

    @staticmethod
    def _clamp(cell: Any, bounds: Any) -> list[int] | None:
        if cell is None:
            return None
        x, y = int(cell[0]), int(cell[1])
        if isinstance(bounds, list) and len(bounds) == 4:
            bx, by, bw, bh = bounds
            x, y = max(bx, min(bx + bw - 1, x)), max(by, min(by + bh - 1, y))
        return [x, y]

    def _army(
        self, plan: Plan, raw: dict[str, Any], obs: dict[str, Any], legal: dict[str, Any],
        units: list[dict[str, Any]], adv: Callable[[str], Any] = lambda slot: None, defend: bool = False,
        forced_mode: str | None = None, idle_combat: list[int] | None = None, army_value: float = 0.0,
        enemy_seen: float = 0.0,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if idle_combat is None:
            idle_combat = [u for u in (legal.get("combat_units") or []) if isinstance(u, int)]
        fresh = [u for u in idle_combat if u not in self._stanced]
        if fresh:
            self._stanced.update(fresh[:20])
            out.append({"cmd": "set_stance", "units": fresh[:20], "stance": "attack_anything"})

        # Reactions and Jev may move the army between hold / attack / retreat, never out of a committed all-in.
        mode = plan.army_mode
        if mode != "all_in":
            if forced_mode:
                mode = forced_mode
            else:
                posture = adv("posture")
                if defend or posture == "retreat":
                    mode = "retreat"
                elif posture in ("hold", "attack"):
                    mode = posture
            if mode == "hold" and army_value >= ATTACK_VALUE and army_value >= 1.5 * enemy_seen \
                    and forced_mode is None and not defend:
                mode = "attack"  # a big army sitting at home is wasted: one match was lost with 8,800 of it idle

        bounds = (raw.get("map") or {}).get("bounds")
        home_cell = plan.gather or _cell(obs.get("base_center"))
        home = self._clamp(home_cell, bounds)
        fighters = [
            u["id"] for u in units if isinstance(u.get("id"), int)
            and str(u.get("type")).lower() not in NON_COMBAT and u["id"] not in self._scouts
        ]

        if mode == "retreat":
            if home and fighters and self._fresh("retreat", 5.0):
                for i in range(0, min(len(fighters), 150), 50):
                    out.append({"cmd": "move", "units": fighters[i:i + 50], "to": home})
            return out

        if mode == "hold":
            if home:
                pos = {u.get("id"): _cell(u) for u in units}
                far = [
                    u for u in idle_combat
                    if u not in self._gathered and (pos.get(u) is None or math.dist(pos[u], home) > 8)
                ]
                if far:
                    self._gathered.update(far[:50])
                    out.append({"cmd": "move", "units": far[:50], "to": home})
            return out

        name: Any = plan.target
        choice = adv("target")
        if isinstance(name, str) and choice in TARGET_NAMES:
            name = choice
        target = self._target(name, raw, obs, legal)
        if target is None:
            return out
        if mode == "all_in":
            if fighters and self._fresh("all_in", TTL["burst"]):
                for i in range(0, min(len(fighters), 150), 50):
                    out.append({"cmd": "attack_move", "units": fighters[i:i + 50], "to": target})
            return out

        # Fighters already out in the field (a wave that took its target) keep going without waiting for a wave.
        pos = {u.get("id"): _cell(u) for u in units}
        home_c = _cell(obs.get("base_center"))
        forward = [
            u for u in idle_combat
            if home_c and pos.get(u) and math.dist(pos[u], home_c) > FORWARD_RADIUS
        ]
        if forward and self._fresh("forward", 2.0):
            out.append({"cmd": "attack_move", "units": forward[:50], "to": target})
        idle_combat = [u for u in idle_combat if u not in forward]
        n = len(idle_combat)
        wave_ok = adv("wave")
        if isinstance(wave_ok, float):
            # Jev judges the moment, but never sends a token force or sits on a huge one forever.
            go = n >= max(3, plan.wave_min // 2) and (wave_ok >= WAVE_YES or n >= max(20, 2 * plan.wave_min))
        else:
            go = n >= plan.wave_min
        if not go and army_value >= ATTACK_VALUE and n >= max(3, plan.wave_min // 2):
            go = True  # big enough to attack: reinforcements join in groups
        if go:
            self._gathered.difference_update(idle_combat)
            out.append({"cmd": "attack_move", "units": idle_combat[:50], "to": target})
        return out

    # -- the model's one-off raw orders ------------------------------------------

    def _raw_orders(
        self, plan: Plan, raw: dict[str, Any], obs: dict[str, Any], units: list[dict[str, Any]],
        buildings: list[dict[str, Any]], avail: set[str],
    ) -> list[dict[str, Any]]:
        version = self.strategist.version
        if version == self._exec_version or not plan.orders:
            self._exec_version = version
            return []
        self._exec_version = version
        bounds = (raw.get("map") or {}).get("bounds")
        bounds_t = tuple(bounds) if isinstance(bounds, list) and len(bounds) == 4 else None
        unit_ids = {u["id"] for u in units if isinstance(u.get("id"), int)}
        building_ids = {b["id"] for b in buildings if isinstance(b.get("id"), int)}
        enemies = (obs.get("visible_enemies") or []) + (obs.get("visible_enemy_buildings") or [])
        enemy_ids = {e["id"] for e in enemies if isinstance(e, dict) and isinstance(e.get("id"), int)}
        core_ids = {
            b["id"] for b in buildings
            if str(b.get("type")).lower() in CORE_STRUCTURES and isinstance(b.get("id"), int)
        }
        clean = (sanitize_order(o, unit_ids, building_ids, enemy_ids, bounds_t, avail) for o in plan.orders)  # type: ignore[arg-type]
        return [o for o in clean if o and not (o["cmd"] == "sell" and o.get("building") in core_ids)]


def create_agent():
    return RedAlertAgent()
