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

import dataclasses
import math
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, WithReasoning
from common.jev import JevClient
from common.llm import GeminiClient
from redalert import doctrine as doctrines
from redalert import roles, trace
from redalert.econ import EconTracker, funds
from redalert.intel import Intel
from redalert.plan import DEFAULT_PLAN, Plan, Reaction, sanitize_order
from redalert.strategist import Strategist
from redalert.tactics import ADVICE_TTL_S, ESCALATE_LEVEL, Candidates, TacticalAdvisor

CORE_STRUCTURES = {"fact", "powr", "apwr", "proc", "weap", "tent", "barr"}
NON_COMBAT = {"mcv", "harv", "medi", "mech", "e6", "spy", "thf", "truk"}
DEFENSES = {"gun", "pbox", "hbox", "ftur", "tsla", "sam", "agun"}
STRUCTURES = {
    "powr",
    "apwr",
    "tent",
    "barr",
    "proc",
    "weap",
    "fix",
    "dome",
    "atek",
    "stek",
    "silo",
    "hpad",
    "afld",
    "syrd",
    "spen",
    "kenn",
    "fcom",
    "iron",
    "pdox",
    "mslo",
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
FALLBACK_SECS = {
    "powr": 14,
    "barr": 20,
    "proc": 42,
    "weap": 45,
    "fix": 34,
    "harv": 26,
    "e1": 3,
    "e3": 5,
    "3tnk": 30,
}
FALLBACK_COST = {
    "powr": 300,
    "tent": 500,
    "barr": 500,
    "proc": 1400,
    "weap": 2000,
    "harv": 1100,
    "e1": 100,
    "e3": 300,
    "fix": 1200,
    "tsla": 1500,
    "pbox": 400,
    "ftur": 600,
    "gun": 600,
    "3tnk": 950,
    "2tnk": 800,
    "1tnk": 700,
    "apc": 850,
    "ftrk": 600,
    "jeep": 500,
    "dog": 200,
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
# running at once starve each other (a harvester alone burns ~42/s against ~20/s income per harvester), and at
# cash 0 the engine draws credits across ALL queues proportionally - everything crawls at once, including the
# harvesters that would fix the income. So: one shared draw ledger (structures + units together), a cash floor
# below which only the economy commits, and a cancel reflex that frees a stuck ledger.
MIN_BUDGET = 35.0  # credits/s allowed before income is known (a refinery burns ~33/s)
BUDGET_FACTOR = (
    1.5  # of measured income: queues may run slightly ahead of income, cash cushions the rest
)
CASH_MARGIN = 300
CASH_FLOOR = 300.0  # below this only harvesters and power commit (the income comes first, always)
BROKE_CANCEL_S = 10.0  # cash pinned at zero this long: cancel the worst non-essential commitment
PRE_QUEUE_PROGRESS = 0.4  # queue the next structure once the current one is this far along
SCOUT_AT_S = 70
SCOUT_MAX = 1  # one scout early: re-sending lone scouts fed the enemy five free kills
SCOUT_RESEND_S = (
    90.0  # a dead scout is replaced after this long: a match played blind is a match played wrong
)
SCOUT_SECOND_MIN_CASH = (
    600.0  # the General's second scout (blind for a minute) needs only this much cash
)
# Locked until the war factory is up; no plan may reorder it. Barracks second: the enemy's opening scout
# walked through our empty base in live matches, and a few riflemen kill it (and anything that follows).
OPENING = (("powr", 1), ("barracks", 1), ("proc", 1), ("weap", 1))
EARLY_GUARD = (
    3  # infantry trained before the war factory; the rest of the money is the war factory's
)
BARRACKS = ("barr", "tent")
# Army floor: after this many seconds, combat value (credits) must keep pace with the clock or units go first
# and non-essential structures (2nd refinery, radar, tech) wait.
ARMY_FLOOR_FROM_S, ARMY_FLOOR_PER_S = 100.0, 20.0
DEFEND_RADIUS = 24  # enemy units this close to the base centre (or 10 to any building) trigger the defence reflex
DEFEND_EVERY_S = 1.5
DEFEND_MASS_RATIO = 0.8  # intruders worth this much of our fighters (and over two of them): mass at home, do not chase
POWER_MARGIN = 40  # after the opening, keep this much spare power: low power slows every queue
ORE_FULL = 0.75  # ore store this full: spend it now (a full store stops the harvesters unloading)
NOT_A_THREAT = {"harv", "mcv", "truk"}  # enemy units that never pull the army home
CRITICAL = {"fact", "proc", "weap"}  # always worth repairing; the rest only with money to spare
EXPECTED = {  # documented observation fields we rely on; a missing one is reported once (the "ore" lesson)
    "observation": (
        "tick",
        "economy",
        "military",
        "units",
        "buildings",
        "production",
        "available_production",
        "visible_enemies",
        "visible_enemy_buildings",
        "base_center",
        "events",
    ),
    "economy": (
        "cash",
        "ore",
        "power_provided",
        "power_drained",
        "resource_capacity",
        "harvesters",
    ),
    "legal_actions": ("deploy", "combat_units", "idle_units", "harvesters", "enemy_base"),
}
ATTACK_VALUE = 5000  # an army worth this much (credits) attacks even if the plan or Jev says hold
FORWARD_RADIUS = (
    36  # idle fighters farther than this from home are already in the field: keep them fighting
)
FORWARD_MIN = (
    3  # fewer than this in the field and they regroup at the staging point instead of pressing
)
WAVE_COOLDOWN_S = (
    25.0  # after a wave leaves, the next one assembles at the muster point instead of trickling out
)
WAVE_MUSTER_RADIUS = 8.0  # idle fighters within this of the muster point count as assembled
FILLER_VEHICLES = ("3tnk", "2tnk", "1tnk", "4tnk", "v2rl", "arty", "apc", "jeep")
OPENING_S = 150  # until then (or a war factory) defences wait unless there is an emergency
TOWER_BY_S = ((200, 1), (300, 2))  # baseline defence timeline: (game seconds, towers)
ATTACK_MEMORY_S = 8.0
ENDGAME_PUSH_S = 300.0  # inside five minutes, behind on trades, only destroying buildings can still win: push all
# Battalions: a defence line in front of the base plus flank packs that raid from the sides, so the army is
# never one blob that is either all defending or all attacking (a lesson from an opponent match).
GUARD_FRONT_TILES = 10.0  # the defence line sits this far from base center toward the enemy
GUARD_RADIUS = 6.0  # holders within this of the line are in position
GUARD_STANCE = "defend"  # engage in range, never chase: the "stance on the hill"
FLANK_CD_S = 60.0  # between flank runs; Jev's read of the enemy style shortens or lengthens this
FLANK_SIZE = 3
FLANK_LIFE_S = 50.0  # a run ends after this; its units rejoin the main force where they stand
FLANK_OFFSET = 14.0  # how wide: perpendicular offset from the target's center, sides alternate
FLANK_FROM_S = 150.0  # no flank runs while the opening is still forming
FLANK_MIN_ARMY = 6
STYLE_FLANK_CD_S = {
    "turtle": 40.0,
    "mass_infantry": 40.0,
    "harass": 45.0,
    "mixed": 55.0,
    "mass_armor": 70.0,
}
FINISH_AHEAD_RATIO = (
    1.5  # army worth this much more than the strongest enemy seen: press to finish them
)
FINISH_MIN_SEEN = 1000.0
FEED_BEHIND_RATIO = (
    0.7  # below this against a big seen enemy, mid-game attacks just feed the tiebreak
)
FEED_MIN_SEEN = 2000.0
AHEAD_COOLDOWN_S = (
    12.0  # when clearly ahead, waves follow twice as fast: constant pressure to finish
)
MODE_HOLD_S = (
    6.0  # minimum dwell between posture flips, so half the army never walks home mid-fight
)
THREAT_RADIUS = 40.0  # raiders this close to home turn the defence line to face them
HARV_FLEE_RADIUS = (
    12.0  # raiders this close to a harvester pull it home: income and tiebreak losses
)
PRODUCERS = {"weap", "barr", "tent"}  # unit-production buildings worth a rally point
ENEMY_SURGE_VALUE = (
    1500.0  # enemy army value appearing within one window: they are staging a push, re-plan
)
ENEMY_SURGE_WINDOW_S = 30.0
ENEMY_TECH = {
    "weap",
    "fix",
    "proc",
}  # new enemy production structures: vehicles, heavy tanks, or an expansion
ALIAS_ITEMS = {"barracks", "tower", "aa"}


def _cell(value: Any) -> tuple[int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(v, (int, float)) for v in value[:2])
    ):
        return int(value[0]), int(value[1])
    if (
        isinstance(value, dict)
        and isinstance(value.get("x"), (int, float))
        and isinstance(value.get("y"), (int, float))
    ):
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
        self.advisor = (
            advisor
            if advisor is not None
            else TacticalAdvisor(
                JevClient.from_env(), telemetry_path=os.environ.get("RA_TELEMETRY") or None
            )
        )
        if strategist is None:
            # Opt-in: in live matches the LLM kept the army on "hold" for ten minutes at ~6.5 s a consult, while the
            # built-in playbook attacks on its own. RA_LLM=1 turns the Gemini strategist back on.
            on = os.environ.get("RA_LLM", "0").strip().lower() in ("1", "true", "yes", "on")
            strategist = Strategist(GeminiClient.from_env() if on else None)
        self.strategist = strategist
        if not self.strategist.enabled:
            print(
                "[redalert] playing the built-in playbook (Gemini off; RA_LLM=1 turns it on)",
                file=sys.stderr,
                flush=True,
            )
        self.econ = EconTracker()
        self._until: dict[str, float] = {}
        self._stanced: set[int] = set()
        self._gathered: set[int] = set()
        self.intel = Intel()
        self.doctrines = doctrines.DoctrineState()
        self._battle: dict[str, Any] = {}
        self._scouts: set[int] = set()
        self._scouts_sent = 0
        self._scout_at = -1e9
        self.general = roles.General()
        self._directive = self.general.decide(
            game_s=0, army_n=0, doctrine=doctrines.DEFAULT, enemy_types={}
        )
        self._enemy_seen_at = 0.0  # last time any enemy unit or building was in view
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
        self._schema_checked = (
            False  # choose_action calls since the last state trace: our real decision rate
        )
        self._attack_seq = -1
        self._attack_at = -1e9
        self._wave_until = -1e9
        self._guard: set[int] = set()  # the home defence line, sticky while its members live
        self._guard_stanced: set[int] = set()
        self._flankers: dict[
            int, tuple[list[int], float]
        ] = {}  # id -> (waypoint, game time the run ends)
        self._flank_at = -1e9
        self._flank_side = 1
        self._enemy_seen_mark = (0.0, -1e9)  # (value, when) for spotting a surge
        self._surge_at = -1e9
        self._enemy_tech_seen: set[str] = set()
        self._mode_cur = ""  # the resolved army mode, with hysteresis
        self._mode_at = -1e9
        self._fired: set[str] = set()
        self._broke_since: float | None = None

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
                (
                    now,
                    f"{order.get('cmd')} {item or ''}: {reason} {row.get('detail') or ''}".strip()[
                        :200
                    ],
                )
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

    def _group_want(
        self,
        item: str,
        count: int,
        faction: str,
        avail: set[str],
        owned: dict[str, int],
        queued: dict[str, int],
    ) -> tuple[str, int] | None:
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
                1
                for e in c["enemies"]
                if home
                and (cell := _cell(e))
                and math.dist(cell, home) <= radius
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
                print(
                    f"[redalert] {name} failed, continuing without it: {key}",
                    file=sys.stderr,
                    flush=True,
                )
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
        enemies = [
            e
            for e in (obs.get("visible_enemies") or [])
            if isinstance(e, dict) and not e.get("wreck")
        ]
        enemy_buildings = [
            e for e in (obs.get("visible_enemy_buildings") or []) if isinstance(e, dict)
        ]
        home = _cell(obs.get("base_center"))
        game_s = (raw.get("time") or {}).get("game_seconds")
        game_s = (
            float(game_s)
            if isinstance(game_s, (int, float))
            else float(obs.get("tick") or 0) / 25.0
        )
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
            entries.setdefault(q, []).append(
                {
                    "item": item,
                    "ready": bool(p.get("ready")),
                    "progress": float(progress) if isinstance(progress, (int, float)) else 0.0,
                }
            )
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

        idle_combat = [
            u
            for u in (legal.get("combat_units") or [])
            if isinstance(u, int) and u not in self._scouts
        ]
        all_combat = [
            u
            for u in units
            if isinstance(u.get("id"), int)
            and str(u.get("type")).lower() not in NON_COMBAT
            and u["id"] not in self._scouts
        ]
        air_seen = any(str(e.get("type")).lower() in ENEMY_AIR for e in enemies)

        # The battle picture (both sides, with memory through the fog) and the battle plan in force.
        self._safe("intel", lambda: self.intel.update(obs, mono), None)
        refineries = owned.get("proc", 0)
        army_value = sum(self.cost_of(str(u.get("type")).lower()) for u in all_combat)
        enemy_seen = self._safe("intel", lambda: self.intel.enemy_army_value(self.cost_of), 0.0)
        # Towers count as opposition: an army of zero behind guns has been winning the trades tiebreak
        # against our pushes all match. Seen and not yet destroyed, so fog does not hide a defended base.
        defence_seen = self._safe(
            "intel", lambda: self.intel.enemy_defence_value(self.cost_of), 0.0
        )
        # The general re-plans when the enemy MOVES, not just on a timer: an army surge or a new production
        # structure each wake the strategist (Jev's style change is the third trigger, in tactics).
        prev, prev_at = self._enemy_seen_mark
        if (
            enemy_seen >= prev + ENEMY_SURGE_VALUE
            and mono - prev_at <= ENEMY_SURGE_WINDOW_S
            and mono - self._surge_at >= 60.0
        ):
            self._surge_at = mono
            self._safe(
                "strategist",
                lambda: self.strategist.notify(f"enemy_army_surge:{int(enemy_seen - prev)}"),
                None,
            )
        if enemy_seen > prev or mono - prev_at > 60.0:
            self._enemy_seen_mark = (enemy_seen, mono)
        for tech in {str(b.get("type")).lower() for b in enemy_buildings} & ENEMY_TECH:
            if tech not in self._enemy_tech_seen:
                self._enemy_tech_seen.add(tech)
                self._safe(
                    "strategist", lambda t=tech: self.strategist.notify(f"enemy_tech:{t}"), None
                )
        near_n, near_v = self._safe(
            "intel", lambda: self.intel.enemy_near(home, self.cost_of, mono), (0, 0.0)
        )
        towers_up = sum(n for t, n in owned.items() if t in DEFENSES or t in TOWER_ORDER)
        army_floor = max(0.0, (game_s - ARMY_FLOOR_FROM_S) * ARMY_FLOOR_PER_S)
        military = obs.get("military") or {}
        numbers = doctrines.battle_state(
            game_s=game_s,
            remaining_s=(raw.get("time") or {}).get("remaining_seconds"),
            my_army=army_value,
            enemy_army_seen=enemy_seen,
            enemy_near_base=near_n,
            enemy_army_near_base=near_v,
            my_losses=military.get("deaths_cost") or 0,
            my_kills=military.get("kills_cost") or 0,
            income=income,
            harvesters=owned.get("harv", 0),
            refineries=refineries,
            towers=towers_up,
            army_floor=army_floor,
            enemy_defence=defence_seen,
        )
        if (
            owned.get("weap") and "doctrine" in plan.jev_slots
        ):  # the opening is locked; a plan may pin its army
            plan = self._safe("doctrine", lambda: self._choose_doctrine(plan, numbers, mono), plan)
        # The General reads the battle and issues the orders every role takes: garrison, raiders, scouts, counters.
        directive = self._safe(
            "general",
            lambda: self._command(
                game_s,
                len(all_combat),
                enemies=enemies,
                enemy_buildings=enemy_buildings,
                near_n=near_n,
                mono=mono,
            ),
            self._directive,
        )
        # Where my army is, in credits: committed in the field (beyond forward radius) vs assembled at home.
        # This is the number that makes "attack now" vs "assemble" a real choice for the wave answers.
        home_v = sum(
            self.cost_of(str(u.get("type")).lower())
            for u in all_combat
            if home is None or _cell(u) is None or math.dist(_cell(u), home) <= FORWARD_RADIUS
        )
        numbers["my_army_value_at_home"] = round(home_v)
        numbers["my_army_value_in_field"] = round(max(0.0, army_value - home_v))
        doc = self.doctrines.doctrine
        army_floor *= doc.floor_scale

        # Reactions: the model's if-then rules, evaluated in microseconds.
        fired = self._safe(
            "reactions",
            lambda: self._react(
                plan,
                {
                    "home": home,
                    "enemies": enemies,
                    "drained": drained,
                    "provided": provided,
                    "air_seen": air_seen,
                    "cash": cash,
                    "game_s": game_s,
                    "army": len(all_combat),
                    "tick": obs.get("tick"),
                    "enemy_building_types": {str(e.get("type")).lower() for e in enemy_buildings},
                },
            ),
            [],
        )
        react_wants: list[tuple[str, int]] = []
        react_units: list[tuple[str, int]] = []
        forced_mode: str | None = None
        army_first = plan.army_first
        for r in fired:
            for action in r.do:
                if "want_building" in action:
                    react_wants.append(
                        (action["want_building"]["item"], action["want_building"]["count"])
                    )
                elif "want_unit" in action:
                    react_units.append((action["want_unit"]["item"], action["want_unit"]["count"]))
                elif "army" in action and forced_mode is None:
                    forced_mode = action["army"]
                elif action.get("army_first"):
                    army_first = True
        # The time limit is decided on trades (kills - losses) first, assets second: hoarding units behind a
        # negative trade score loses no matter how big the army is. So the closing minutes are deterministic —
        # behind on trades, everything pushes for the destruction win; ahead, the wave rules protect the lead.
        # A live threat or a reaction still overrides this: defending wins trades when the enemy attacks us.
        trades = numbers.get("kills_value_minus_losses_value")
        remaining = numbers.get("remaining_s")
        if (
            forced_mode is None
            and isinstance(trades, (int, float))
            and trades < 0
            and isinstance(remaining, (int, float))
            and remaining <= ENDGAME_PUSH_S
        ):
            forced_mode = "all_in"

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
                p.get("ready")
                and item not in placed
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
            w = (
                self._group_want(item, count, faction, avail, owned, queued)
                if item in ("tower", "aa")
                else (self._resolve(item, faction, avail, owned), count)
            )
            if w:
                wants.insert(0, w)
        for item, count in reversed(react_units):
            if item not in AA_ONLY or air_seen:
                unit_wants.insert(0, (item, count))

        # Baseline defence timeline and automatic harvesters (the model may ask for more).
        for at_s, n in reversed(TOWER_BY_S):
            if game_s >= at_s:
                w = self._group_want("tower", n, faction, avail, owned, queued)
                if w and owned.get(
                    "weap"
                ):  # the Defense queue is separate: this never delays structures
                    wants.append(w)
                break
        if doc.extra_towers and (owned.get("weap") or near_n):
            w = self._group_want(
                "tower",
                min(MAX_DEFENSES, towers_up + doc.extra_towers),
                faction,
                avail,
                owned,
                queued,
            )
            if w:
                wants.insert(0, w)
        # Harvesters are the income: the war factory's 2,000 must not leave the first refinery with one harvester
        # for two minutes (the lost match sat at cash 0 from 126 s on with one or two).
        harv_goal = (
            min(MAX_HARV, doc.harv_per_proc * refineries)
            if owned.get("weap")
            else (2 if refineries else 0)
        )
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
            return not es or (
                len(es) == 1 and (es[0]["ready"] or es[0]["progress"] >= PRE_QUEUE_PROGRESS)
            )

        def unit_slot_free(item: str) -> bool:
            q = self.queue_of(item)
            if item == "harv":
                # One harvester queued at a time: two filled the whole vehicle queue and no tank got built
                # for minutes (tanks_at_330s was 0 in a lost match while two harvesters crawled at cash 0).
                return not any(e["item"] == "harv" and not e["ready"] for e in entries.get(q, []))
            return len(entries.get(q, [])) < (3 if q == "infantry" else 2)

        struct_cands = tuple(
            dict.fromkeys(
                i for i, c in wants if short(i, c) and i in avail and slot_free(self.queue_of(i))
            )
        )
        unit_cands = tuple(
            dict.fromkeys(
                i
                for i, c in unit_wants
                if short(i, c) and i in avail and self.cost_of(i) <= cash and unit_slot_free(i)
            )
        )

        advice = None
        roster = tuple(
            dict.fromkeys(
                i
                for i in avail
                if i not in NON_COMBAT
                and (i not in AA_ONLY or air_seen)
                and self.queue_of(i) in ("infantry", "vehicle")
            )
        )
        if self.advisor.enabled:
            self._safe(
                "jev",
                lambda: self.advisor.maybe_refresh(
                    raw,
                    plan,
                    Candidates(
                        structures=struct_cands,
                        units=unit_cands,
                        idle_combat=len(idle_combat),
                        all_combat=len(all_combat),
                        has_harvester_item="harv" in avail,
                        has_power_item="powr" in avail,
                        describe={
                            i: f"cost {self.cost_of(i)}"
                            for i in (*struct_cands, *unit_cands, *roster)
                        },
                        costs=dict(self._cost),
                        formation={
                            "scout": len(self._scouts),
                            "guard": len(self._guard),
                            "flanker": len(self._flankers),
                            "striker": max(
                                0,
                                len(all_combat)
                                - len(self._scouts)
                                - len(self._guard)
                                - len(self._flankers),
                            ),
                        },
                        income=income,
                        scout_alive=bool(self._scouts),
                        burn_struct=burn_struct,
                        burn_units=burn_units,
                        budget=budget,
                        combat_roster=roster,
                        battle=self._battle_summary(obs, mono, home, legal, idle_combat, numbers),
                        doctrine=self.doctrines.name,
                        doctrine_for_s=mono - self.doctrines.since,
                    ),
                ),
                None,
            )
            advice = self.advisor.current()
        now_adv = self.advisor.now()

        def adv(slot: str) -> Any:
            return advice.get(slot, plan, now=now_adv) if advice else None

        pick = adv("structure")
        if pick in struct_cands:
            probs = advice.probs.get("structure") if advice else None
            if probs:  # soft-rank the whole menu by Jev's probability mass, not one binary pick
                wants.sort(key=lambda w: -probs.get(w[0], 0.0))
            else:
                wants.sort(
                    key=lambda w: w[0] != pick
                )  # stable: Jev's pick first, the rest keep plan order
        pick = adv("unit")
        if pick in unit_cands:
            probs = advice.probs.get("unit") if advice else None
            if probs:
                unit_wants.sort(key=lambda w: -probs.get(w[0], 0.0))
            else:
                unit_wants.sort(key=lambda w: w[0] != pick)
        focus = adv("focus")
        if isinstance(focus, str) and focus in roster:
            # Jev's composition call goes in front of the plan's lists and stays wanted while short.
            unit_wants.insert(0, (focus, owned.get(focus, 0) + queued.get(focus, 0) + 6))
        # Combined arms: an infantry-only army loses to a mixed one (the lost match built a war factory and no
        # tank all game). Below the General's vehicle share the best counter vehicle goes first, and the unit
        # loop saves for it instead of letting cheaper infantry eat the money every tick.
        tank_item: str | None = None
        if owned.get("weap") and army_value > 0 and game_s >= 120:
            veh_value = sum(
                self.cost_of(str(u.get("type")).lower())
                for u in all_combat
                if self.queue_of(str(u.get("type")).lower()) == "vehicle"
            )
            in_prod = sum(
                n for i, n in queued.items() if self.queue_of(i) == "vehicle" and i != "harv"
            )
            if veh_value < directive.vehicle_share * army_value and not in_prod:
                tank_item = next(
                    (
                        i
                        for i in (*directive.units, *FILLER_VEHICLES)
                        if i in avail and self.queue_of(i) == "vehicle" and i not in NON_COMBAT
                    ),
                    None,
                )
                if tank_item:
                    unit_wants.insert(
                        0, (tank_item, owned.get(tank_item, 0) + queued.get(tank_item, 0) + 1)
                    )
        unit_wants.sort(
            key=lambda w: w[0] != "harv"
        )  # harvesters stay first under any ranking (stable)
        need = adv("harvester")
        harvs = owned.get("harv", 0) + queued.get("harv", 0)
        if (
            isinstance(need, float)
            and need >= YES
            and "harv" in avail
            and harvs < MAX_HARV
            and not army_short
        ):
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
        elif (
            owned.get("weap")
            and provided - drained < POWER_MARGIN
            and "powr" in avail
            and not queued.get("powr")
            and owned.get("powr", 0) < MAX_POWR
        ):
            # Before the next building tips us into low power (40 s of it in one match), not after.
            wants.insert(0, ("powr", owned.get("powr", 0) + 1))
        ore, capacity = econ.get("ore"), econ.get("resource_capacity")
        ore_full = (
            isinstance(ore, (int, float))
            and isinstance(capacity, (int, float))
            and capacity > 0
            and ore >= ORE_FULL * capacity
        )
        if ore_full and "silo" in avail and owned.get("silo", 0) + queued.get("silo", 0) < 3:
            wants.append(("silo", owned.get("silo", 0) + queued.get("silo", 0) + 1))

        # Army floor (computed above): once the war factory is up, an army that lags the clock gets the money first.
        essential: set[str] = {"powr", "weap", *BARRACKS}
        if (
            refineries < 2
        ):  # (counts finished ones only: it stays protected from the cancel reflex while it builds)
            essential.add(
                "proc"
            )  # the second refinery is the economy: an army floor must not postpone it
        unlocks: set[str] = set()
        for (
            item,
            count,
        ) in unit_wants:  # what unlocks a wanted unit is essential too (e.g. fix for heavy tanks)
            if item not in avail:
                unlocks.update(n for n in self._needs.get(item) or [] if not owned.get(n))
        essential |= unlocks
        if army_short:
            army_first = True

        # Whoever has priority (structures, or units when army_first) sees the other side's commitments as free.
        free_s = max(0.0, cash - committed_s - (committed_u if army_first else 0.0))
        free_u = max(0.0, cash - committed_u - (0.0 if army_first else committed_s))
        spent_s = 0.0  # structures ordered this decision
        spent_u = 0.0  # units ordered this decision
        emergency = bool(fired) or defend or drained > provided
        # When Jev's stage says the money must pivot to the army, structures reserve the army's next order's
        # draw in the ledger, so a structure cannot spend the budget the units were sent to claim. Doctrine
        # army_first does not reserve: the army floor and the doctrine's own unit wants pace it already.
        army_claim = 0.0
        if stage in ("build_army", "defend_now"):
            for i, c in unit_wants:
                if i == "harv" or not short(i, c) or i not in avail:
                    continue
                army_claim = self.rate_of(i)
                break

        def affordable(
            cost: float,
            rate: float,
            *,
            floor_exempt: bool = False,
            budget_exempt: bool = False,
            reserve: float = 0.0,
        ) -> bool:
            """The spending governor, as one ledger. Every queue draws credits per second while it works, and at
            cash 0 the engine draws from ALL queues at once - so committing more than the income supports starves
            everything at once, harvesters included (the lost-match trace: 250/s committed against 5/s of income,
            cash at 0 for four minutes, no tank ever finished). So the total draw of structures + units together
            must fit the budget, or cash above the floor must cover every remaining commitment plus this price
            outright. Below the floor only the economy (harvesters, power) commits; an emergency overrides the
            floor because surviving the next thirty seconds outranks solvency.
            """
            if not floor_exempt and not emergency and cash < CASH_FLOOR:
                return False
            if budget_exempt or burn_struct + burn_units + rate + reserve <= budget:
                return True
            return (
                cash - (0.0 if floor_exempt else CASH_FLOOR)
                >= committed_s + committed_u + spent_s + spent_u + cost + reserve
            )

        # Structures: priority order; one per queue, pre-queued, within the spending budget.
        held: set[str] = (
            set()
        )  # queues where a higher-priority want is still waiting on a prerequisite
        cooling: set[str] = (
            set()
        )  # queues whose top want is inside its resend cooldown: nothing may slip past it
        for item, count in wants:
            if not short(item, count):
                continue
            q = self.queue_of(item)
            if item not in avail:
                needs = self._needs.get(item) or []
                if not (
                    needs
                    and all(owned.get(n) or queued.get(n) for n in needs)
                    and any(queued.get(n) for n in needs)
                ):
                    diag.append(
                        f"wanted {item} x{count} is not in available_production (prerequisites?)"
                    )
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
            if not affordable(
                cost, rate, floor_exempt=(item == "powr" and drained > provided), reserve=army_claim
            ):
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
                if not (
                    needs
                    and all(owned.get(n) or queued.get(n) for n in needs)
                    and any(queued.get(n) for n in needs)
                ):
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
            rate = (
                self.rate_of(item) * n
            )  # a 3-rifleman order draws three units' worth while it trains
            if (
                item != "harv"
            ):  # harvesters are the economy: never held back, though they count toward the burn
                if not affordable(cost, rate):
                    if item == tank_item and not saving_for:
                        saving_for = float(
                            cost
                        )  # the vehicle comes first: infantry buys only the surplus
                        used.add(q)
                    continue
                if spendable - saving_for < cost:
                    if not saving_for and spendable < cost <= 2 * (spendable + budget * 8):
                        saving_for = cost  # within reach (~8 s of income): save for it instead of buying cheaper things
                        used.add(q)
                    continue
            if self._cooling(f"train:{item}"):
                used.add(
                    q
                )  # hold the queue: a lower-priority unit must not take the slot this tick
                continue
            if not self._fresh(f"train:{item}", TTL["train"]):
                continue
            orders.append({"cmd": "train", "item": item, "count": n})
            entries.setdefault(q, []).append({"item": item, "ready": False, "progress": 0.0})
            used.add(q)
            if item != "harv":
                spendable -= cost
                spent_u += cost
                if army_first:
                    free_s = max(0.0, free_s - cost)
            burn_units += rate
            trained = True

        # Keep busy: a barracks or war factory with nothing queued is wasted. Once the plan's counts are met,
        # keep training (the opponent that beat us never let its barracks idle).
        waiting = any(
            short(i, c) and i in avail and i in essential and not queued.get(i) for i, c in wants
        )
        if (
            not saving_for and not waiting
        ):  # an unfunded war factory / depot / barracks gets the money first
            enemy_vehicles = sum(
                1 for e in enemies if self.queue_of(str(e.get("type")).lower()) == "vehicle"
            )
            enemy_infantry = sum(
                1 for e in enemies if self.queue_of(str(e.get("type")).lower()) == "infantry"
            )
            for q in ("infantry", "vehicle"):
                if q in used or entries.get(q):
                    continue
                guard_left = EARLY_GUARD - sum(owned.get(i, 0) + queued.get(i, 0) for i in INFANTRY)
                if not owned.get("weap") and guard_left <= 0:
                    continue  # a guard, not an army: the war factory comes first
                item = self._filler(
                    q,
                    avail,
                    owned,
                    plan,
                    air_seen,
                    enemy_vehicles > enemy_infantry or directive.stance == "armor",
                )
                if item is None or self._cooling(f"train:{item}"):
                    continue
                unit_cost = max(1, self.cost_of(item))
                n = min(3, max(1, int(spendable // unit_cost))) if q == "infantry" else 1
                if not owned.get("weap"):
                    n = min(n, guard_left)
                # Floating cash (or an army behind the clock) is what this spends; the plan's wants came first.
                margin = 0 if (army_short or ore_full) else CASH_MARGIN
                if (
                    spendable < unit_cost + margin
                    or not self._fresh(f"train:{item}", TTL["train"])
                    or not affordable(unit_cost * n, self.rate_of(item) * n)
                ):
                    continue
                orders.append({"cmd": "train", "item": item, "count": n})
                entries.setdefault(q, []).append({"item": item, "ready": False, "progress": 0.0})
                spendable -= unit_cost * n
                spent_u += unit_cost * n
                burn_units += self.rate_of(item) * n
                trained = True

        orders += self._safe(
            "scout", lambda: self._scout(units, legal, avail, entries, game_s, home, cash, raw), []
        )

        # Cancel reflex: cash pinned at zero means every queue is crawling at once (the engine shares the
        # incoming credits across them) - even the harvesters that would restore the income. Drop the worst
        # non-essential commitment so the ledger actually drains; OpenRA refunds the progress paid so far.
        if cash < 50:
            self._broke_since = self._broke_since if self._broke_since is not None else mono
        else:
            self._broke_since = None
        if (
            owned.get("weap")
            and not emergency
            and self._broke_since is not None
            and mono - self._broke_since >= BROKE_CANCEL_S
            and burn_struct + burn_units > 2.0 * max(income, 1.0)
            and self._fresh("cancel", 8.0)
        ):
            # The harvest is protected, and so is power - but only when the base actually needs it. A plant
            # queued while the lights are on is just another crawler drawing from the recovery harvester.
            protected = essential | {"harv"}
            if drained <= provided:
                protected = protected - {"powr"}
            victims = [
                p
                for p in production
                if not p.get("ready") and str(p.get("item")).lower() not in protected
            ]
            if victims:

                def owed(p: dict[str, Any]) -> float:
                    prog = p.get("progress")
                    return self.cost_of(str(p.get("item")).lower()) * (
                        1.0 - prog if isinstance(prog, (int, float)) else 1.0
                    )

                victim = max(victims, key=owed)
                orders.append({"cmd": "cancel", "item": victim.get("item")})
                trace.write(
                    "cancel",
                    item=victim.get("item"),
                    progress=victim.get("progress"),
                    cash=cash,
                    income=round(income, 1),
                    burn=[round(burn_struct, 1), round(burn_units, 1)],
                )

        unmet_structure = any(short(i, c) for i, c in wants)
        if not unmet_structure and not trained and not production and cash > 1500:
            diag.append(f"{cash} cash unspent and nothing is queued or wanted")
        if (
            not diag
            and not unmet_structure
            and not production
            and all(
                owned.get(i, 0) + queued.get(i, 0) >= c
                for i, c in (*plan.buildings, *plan.units)
                if i not in ALIAS_ITEMS
            )
        ):
            diag.append("every structure and unit want in the plan is satisfied")

        # Reflex 3: repairs. Repairing costs money, so only the core buildings when cash is short.
        for b in buildings:
            if (
                (b.get("hp", 100) or 100) < 75
                and b.get("id") is not None
                and not b.get("repairing")
                and (str(b.get("type")).lower() in CRITICAL or cash >= 600)
                and self._fresh(f"repair:{b['id']}", 2.0)
            ):
                orders.append({"cmd": "repair", "building": b["id"], "on": True})

        # Reflex 4: raiders near the ore field pull the harvesters home. Dying harvesters are income and
        # tiebreak losses; an idle harvester walks back to work on its own once the raid is gone.
        fleeing: list[int] = []
        threat_cells = [c for c in (_cell(e) for e in enemies) if c]
        if threat_cells and self._fresh("harv_flee", 4.0):
            for u in units:
                if str(u.get("type")).lower() != "harv" or not isinstance(u.get("id"), int):
                    continue
                p = _cell(u)
                if p and any(math.dist(p, c) <= HARV_FLEE_RADIUS for c in threat_cells):
                    fleeing.append(u["id"])
            if fleeing and home:
                orders.append(
                    {"cmd": "move", "units": fleeing[:20], "to": [int(home[0]), int(home[1])]}
                )

        # Reflex 5: idle harvesters go back to work (a depleted patch or a retreat leaves them standing).
        idle_ids = set(legal.get("idle_units") or [])
        lazy = [
            u["id"]
            for u in units
            if str(u.get("type")).lower() == "harv"
            and isinstance(u.get("id"), int)
            and (u.get("idle") is True or u["id"] in idle_ids)
            and u["id"] not in fleeing
        ]
        lazy = [h for h in lazy if self._fresh(f"harvest:{h}", 8.0)]
        if lazy:
            orders.append({"cmd": "harvest", "units": lazy[:50]})

        # Rally points: fresh units roll toward the staging point, so reinforcements join the push instead
        # of idling at the gate. While holding (or defending) they rally home, which also clears a stale one.
        if legal.get("production_buildings") and self._fresh("rally", 5.0):
            staging = (
                self._staging_cell(
                    _cell(obs.get("base_center")),
                    self._wave_target(plan, raw, obs, legal, adv),
                    (raw.get("map") or {}).get("bounds"),
                )
                if (plan.army_mode in ("attack", "all_in") and not defend)
                else home
            )
            if staging:
                rows = {b.get("id"): b for b in buildings}
                for p in (legal.get("production_buildings") or [])[:4]:
                    b = rows.get(p.get("id"))
                    if not isinstance(b, dict) or str(p.get("type")).lower() not in PRODUCERS:
                        continue
                    cur = b.get("rally")
                    if (
                        isinstance(cur, (list, tuple))
                        and len(cur) == 2
                        and math.dist((cur[0], cur[1]), (staging[0], staging[1])) <= 1.5
                    ):
                        continue  # already pointing there
                    orders.append(
                        {
                            "cmd": "rally",
                            "building": p["id"],
                            "to": [int(staging[0]), int(staging[1])],
                        }
                    )

        idle_combat = [
            u for u in idle_combat if u not in self._scouts
        ]  # a scout picked this tick is not a fighter
        guard = (
            None
            if plan.army_mode == "all_in"
            else self._safe(
                "defend", lambda: self._defend(home, buildings, enemies, all_combat), None
            )
        )
        if guard is None:
            orders += self._army(
                plan,
                raw,
                obs,
                legal,
                units,
                adv,
                defend,
                forced_mode,
                idle_combat,
                army_value,
                enemy_seen,
                defence_seen,
            )
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
                jev = {
                    "age_s": round(now_adv - advice.at, 1),
                    "values": advice.values,
                    "confidence": advice.confidence,
                }
            self.strategist.maybe_update(
                raw,
                config,
                stuck,
                [m for _, m in self._refusals],
                jev=jev,
                extra={
                    "game_seconds": round(game_s),
                    "measured_income_per_s": round(income, 1),
                    "harvesters": owned.get("harv", 0),
                    "refineries": refineries,
                    "queue_eta_s": [
                        {"item": p.get("item"), "remaining_s": p.get("remaining_s")}
                        for p in production[:8]
                    ],
                    "reactions_fired": [r.name for r in fired],
                    "scout_alive": bool(self._scouts),
                    "anti_air_filtered_until_air_seen": sorted(AA_ONLY) if not air_seen else [],
                },
            )

        self._decisions += 1
        if now - self._state_at >= 5.0:
            rate = self._decisions / max(1e-6, now - self._state_at) if self._state_at else None
            self._state_at, self._decisions = now, 0
            trace.write(
                "state",
                tick=obs.get("tick"),
                cash=cash,
                ore=econ.get("ore"),
                ore_capacity=econ.get("resource_capacity"),
                power=[provided, drained],
                income=round(income, 1),
                time=raw.get("time"),
                units=dict(owned),
                queue=[p.get("item") for p in production],
                idle_combat=len(idle_combat),
                enemies=len(enemies),
                enemy_buildings=len(enemy_buildings),
                military=obs.get("military"),
                plan_version=self.strategist.version,
                jev=(advice.values if advice else None),
                reactions=[r.name for r in fired],
                burn=[round(burn_struct, 1), round(burn_units, 1)],
                budget=round(budget, 1),
                scout=sorted(self._scouts),
                decisions_per_s=round(rate, 2) if rate is not None else None,
                army_value=army_value,
                enemy_types=self.intel.enemy_types(),
                enemy_value=round(enemy_seen),
                enemy_defence=round(defence_seen),
                doctrine=self.doctrines.name,
                stance=directive.stance,
                roles={
                    "scout": len(self._scouts),
                    "garrison": len(self._guard),
                    "raider": len(self._flankers),
                    "strike": max(0, len(all_combat) - len(self._guard) - len(self._flankers)),
                },
            )
        if not orders:
            return WAIT
        batch = orders[:MAX_BATCH]
        trace.write(
            "orders",
            tick=obs.get("tick"),
            orders=[
                {k: v for k, v in o.items() if k != "units"}
                | ({"n": len(o["units"])} if "units" in o else {})
                for o in batch
            ],
        )
        self._last_sent = batch
        return WithReasoning(
            {"type": "orders", "orders": batch}, self._label(batch, guard is not None)
        )

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
            return "placing " + ", ".join(
                str(o.get("item")) for o in batch if o.get("cmd") == "place"
            )
        if "move" in cmds:
            return "moving"
        return str(next(iter(cmds), "orders"))

    # -- command -----------------------------------------------------------------

    def _command(
        self,
        game_s: float,
        army_n: int,
        *,
        enemies: list[dict[str, Any]],
        enemy_buildings: list[dict[str, Any]],
        near_n: int,
        mono: float,
    ) -> roles.Directive:
        """The General's directive for this decision; traced whenever the picture changes."""
        if enemies or enemy_buildings:
            self._enemy_seen_at = mono
        d = self.general.decide(
            game_s=game_s,
            army_n=army_n,
            doctrine=self.doctrines.name,
            enemy_types=self.intel.enemy_types(),
            enemy_towers=self.intel.enemy_tower_count(),
            raided_recently=mono - self._attack_at <= 90.0,
            enemy_near=near_n,
            blind_for_s=mono - self._enemy_seen_at,
        )
        old = self._directive
        if (d.stance, d.garrison, d.raiders, d.scouts) != (
            old.stance,
            old.garrison,
            old.raiders,
            old.scouts,
        ):
            trace.write(
                "directive",
                game_s=round(game_s),
                stance=d.stance,
                garrison=d.garrison,
                raiders=d.raiders,
                scouts=d.scouts,
                units=list(d.units),
                towers=list(d.towers),
                army=army_n,
                why=d.reason,
            )
        self._directive = d
        return d

    # -- scouting ----------------------------------------------------------------

    def _scout(
        self,
        units: list[dict[str, Any]],
        legal: dict[str, Any],
        avail: set[str],
        entries: dict[str, list[dict[str, Any]]],
        game_s: float,
        home: tuple[int, int] | None,
        cash: float,
        raw: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Keep one cheap fast unit sent toward the enemy so the model and Jev can see it coming.

        One scout at a time, replaced no sooner than ``SCOUT_RESEND_S`` after it is sent (instant re-sends fed
        the enemy five free kills, but never replacing it left whole matches blind). The General asks for a
        second when nothing of the enemy has been seen for a minute.
        """
        alive = {u["id"] for u in units if isinstance(u.get("id"), int)}
        self._scouts &= alive
        enemy = _cell(legal.get("enemy_base"))
        limit = SCOUT_MAX
        if self._directive.scouts >= 2 and cash >= SCOUT_SECOND_MIN_CASH:
            limit = 2  # the General: blind for a minute, send a second one
        now = self._clock()
        if (
            len(self._scouts) >= limit
            or game_s < SCOUT_AT_S
            or enemy is None
            or home is None
            or (self._scouts_sent and now - self._scout_at < SCOUT_RESEND_S)
        ):
            return []
        idle = set(legal.get("idle_units") or [])
        kinds = {u["id"]: str(u.get("type")).lower() for u in units if isinstance(u.get("id"), int)}
        for kind in SCOUT_PREF:
            for uid, k in kinds.items():
                if k == kind and uid in idle and self._fresh("scout", 40.0):
                    mid = self._clamp(
                        (
                            home[0] + (enemy[0] - home[0]) * 0.55,
                            home[1] + (enemy[1] - home[1]) * 0.55,
                        ),
                        (raw.get("map") or {}).get("bounds"),
                    )
                    far = self._clamp(enemy, (raw.get("map") or {}).get("bounds"))
                    self._scouts.add(uid)
                    self._scouts_sent += 1
                    self._scout_at = now
                    trace.write(
                        "scout", unit=uid, unit_type=kind, via=mid, to=far, game_s=round(game_s)
                    )
                    return [
                        {"cmd": "move", "units": [uid], "to": mid},
                        {"cmd": "move", "units": [uid], "to": far, "queued": True},
                    ]
        for kind in ("dog", "e1"):
            if (
                kind in avail
                and len(entries.get("infantry", [])) < 3
                and self._fresh("scout_train", 25.0)
            ):
                entries.setdefault("infantry", []).append(
                    {"item": kind, "ready": False, "progress": 0.0}
                )
                return [{"cmd": "train", "item": kind, "count": 1}]
        return []

    def _check_schema(
        self, obs: dict[str, Any], econ: dict[str, Any], legal: dict[str, Any]
    ) -> None:
        """Record the real observation's shape once, and say loudly if a field we rely on is missing."""
        missing = [f"observation.{k}" for k in EXPECTED["observation"] if k not in obs]
        missing += [f"economy.{k}" for k in EXPECTED["economy"] if k not in econ]
        missing += [
            f"legal_actions.{k}" for k in EXPECTED["legal_actions"] if legal and k not in legal
        ]
        unit = next((u for u in obs.get("units") or [] if isinstance(u, dict)), {})
        trace.write(
            "schema",
            observation=sorted(obs),
            economy=econ,
            unit_fields=sorted(unit),
            legal_actions=sorted(legal),
            missing=missing,
        )
        if missing:
            print(
                f"[redalert] observation is missing fields the agent relies on: {missing}",
                file=sys.stderr,
                flush=True,
            )

    def _choose_doctrine(self, plan: Plan, numbers: dict[str, Any], now: float) -> Plan:
        """Jev's battle-plan pick (fresh, allowed, confident) or the rules' pick, with hysteresis; then apply it."""
        pick = conf = answer_id = None
        advice = self.advisor.current() if self.advisor.enabled else None
        if advice is not None and "doctrine" in plan.jev_slots:
            if self.advisor.now() - advice.at <= ADVICE_TTL_S:
                pick, conf, answer_id = (
                    advice.values.get("doctrine"),
                    advice.confidence.get("doctrine"),
                    advice.at,
                )
        if pick in ("pressure", "all_in") and not doctrines.attack_allowed(pick, numbers):
            # Jev read "enemy barely seen" as "enemy is weak" and sent 8 riflemen all in at 1,300 vs 300 seen;
            # an attack pick needs a real estimate and a clear lead, else it is refused and the rules decide.
            pick = conf = answer_id = None
        rule = doctrines.fallback(numbers, self.doctrines.name)
        before = self.doctrines.name
        if not self.doctrines.since:
            self.doctrines.since = now  # the dwell clock starts when the war factory does
        if self.doctrines.update(now, pick, conf, rule, answer_id):
            trace.write(
                "doctrine",
                name=self.doctrines.name,
                was=before,
                source=self.doctrines.source,
                jev=pick,
                jev_confidence=conf,
                rules=rule,
                numbers=numbers,
            )
        return doctrines.apply(self.doctrines.doctrine, plan)

    def _battle_summary(
        self,
        obs: dict[str, Any],
        now: float,
        home: tuple[int, int] | None,
        legal: dict[str, Any],
        idle: list[int],
        numbers: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._safe(
            "intel",
            lambda: self.intel.summary(
                obs=obs,
                now=now,
                home=home,
                enemy_base=_cell(legal.get("enemy_base")),
                cost_of=self.cost_of,
                idle=set(idle),
            ),
            {},
        )
        out["key_numbers"] = numbers
        self._battle = out
        return out

    def _filler(
        self,
        q: str,
        avail: set[str],
        owned: dict[str, int],
        plan: Plan,
        air_seen: bool,
        vs_armor: bool,
    ) -> str | None:
        """What an idle barracks / war factory should train next when the plan's counts are already met."""
        ok = [
            i
            for i in avail
            if self.queue_of(i) == q and i not in NON_COMBAT and (i not in AA_ONLY or air_seen)
        ]
        if not ok:
            return None
        if q == "infantry":
            if vs_armor and "e3" in ok:
                return "e3"  # rockets against tanks
            mix = [i for i in ("e1", "e3") if i in ok]
            if mix:  # roughly two riflemen per rocket soldier: cheap bodies plus anti-armour
                return (
                    "e3" if "e3" in mix and owned.get("e3", 0) * 2 < owned.get("e1", 0) else mix[0]
                )
        for item, _ in plan.units:  # the plan's own unit choices first, in its order
            if item in ok:
                return item
        for item in FILLER_VEHICLES:
            if item in ok:
                return item
        return None

    # -- army --------------------------------------------------------------------

    def _defend(
        self,
        home: tuple[int, int] | None,
        buildings: list[dict[str, Any]],
        enemies: list[dict[str, Any]],
        army: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Reflex: enemy units in the base pull every fighter onto them, re-aimed every 1.5 s.

        ``None`` when the base is clear (the plan's army mode applies); a list (maybe empty while the
        re-aim timer runs) while it is not, so nothing else moves the army away mid-fight.
        """
        if home is None or not army:
            return None
        spots = [c for c in (_cell(b) for b in buildings) if c]
        raiders = [
            (c, str(e.get("type")).lower())
            for e, c in ((e, _cell(e)) for e in enemies)
            if c
            and str(e.get("type")).lower() not in NOT_A_THREAT
            and (math.dist(c, home) <= DEFEND_RADIUS or any(math.dist(c, s) <= 10 for s in spots))
        ]
        intruders = [c for c, _ in raiders]
        if not intruders:
            return None
        # A raid of one or two (a scout, a lone harasser) is handled by whoever is home; a push recalls everyone.
        fighters = [
            u["id"]
            for u in army
            if u["id"] not in self._scouts
            and (
                len(intruders) > 2
                or (_cell(u) is not None and math.dist(_cell(u), home) <= 2 * DEFEND_RADIUS)
            )
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
            # A push that matches or outweighs the army is fought AT HOME, together, under the towers and the
            # repair crews: chasing the nearest intruder walked 12 riflemen into 14 enemies one group at a time
            # and lost the base (the enemy keeps its army together; so must we).
            mine = sum(
                self.cost_of(str(u.get("type")).lower()) for u in army if u["id"] in set(fighters)
            )
            theirs = sum(self.cost_of(t) for _, t in raiders)
            if len(intruders) > 2 and theirs >= DEFEND_MASS_RATIO * mine:
                target = home
            self._gathered.clear()  # after the fight, hold mode regroups everyone
            for i in range(0, min(len(fighters), 150), 50):
                out.append(
                    {
                        "cmd": "attack_move",
                        "units": fighters[i : i + 50],
                        "to": [int(target[0]), int(target[1])],
                    }
                )
            trace.write("defend", intruders=len(intruders), fighters=len(fighters), to=list(target))
        return out

    def _target(
        self, target: Any, raw: dict[str, Any], obs: dict[str, Any], legal: dict[str, Any]
    ) -> list[int] | None:
        bounds = (raw.get("map") or {}).get("bounds")
        base = _cell(legal.get("enemy_base"))
        cell: tuple[int, int] | None
        if isinstance(target, tuple):
            cell = target
        elif target == "building":
            mine = _cell(obs.get("base_center")) or (0, 0)
            seen = [
                c
                for c in (
                    _cell(e)
                    for e in (obs.get("visible_enemy_buildings") or [])
                    if isinstance(e, dict) and not e.get("wreck")
                )
                if c
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

    def _prune_detachments(self, alive: set[int]) -> set[int]:
        """Dead or expired members rejoin the main force; survivors of a finished flank run walk back."""
        self._guard &= alive
        now = self._clock()
        expired: set[int] = set()
        for uid, run in list(self._flankers.items()):
            if uid not in alive or now >= run[1]:
                del self._flankers[uid]
                if uid in alive:
                    expired.add(uid)
        return expired

    def _front_cell(
        self,
        home_c: tuple[int, int] | None,
        enemy_c: tuple[int, int] | None,
        bounds: Any,
        tiles: float,
    ) -> list[int] | None:
        if not home_c or not enemy_c:
            return None
        d = math.dist(home_c, enemy_c)
        step = min(tiles / d, 0.35) if d > 1 else 0.0
        return self._clamp(
            [
                int(home_c[0] + step * (enemy_c[0] - home_c[0])),
                int(home_c[1] + step * (enemy_c[1] - home_c[1])),
            ],
            bounds,
        )

    def _staging_cell(
        self,
        home_c: tuple[int, int] | None,
        target: list[int] | tuple[int, int] | None,
        bounds: Any,
    ) -> list[int] | None:
        """The muster/rally point: a step toward the enemy that stays inside forward radius."""
        if not home_c or not target:
            return None
        step = min(0.2, (FORWARD_RADIUS - 8) / max(1.0, math.dist(home_c, target)))
        return self._clamp(
            [
                int(home_c[0] + step * (target[0] - home_c[0])),
                int(home_c[1] + step * (target[1] - home_c[1])),
            ],
            bounds,
        )

    def _guard_orders(
        self,
        out: list[dict[str, Any]],
        idle: list[int],
        units: list[dict[str, Any]],
        mode: str,
        defend: bool,
        bounds: Any,
        home_c: tuple[int, int] | None,
        enemy_c: tuple[int, int] | None,
    ) -> list[int]:
        """The defence line in front of the base: sticky holders on `defend` stance - they engage in range
        and never chase, so the army is never entirely away attacking (or entirely home waiting)."""
        now = self._clock()
        n_combat = sum(
            1
            for u in units
            if isinstance(u.get("id"), int) and str(u.get("type")).lower() not in NON_COMBAT
        )
        d = self._directive
        # The General sizes the home line (a quarter of the army, half when threatened, never under four after
        # the opening); an all-in or a tiny army keeps none. A defend posture holds at least the old floor.
        want = 0 if (mode == "all_in" or n_combat < 5) else d.garrison
        if want == 0:
            self._guard.clear()
            return idle
        type_of = {
            u.get("id"): str(u.get("type")).lower() for u in units if isinstance(u.get("id"), int)
        }
        idle_set = set(idle)
        pool = [
            roles.Soldier(u, type_of.get(u, ""), self.cost_of(type_of.get(u, "")))
            for u in (idle_set | self._guard)
            if u not in self._flankers
        ]
        # The roster gives every soldier one role: the garrison takes the types that counter the enemy, then the
        # cheapest; whoever is already on the line stays unless the quota shrank.
        sized = dataclasses.replace(d, garrison=want, raiders=0)
        assigned = roles.assign(
            pool,
            {u: roles.Role.GARRISON for u in self._guard},
            sized,
            keep_scouts=set(),
            preferred=d.units,
        )
        self._guard = {u for u, r in assigned.items() if r == roles.Role.GARRISON}
        guards = [u for u in self._guard if u in set(idle)]
        if not guards:
            return [u for u in idle if u not in self._guard]
        recruit = [u for u in guards if u not in self._guard_stanced]
        if recruit:
            self._guard_stanced.update(recruit)
            out.append({"cmd": "set_stance", "units": recruit[:20], "stance": GUARD_STANCE})
        line = self._front_cell(home_c, enemy_c, bounds, GUARD_FRONT_TILES)
        pos = {u.get("id"): _cell(u) for u in units}
        away = [
            u
            for u in guards
            if line and (pos.get(u) is None or math.dist(pos[u], line) > GUARD_RADIUS)
        ]
        if away and line and self._fresh("guard_move", 4.0):
            out.append({"cmd": "move", "units": away[:50], "to": line})
        return [u for u in idle if u not in self._guard]

    def _flank_orders(
        self,
        out: list[dict[str, Any]],
        idle: list[int],
        raw: dict[str, Any],
        home_c: tuple[int, int] | None,
        target: list[int] | None,
        bounds: Any,
        plan: Plan,
        adv: Callable[[str], Any],
    ) -> list[int]:
        """Small packs that run wide around the fight and come at the target from the sides; each run
        alternates sides, so attacks arrive from different directions instead of one file. The pack runs on
        `defend` stance - fighting whatever it meets on the arc, then digging in at the point - and joins the
        main wave the moment that launches, so the flank and the push are one plan, not two solo fights."""
        now = self._clock()
        idle = [u for u in idle if u not in self._flankers]
        style = adv("style")
        cd = STYLE_FLANK_CD_S.get(style if isinstance(style, str) else "", FLANK_CD_S)
        game_s = (raw.get("time") or {}).get("game_seconds")
        if (
            plan.army_mode == "all_in"
            or not target
            or not home_c
            or not isinstance(game_s, (int, float))
            or game_s < FLANK_FROM_S
            or len(idle) < FLANK_MIN_ARMY
            or now - self._flank_at < cd
            or self._directive.raiders < 2
        ):
            return idle
        type_of = {
            u.get("id"): str(u.get("type")).lower()
            for u in raw.get("observation", {}).get("units", [])
            if isinstance(u, dict) and isinstance(u.get("id"), int)
        }
        pack = sorted(idle, key=lambda u: self.cost_of(type_of.get(u, "")))[
            : self._directive.raiders
        ]
        if len(pack) < 2:
            return idle
        self._flank_at = now
        self._flank_side = -self._flank_side
        d = math.dist(home_c, target) or 1.0
        px, py = -(target[1] - home_c[1]) / d, (target[0] - home_c[0]) / d
        wp = self._clamp(
            [
                int(target[0] + px * self._flank_side * FLANK_OFFSET),
                int(target[1] + py * self._flank_side * FLANK_OFFSET),
            ],
            bounds,
        )
        if wp is None:
            return idle
        self._flankers.update({u: (wp, now + FLANK_LIFE_S) for u in pack})
        out.append({"cmd": "set_stance", "units": pack, "stance": GUARD_STANCE})
        out.append({"cmd": "attack_move", "units": pack, "to": wp})
        return [u for u in idle if u not in self._flankers]

    def _wave_target(
        self,
        plan: Plan,
        raw: dict[str, Any],
        obs: dict[str, Any],
        legal: dict[str, Any],
        adv: Callable[[str], Any],
    ) -> list[int] | None:
        """The objective the army fights toward: the plan's, or Jev's override - shared by waves and rallies."""
        name: Any = plan.target
        choice = adv("target")
        if isinstance(name, str) and choice in TARGET_NAMES:
            name = choice
        return self._target(name, raw, obs, legal)

    def _army(
        self,
        plan: Plan,
        raw: dict[str, Any],
        obs: dict[str, Any],
        legal: dict[str, Any],
        units: list[dict[str, Any]],
        adv: Callable[[str], Any] = lambda slot: None,
        defend: bool = False,
        forced_mode: str | None = None,
        idle_combat: list[int] | None = None,
        army_value: float = 0.0,
        enemy_seen: float = 0.0,
        enemy_defence: float = 0.0,
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
        now0 = self._clock()
        if mode == "all_in":
            self._mode_cur, self._mode_at = "all_in", now0
        else:
            if forced_mode:
                mode = forced_mode
            else:
                posture = adv("posture")
                if defend or posture == "retreat":
                    mode = "retreat"
                elif posture in ("hold", "attack"):
                    mode = posture
            if (
                mode == "hold"
                and army_value >= ATTACK_VALUE
                and army_value >= 1.5 * enemy_seen
                and forced_mode is None
                and not defend
            ):
                mode = "attack"  # a big army sitting at home is wasted: one match was lost with 8,800 of it idle
            # One goal at a time: a posture flap must not pull half the army home mid-fight. Emergencies
            # (a reaction, defend, retreat) land at once, and recovery from a retreat is instant.
            if mode == "retreat":
                self._mode_cur, self._mode_at = "retreat", now0 - MODE_HOLD_S
            elif mode != self._mode_cur:
                if now0 - self._mode_at < MODE_HOLD_S and forced_mode is None and not defend:
                    mode = self._mode_cur
                else:
                    self._mode_cur, self._mode_at = mode, now0

        bounds = (raw.get("map") or {}).get("bounds")
        home_cell = plan.gather or _cell(obs.get("base_center"))
        home = self._clamp(home_cell, bounds)
        fighters = [
            u["id"]
            for u in units
            if isinstance(u.get("id"), int)
            and str(u.get("type")).lower() not in NON_COMBAT
            and u["id"] not in self._scouts
        ]
        expired = self._prune_detachments(set(fighters))

        if mode == "retreat":
            if home and fighters and self._fresh("retreat", 5.0):
                for i in range(0, min(len(fighters), 150), 50):
                    out.append({"cmd": "move", "units": fighters[i : i + 50], "to": home})
            return out

        enemy_c = _cell(legal.get("enemy_base"))
        home_c0 = _cell(obs.get("base_center"))
        # Defence faces the actual threat: raiders near home point the line at themselves, not at their base.
        near_threat = [
            c
            for c in (_cell(e) for e in (obs.get("visible_enemies") or []))
            if c and home_c0 and math.dist(c, home_c0) <= THREAT_RADIUS
        ]
        threat_c = None
        if near_threat:
            threat_c = (
                int(sum(c[0] for c in near_threat) / len(near_threat)),
                int(sum(c[1] for c in near_threat) / len(near_threat)),
            )
        defence_toward = threat_c or enemy_c
        if mode == "hold":
            # The defence line holds the front; the rest of the army gathers at home. Flankers keep working.
            idle_combat = self._guard_orders(
                out, idle_combat, units, mode, defend, bounds, home_c0, defence_toward
            )
            if home:
                pos = {u.get("id"): _cell(u) for u in units}
                far = [
                    u
                    for u in idle_combat
                    if u not in self._flankers
                    and u not in self._gathered
                    and (pos.get(u) is None or math.dist(pos[u], home) > 8)
                ]
                if far:
                    self._gathered.update(far[:50])
                    out.append({"cmd": "move", "units": far[:50], "to": home})
            return out

        target = self._wave_target(plan, raw, obs, legal, adv)
        if target is None:
            return out
        if mode == "all_in":
            if fighters and self._fresh("all_in", TTL["burst"]):
                for i in range(0, min(len(fighters), 150), 50):
                    out.append({"cmd": "attack_move", "units": fighters[i : i + 50], "to": target})
            return out

        # Fighters already out in the field (a wave that took its target) keep going without waiting for a wave —
        # but a lone straggler regroups at the staging point instead of trickling into the enemy's guns.
        pos = {u.get("id"): _cell(u) for u in units}
        home_c = home_c0
        idle_combat = self._guard_orders(
            out, idle_combat, units, mode, defend, bounds, home_c, defence_toward
        )
        idle_combat = self._flank_orders(out, idle_combat, raw, home_c, target, bounds, plan, adv)
        if expired:
            idle_combat = [u for u in idle_combat if u not in expired]
            back = self._staging_cell(home_c, target, bounds) or home
            if self._fresh("flank_recall", 5.0):
                out.append({"cmd": "move", "units": sorted(expired)[:50], "to": back})
        forward = [
            u
            for u in idle_combat
            if u not in self._flankers
            and home_c
            and pos.get(u)
            and math.dist(pos[u], home_c) > FORWARD_RADIUS
        ]
        if forward:
            if len(forward) >= FORWARD_MIN:
                if self._fresh("forward", 2.0):
                    out.append({"cmd": "attack_move", "units": forward[:50], "to": target})
            elif self._fresh("regroup", 4.0):
                back = self._staging_cell(home_c, target, bounds) or home
                out.append({"cmd": "move", "units": forward[:50], "to": back})
        idle_combat = [u for u in idle_combat if u not in forward]
        n = len(idle_combat)
        now = self._clock()
        remaining = (raw.get("time") or {}).get("remaining_seconds")
        opposition = (
            enemy_seen + enemy_defence
        )  # towers are the army you have to beat to take a base
        ahead = opposition >= FINISH_MIN_SEEN and army_value >= FINISH_AHEAD_RATIO * opposition
        behind = opposition >= FEED_MIN_SEEN and army_value < FEED_BEHIND_RATIO * opposition
        wave_ok = adv("wave")
        if ahead:
            go = (
                n >= plan.wave_min
            )  # clearly stronger: finish them; a "wait" answer cannot hold the wave home
        elif isinstance(wave_ok, float):
            # Jev judges the moment, but a wave is the whole idle army: approval never halves it.
            go = n >= plan.wave_min and (wave_ok >= WAVE_YES or n >= max(20, 2 * plan.wave_min))
        else:
            go = n >= plan.wave_min
        if not go and army_value >= ATTACK_VALUE and n >= plan.wave_min:
            go = True  # a big army must not sit home all match - but it attacks as a whole wave, never a piece
        if behind and not (isinstance(remaining, (int, float)) and remaining <= ENDGAME_PUSH_S):
            go = False  # feeding a clearly stronger enemy loses the tiebreak: assemble instead (the closing
            # push still fires inside the last five minutes when the trades are negative anyway)
        if now < self._wave_until:
            go = False  # a wave just left: let the next one assemble instead of dribbling out
        if go:
            self._gathered.difference_update(idle_combat)
            self._wave_until = now + (AHEAD_COOLDOWN_S if ahead else WAVE_COOLDOWN_S)
            out.append({"cmd": "attack_move", "units": idle_combat[:50], "to": target})
            raiders = list(self._flankers)
            if (
                raiders
            ):  # the anvil joins the hammer: the flank and the push hit the target together
                self._flankers.clear()
                out.append(
                    {"cmd": "set_stance", "units": raiders[:50], "stance": "attack_anything"}
                )
                out.append({"cmd": "attack_move", "units": raiders[:50], "to": target})
        elif idle_combat and self._fresh("muster", 4.0):
            # Waiting for the next wave: mass at a staging point toward the enemy so the strike leaves together.
            muster = self._staging_cell(home_c, target, bounds) or home
            away = [
                u
                for u in idle_combat
                if pos.get(u) is None or math.dist(pos[u], muster) > WAVE_MUSTER_RADIUS
            ]
            if away:
                out.append({"cmd": "move", "units": away[:50], "to": muster})
        return out

    # -- the model's one-off raw orders ------------------------------------------

    def _raw_orders(
        self,
        plan: Plan,
        raw: dict[str, Any],
        obs: dict[str, Any],
        units: list[dict[str, Any]],
        buildings: list[dict[str, Any]],
        avail: set[str],
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
        enemy_ids = {
            e["id"] for e in enemies if isinstance(e, dict) and isinstance(e.get("id"), int)
        }
        core_ids = {
            b["id"]
            for b in buildings
            if str(b.get("type")).lower() in CORE_STRUCTURES and isinstance(b.get("id"), int)
        }
        clean = (
            sanitize_order(o, unit_ids, building_ids, enemy_ids, bounds_t, avail)
            for o in plan.orders
        )  # type: ignore[arg-type]
        return [
            o for o in clean if o and not (o["cmd"] == "sell" and o.get("building") in core_ids)
        ]


def create_agent():
    return RedAlertAgent()
