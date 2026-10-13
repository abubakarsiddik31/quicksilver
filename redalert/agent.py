"""Real-time Red Alert player: an LLM strategist plus a fast plan executor.

Red Alert never waits, so the two halves run at different speeds:

* ``redalert.strategist.Strategist`` consults an LLM in a background thread,
  woken by game events (see that module), and publishes a *plan*
  (``redalert.plan.Plan``): ordered structure and unit wants, army posture, and
  optional one-off raw orders. Every strategic decision comes from the model;
  with no model the built-in default opening runs instead.
* ``RedAlertAgent`` (here) is called ~5 times a second and executes the current
  plan against the latest state in milliseconds. It also owns the *reflexes*
  that must not wait for a model: deploy the MCV, place finished buildings,
  repair, and an emergency power plant when the base is really in low power.

A third part, ``redalert.tactics.TacticalAdvisor`` (TypeSafe Jev), answers
bounded tactical questions about the live state about once a second: the next
structure or unit, attack/hold/retreat, wave timing, target, threat level, and
whether more harvesters or power are needed. Its answers are used only when
fresh, allowed by the plan and confident enough; the LLM decides which slots Jev
may drive and is woken when Jev is unsure or sees a serious threat. Without a
Jev key those slots simply fall back to the plan and the rules.

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
from redalert.plan import DEFAULT_PLAN, Plan, sanitize_order
from redalert import trace
from redalert.strategist import Strategist
from redalert.tactics import ESCALATE_LEVEL, Candidates, TacticalAdvisor

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
ALLIED_FACTIONS = {"allies", "allied", "england", "france", "germany"}
SOVIET_FACTIONS = {"soviet", "soviets", "russia", "ukraine"}
# Fallback costs when the rules table is not available.
FALLBACK_COST = {"powr": 300, "tent": 500, "barr": 500, "proc": 1400, "weap": 2000, "harv": 1100, "e1": 100}

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


def _cell(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
        return int(value[0]), int(value[1])
    if isinstance(value, dict) and isinstance(value.get("x"), (int, float)) and isinstance(value.get("y"), (int, float)):
        return int(value["x"]), int(value["y"])
    return None


class RedAlertAgent:
    """Executes the strategist's plan at real-time speed."""

    def __init__(self, strategist: Strategist | None = None, advisor: TacticalAdvisor | None = None) -> None:
        self.advisor = advisor if advisor is not None else TacticalAdvisor(
            JevClient.from_env(), telemetry_path=os.environ.get("RA_TELEMETRY") or None
        )
        self.strategist = strategist if strategist is not None else Strategist(GeminiClient.from_env())
        if not self.strategist.enabled:
            print("[redalert] no GEMINI_API_KEY: playing the fixed default plan, no AI strategy", file=sys.stderr, flush=True)
        self._until: dict[str, float] = {}
        self._stanced: set[int] = set()
        self._gathered: set[int] = set()
        self._exec_version = -1
        self._diag_since: dict[str, float] = {}
        self._refusals: list[tuple[float, str]] = []
        self._last_sent: list[dict[str, Any]] = []
        self._rules_for: int | None = None
        self._cost: dict[str, int] = {}
        self._queue: dict[str, str] = {}
        self._needs: dict[str, list[str]] = {}
        self._errors_logged: set[str] = set()
        self._state_at = 0.0

    # -- throttling and learning -------------------------------------------------

    def _fresh(self, key: str, ttl: float) -> bool:
        """True (and arms the cooldown) unless ``key`` was sent or refused too recently."""
        now = time.monotonic()
        if now < self._until.get(key, 0.0):
            return False
        self._until[key] = now + ttl
        return True

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

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """Learn which orders were refused: back off them and tell the strategist why."""
        sent = self._last_sent
        now = time.monotonic()
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

    def _reachable(self, item: str, avail: set[str], owned: dict[str, int], wanted: set[str]) -> bool:
        """Could ``item`` become buildable soon? (available, or its prerequisites are owned or wanted)."""
        if item in avail:
            return True
        needs = self._needs.get(item)
        if needs is None:
            return item in STRUCTURES  # no rules: assume a known structure is reachable
        return all(owned.get(n) or n in wanted for n in needs)

    # -- the fast loop -----------------------------------------------------------

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

        plan = self.strategist.current() if self.strategist.enabled else DEFAULT_PLAN
        legal = raw.get("legal_actions") or {}
        production = [p for p in (obs.get("production") or []) if isinstance(p, dict)]
        avail = {str(x).lower() for x in (obs.get("available_production") or [])}
        econ = obs.get("economy") or {}
        cash = econ.get("cash", 0) or 0
        faction = str(obs.get("faction") or "").lower()
        units = [u for u in (obs.get("units") or []) if isinstance(u, dict) and not u.get("wreck")]
        buildings = [b for b in (obs.get("buildings") or []) if isinstance(b, dict)]
        owned: dict[str, int] = {}
        for row in (*units, *buildings):
            t = str(row.get("type")).lower()
            owned[t] = owned.get(t, 0) + 1
        queued: dict[str, int] = {}
        depth: dict[str, int] = {}
        busy: set[str] = set()
        for p in production:
            item = str(p.get("item")).lower()
            q = str(p.get("queue") or self.queue_of(item)).lower()
            queued[item] = queued.get(item, 0) + 1
            depth[q] = depth.get(q, 0) + 1
            if q in ("building", "defense"):
                busy.add(q)  # building, or finished and waiting to be placed

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

        # Jev's fast tactical advice: used only if fresh, allowed by the plan and confident enough.
        wants = [(self._barracks(i, faction, avail, owned), c) for i, c in plan.buildings]
        unit_wants = list(plan.units)
        idle_combat = [u for u in (legal.get("combat_units") or []) if isinstance(u, int)]
        all_combat = [
            u for u in units if isinstance(u.get("id"), int) and str(u.get("type")).lower() not in NON_COMBAT
        ]

        def short(item: str, count: int) -> bool:
            return owned.get(item, 0) + queued.get(item, 0) < count

        struct_cands = tuple(dict.fromkeys(
            i for i, c in wants if short(i, c) and i in avail and self.queue_of(i) not in busy
        ))
        unit_cands = tuple(dict.fromkeys(
            i for i, c in unit_wants
            if short(i, c) and i in avail and self.cost_of(i) <= cash
            and depth.get(self.queue_of(i), 0) < (3 if self.queue_of(i) == "infantry" else 2)
        ))
        advice = None
        if self.advisor.enabled:
            self.advisor.maybe_refresh(raw, plan, Candidates(
                structures=struct_cands, units=unit_cands, idle_combat=len(idle_combat),
                all_combat=len(all_combat), has_harvester_item="harv" in avail, has_power_item="powr" in avail,
                describe={i: f"cost {self.cost_of(i)}" for i in (*struct_cands, *unit_cands)},
            ))
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
        if isinstance(need, float) and need >= YES and "harv" in avail and harvs < MAX_HARV:
            unit_wants.insert(0, ("harv", harvs + 1))
        need = adv("power")
        powrs = owned.get("powr", 0) + queued.get("powr", 0)
        if isinstance(need, float) and need >= YES and "powr" in avail and powrs < MAX_POWR:
            wants.insert(0, ("powr", powrs + 1))
        threat = adv("threat")
        defend = isinstance(threat, int) and threat >= ESCALATE_LEVEL
        if defend:
            tower = next((d for d in DEFENSE_ORDER if d in avail), None)
            have_d = sum(owned.get(d, 0) + queued.get(d, 0) for d in DEFENSE_ORDER)
            if tower and have_d < MAX_DEFENSES:
                wants.insert(0, (tower, owned.get(tower, 0) + queued.get(tower, 0) + 1))

        # Structures: the wants in priority order, plus an emergency power plant.
        provided, drained = econ.get("power_provided", 0) or 0, econ.get("power_drained", 0) or 0
        if drained > provided and "powr" in avail:
            wants.insert(0, ("powr", owned.get("powr", 0) + queued.get("powr", 0) + 1))
        wanted_names = {i for i, _ in wants}
        unmet: list[str] = []  # unsatisfied structure wants, for the cash reserve
        for item, count in wants:
            have = owned.get(item, 0) + queued.get(item, 0)
            if have >= count:
                continue
            if self._reachable(item, avail, owned, wanted_names) and len(unmet) < 3:
                unmet.append(item)
            if item not in avail:
                diag.append(f"wanted {item} x{count} is not in available_production (prerequisites?)")
                continue
            q = self.queue_of(item)
            if q in busy:
                continue
            if self._fresh(f"build:{item}", TTL["build"]):
                orders.append({"cmd": "build", "item": item})
                busy.add(q)

        # Units: the plan's wants in priority order, one new order per queue per decision.
        reserve = 0 if plan.army_first else sum(self.cost_of(i) for i in unmet)
        spendable = cash - reserve
        used: set[str] = set()
        trained = False
        for item, count in unit_wants:
            have = owned.get(item, 0) + queued.get(item, 0)
            if have >= count:
                continue
            if item not in avail:
                diag.append(f"wanted unit {item} x{count} is not in available_production")
                continue
            q = self.queue_of(item)
            if q in used or depth.get(q, 0) >= (3 if q == "infantry" else 2):
                continue
            n = min(count - have, 3) if q == "infantry" else 1
            if spendable < self.cost_of(item) * n or not self._fresh(f"train:{item}", TTL["train"]):
                continue
            orders.append({"cmd": "train", "item": item, "count": n})
            used.add(q)
            spendable -= self.cost_of(item) * n
            trained = True

        if not unmet and not trained and not production and cash > 1500:
            diag.append(f"{cash} cash unspent and nothing is queued or wanted")
        if not diag and not unmet and not production and all(
            owned.get(i, 0) + queued.get(i, 0) >= c for i, c in (*plan.buildings, *plan.units)
        ):
            diag.append("every structure and unit want in the plan is satisfied")

        # Reflex 3: repairs.
        for b in buildings:
            if (
                (b.get("hp", 100) or 100) < 75 and b.get("id") is not None and not b.get("repairing")
                and self._fresh(f"repair:{b['id']}", 2.0)
            ):
                orders.append({"cmd": "repair", "building": b["id"], "on": True})

        orders += self._army(plan, raw, obs, legal, units, adv, defend)

        now = time.monotonic()
        self._diag_since = {d: self._diag_since.get(d, now) for d in diag}
        stuck = [d for d, t in self._diag_since.items() if now - t >= DIAG_AFTER_S]
        self._refusals = [(t, m) for t, m in self._refusals if now - t < REFUSAL_KEEP_S]
        if self.strategist.enabled:
            for reason in self.advisor.pop_escalations():
                self.strategist.notify(reason)
            jev = None
            if advice is not None:
                jev = {"age_s": round(now_adv - advice.at, 1), "values": advice.values, "confidence": advice.confidence}
            self.strategist.maybe_update(raw, config, stuck, [m for _, m in self._refusals], jev=jev)

        if now - self._state_at >= 5.0:
            self._state_at = now
            trace.write(
                "state", tick=obs.get("tick"), cash=cash, power=[provided, drained],
                time=raw.get("time"), units={k: v for k, v in owned.items()},
                queue=[p.get("item") for p in production], idle_combat=len(idle_combat),
                enemies=len(obs.get("visible_enemies") or []),
                enemy_buildings=len(obs.get("visible_enemy_buildings") or []),
                military=obs.get("military"), plan_version=self.strategist.version,
                jev=(advice.values if advice else None),
            )
        if not orders:
            return WAIT
        batch = orders[:MAX_BATCH]
        trace.write("orders", tick=obs.get("tick"), orders=[
            {k: v for k, v in o.items() if k != "units"} | ({"n": len(o["units"])} if "units" in o else {})
            for o in batch
        ])
        self._last_sent = batch
        summary = f"v{self.strategist.version} {plan.assessment}"[:480] if self.strategist.enabled else "default plan"
        return WithReasoning({"type": "orders", "orders": batch}, summary)

    # -- army --------------------------------------------------------------------

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
    def _clamp(cell: tuple[int, int] | None, bounds: Any) -> list[int] | None:
        if cell is None:
            return None
        if isinstance(bounds, list) and len(bounds) == 4:
            bx, by, bw, bh = bounds
            cell = (max(bx, min(bx + bw - 1, cell[0])), max(by, min(by + bh - 1, cell[1])))
        return [cell[0], cell[1]]

    def _army(
        self, plan: Plan, raw: dict[str, Any], obs: dict[str, Any], legal: dict[str, Any],
        units: list[dict[str, Any]], adv: Callable[[str], Any] = lambda slot: None, defend: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        idle_combat = [u for u in (legal.get("combat_units") or []) if isinstance(u, int)]
        fresh = [u for u in idle_combat if u not in self._stanced]
        if fresh:
            self._stanced.update(fresh[:20])
            out.append({"cmd": "set_stance", "units": fresh[:20], "stance": "attack_anything"})

        # Jev may move the army between hold / attack / retreat, never out of a committed all-in.
        mode = plan.army_mode
        if mode != "all_in":
            posture = adv("posture")
            if defend or posture == "retreat":
                mode = "retreat"
            elif posture in ("hold", "attack"):
                mode = posture

        bounds = (raw.get("map") or {}).get("bounds")
        home_cell = plan.gather or _cell(obs.get("base_center"))
        home = self._clamp(home_cell, bounds)

        if mode == "retreat":
            everyone = [u["id"] for u in units if isinstance(u.get("id"), int)
                        and str(u.get("type")).lower() not in NON_COMBAT]
            if home and everyone and self._fresh("retreat", 5.0):
                for i in range(0, min(len(everyone), 150), 50):
                    out.append({"cmd": "move", "units": everyone[i:i + 50], "to": home})
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
            everyone = [
                u["id"] for u in units
                if isinstance(u.get("id"), int) and str(u.get("type")).lower() not in NON_COMBAT
            ]
            if everyone and self._fresh("all_in", TTL["burst"]):
                for i in range(0, min(len(everyone), 150), 50):
                    out.append({"cmd": "attack_move", "units": everyone[i:i + 50], "to": target})
            return out

        n = len(idle_combat)
        wave_ok = adv("wave")
        if isinstance(wave_ok, float):
            # Jev judges the moment, but never sends a token force or sits on a huge one forever.
            go = n >= max(3, plan.wave_min // 2) and (wave_ok >= WAVE_YES or n >= max(20, 2 * plan.wave_min))
        else:
            go = n >= plan.wave_min
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
        core_ids = {b["id"] for b in buildings if str(b.get("type")).lower() in CORE_STRUCTURES and isinstance(b.get("id"), int)}
        clean = (sanitize_order(o, unit_ids, building_ids, enemy_ids, bounds_t, avail) for o in plan.orders)  # type: ignore[arg-type]
        return [o for o in clean if o and not (o["cmd"] == "sell" and o.get("building") in core_ids)]


def create_agent():
    return RedAlertAgent()
