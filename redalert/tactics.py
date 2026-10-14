"""Jev tactical advisor: fast, typed, calibrated decisions for Red Alert.

The LLM strategist (:mod:`redalert.strategist`) is slow and writes the plan.
Jev is fast (~0.5 s) and answers *bounded* questions about the live state:
which structure or unit next, attack/hold/retreat, is a wave worth sending, how
big is the threat, do we need more harvesters or power. All questions go out in
one request every ~second, in the background; the 5 Hz executor reads the
newest answer from memory and never waits.

Safety, in layers:

- Jev only ever picks among options the harness itself built from the legal
  state, and the executor re-checks the pick against the *current* state.
- An answer is used only if it is fresh (``ADVICE_TTL_S``), its slot is allowed
  by the plan, and its confidence reaches the plan's ``min_confidence``.
  Otherwise that slot silently falls back to the plan / the rules.
- The LLM stays in charge: it picks which slots Jev may drive, writes the brief
  Jev sees, and is woken (:meth:`pop_escalations`) when Jev is unsure or sees a
  serious threat. A committed ``all_in`` plan is never overridden.
- Only engine strings and numbers go to Jev, never player-controlled text.

The request carries the *full* battle state (Jev bills input tokens only, so a
rich state is free) and up to the API's ten questions: build/train picks, the
battle plan, the money's stage, the threat, wave timing and target, which unit
type should dominate the army, and harvester/power need. Per-option
``probabilities`` ride along to the executor, which soft-ranks its wants by
them instead of obeying a single binary pick.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from common.jev import Answer, JevClient
from redalert import trace
from redalert.doctrine import DOCTRINES
from redalert.intel import NON_COMBAT
from redalert.plan import Plan

REFRESH_S = 0.6
REFRESH_URGENT_S = 0.4
ADVICE_TTL_S = 3.0
MAX_CALLS = 4000
BACKOFF_FIRST_S, BACKOFF_MAX_S = 2.0, 20.0
UNCERTAIN_COOLDOWN_S = 20.0
THREAT_COOLDOWN_S = 8.0
STYLE_COOLDOWN_S = (
    60.0  # re-reading the enemy's style may change the whole plan: tell the general once
)
THREAT_LEVELS = (
    "none: nothing hostile seen near my base or harvesters",
    "scouting: a few enemy units seen, no real attack",
    "raid: a small attack on my base or harvesters",
    "army: a sizeable enemy army is attacking or about to",
    "assault: an overwhelming attack on my base",
)
ESCALATE_LEVEL = 3  # "army" and above
POSTURES = {
    "attack": "send the army to attack the enemy now",
    "hold": "keep the army at home to defend and keep building up",
    "retreat": "pull the army back to base: the fight is being lost",
}
STAGES = {
    "expand_economy": "keep investing in harvesters, refineries and tech: no danger yet",
    "build_army": "stop expanding and put money into combat units now",
    "defend_now": "an attack is coming or here: defences and units before anything else",
}
TARGETS = {
    "nearest": "the nearest enemy thing we can see (or the enemy base if none)",
    "building": "the nearest enemy structure",
    "base": "straight at the enemy start position",
}
URGENT_EVENTS = {"under_attack", "building_lost", "unit_lost"}


@dataclass(frozen=True)
class Candidates:
    """What the executor could do right now; the only things Jev may pick from."""

    structures: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    idle_combat: int = 0
    all_combat: int = 0
    has_harvester_item: bool = False
    has_power_item: bool = False
    income: float = 0.0
    scout_alive: bool = False
    describe: dict[str, str] = field(default_factory=dict)
    costs: dict[str, float] = field(
        default_factory=dict
    )  # item -> credit cost, for army composition values
    formation: dict[str, int] = field(
        default_factory=dict
    )  # my battalions: scout/guard/flanker/striker counts
    battle: dict[str, Any] = field(default_factory=dict)  # both sides, from redalert.intel
    doctrine: str = ""  # the battle plan in force
    doctrine_for_s: float = 0.0
    # The executor's own money numbers, so Jev answers against the same ledger it runs on.
    burn_struct: float = 0.0  # credits/s already committed by structure queues
    burn_units: float = 0.0  # credits/s already committed by unit queues
    budget: float = 0.0  # the draw budget the executor enforces
    combat_roster: tuple[str, ...] = ()  # every combat unit buildable now (the army-focus menu)


@dataclass(frozen=True)
class Advice:
    at: float
    tick: int | None
    latency: float
    values: dict[str, Any]  # slot -> chosen key / level / probability
    confidence: dict[str, float]  # slot -> confidence (choice and score only)
    probs: dict[str, dict[str, float]] = field(
        default_factory=dict
    )  # slot -> per-option probabilities

    def get(self, slot: str, plan: Plan, *, now: float) -> Any:
        """The value for ``slot`` if fresh, allowed by the plan and confident enough, else ``None``."""
        if now - self.at > ADVICE_TTL_S or slot not in plan.jev_slots or slot not in self.values:
            return None
        conf = self.confidence.get(slot)
        if conf is not None and conf < plan.min_confidence:
            return None
        return self.values[slot]


def _level(score: float | None, top: int) -> int | None:
    return None if score is None else max(0, min(top, int(round(score))))


class TacticalAdvisor:
    def __init__(
        self,
        client: JevClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
        telemetry_path: str | None = None,
        max_calls: int = MAX_CALLS,
    ) -> None:
        self.client = client
        self._clock = clock
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._telemetry = telemetry_path
        self._max_calls = max_calls
        self._lock = threading.Lock()
        self._advice: Advice | None = None
        self._busy = False
        self._last_start = -1e9
        self._calls = 0
        self._errors = 0
        self._backoff_until = 0.0
        self._last_seq = -1
        self._urgent = False
        self._escalations: list[str] = []
        self._esc_at: dict[str, float] = {}
        self._style_last = ""
        self._announced = False

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def now(self) -> float:
        return self._clock()

    # -- read side (every decision) ----------------------------------------------

    def current(self) -> Advice | None:
        with self._lock:
            return self._advice

    def pop_escalations(self) -> list[str]:
        with self._lock:
            out, self._escalations = self._escalations, []
            return out

    # -- request building ----------------------------------------------------------

    @staticmethod
    def _counts(items: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and not it.get("wreck"):
                name = str(it.get("type")).lower()
                out[name] = out.get(name, 0) + 1
        return out

    @staticmethod
    def _detail(
        items: Any, idle_ids: set[int], costs: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Per-type combat composition: how many, how many idle, what the group is worth in credits."""
        out: dict[str, dict[str, float]] = {}
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or it.get("wreck"):
                continue
            name = str(it.get("type")).lower()
            if name in NON_COMBAT:
                continue
            d = out.setdefault(name, {"count": 0, "idle": 0, "value": 0})
            d["count"] += 1
            d["idle"] += 1 if it.get("id") in idle_ids else 0
            d["value"] += costs.get(name, 0)
        return out

    def build_request(
        self, raw: dict[str, Any], plan: Plan, cands: Candidates
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = raw.get("observation") or {}
        events = [e for e in (obs.get("events") or []) if isinstance(e, dict)]
        kinds: dict[str, int] = {}
        for e in events:
            kinds[str(e.get("kind"))] = kinds.get(str(e.get("kind")), 0) + 1
        time_info = raw.get("time") or {}
        military = obs.get("military") or {}
        econ = obs.get("economy") or {}
        provided, drained = econ.get("power_provided"), econ.get("power_drained")
        production = [p for p in (obs.get("production") or []) if isinstance(p, dict)][:10]
        state = {
            "clock": {
                "game_s": time_info.get("game_seconds"),
                "remaining_s": time_info.get("remaining_seconds"),
            },
            "my_faction": obs.get("faction"),
            "enemy_faction": obs.get("enemy_faction"),
            "economy": econ,
            "power_headroom": (
                provided - drained
                if isinstance(provided, (int, float)) and isinstance(drained, (int, float))
                else None
            ),
            "explored_pct": obs.get("explored_percent"),
            # The executor's own money ledger: what income remains against what is already committed.
            "econ_flow": {
                "income_per_s": round(cands.income, 1),
                "burn_structures_per_s": round(cands.burn_struct, 1),
                "burn_units_per_s": round(cands.burn_units, 1),
                "draw_budget_per_s": round(cands.budget, 1),
            },
            "production_detail": [
                {k: p.get(k) for k in ("item", "queue", "progress", "remaining_s", "ready")}
                for p in production
            ],
            "ready_to_place": [
                str(p.get("item")).lower()
                for p in production
                if p.get("ready") and str(p.get("queue") or "").lower() in ("building", "defense")
            ],
            "my_units": self._counts(obs.get("units")),
            "my_buildings": self._counts(obs.get("buildings")),
            "idle_combat_units": cands.idle_combat,
            "all_combat_units": cands.all_combat,
            # How my army is organized right now: who holds, who raids, who is in the wave.
            "my_formation": cands.formation,
            # The armies in detail: per type, how many, how many idle, what the group is worth in credits.
            "my_army_detail": self._detail(
                obs.get("units"),
                {
                    i
                    for i in ((raw.get("legal_actions") or {}).get("combat_units") or [])
                    if isinstance(i, int)
                },
                cands.costs,
            ),
            "enemy_army_detail": self._detail(obs.get("visible_enemies"), set(), cands.costs),
            "enemy_units_seen": self._counts(obs.get("visible_enemies")),
            "enemy_buildings_seen": self._counts(obs.get("visible_enemy_buildings")),
            "recent_event_counts": kinds,
            "recent_events": [
                {"kind": e.get("kind"), "type": e.get("type") or e.get("actor_type")}
                for e in events[-6:]
            ],
            "military": {
                k: military.get(k)
                for k in (
                    "kills_cost",
                    "deaths_cost",
                    "army_value",
                    "assets_value",
                    "units_lost",
                    "units_killed",
                )
            },
            "scout_alive": cands.scout_alive,
            # Both sides in numbers. Fog of war: the enemy is what we have seen (units for 90 s, buildings until
            # destroyed). No plan text or army mode goes in: Jev echoed "hold" from it for ten minutes.
            "battle": cands.battle,
            "current_battle_plan": {"name": cands.doctrine, "for_s": round(cands.doctrine_for_s)},
            # The rules an answer has to respect. Trades decide the time limit before assets do, so the
            # composition of a fight matters more than the size of the army left standing at home.
            "rules": {
                "win": "Destroy every enemy building and the MCV before the time limit.",
                "time_limit": "At the limit the higher (kills value - losses value) wins; if tied, the higher "
                "assets value wins. Check clock.remaining_s. A big army sitting at home earns "
                "nothing: it must either win trades or destroy buildings.",
                "trades": "Every fight should cost the enemy more than it costs us: feeding units piecemeal into "
                "a stronger army or into defences loses the match even with more units and more assets. "
                "A wave should leave only when it can beat what defends the target.",
                "closing": "With under five minutes left and negative trades the executor attacks with "
                "everything; when ahead on trades it protects the lead instead.",
                "economy": "Income is only harvesters on ore. At cash 0 the engine draws credits across ALL "
                "production queues at once, so overcommitting crawls everything; low power halves "
                "production speed.",
            },
        }
        if plan.tactics_hint:
            state["strategy_hint"] = plan.tactics_hint
        questions: dict[str, dict[str, Any]] = {}
        meta: dict[str, Any] = {}

        def choice(name: str, slot: str, text: str, options: dict[str, str | None]) -> None:
            if slot in plan.jev_slots and len(options) >= 2:
                questions[name] = {"type": "choice", "instructions": text, "criteria": options}
                meta[name] = slot

        choice(
            "next_structure",
            "structure",
            "Which structure should I build next to win this match given the economy, power, tech and what the enemy has?",
            {i: cands.describe.get(i) for i in cands.structures},
        )
        choice(
            "next_unit",
            "unit",
            "Which unit should I train next to counter what the enemy fields and keep my economy and army balanced?",
            {i: cands.describe.get(i) for i in cands.units},
        )
        choice(
            "battle_plan",
            "doctrine",
            "You command my side in this real-time Red Alert 1v1. Compare both armies: `battle` has the "
            "values, the trade score and where my army sits (home or in the field), `my_army_detail` and "
            "`enemy_army_detail` have the per-type compositions. Which battle plan should my side follow for "
            "the next ten seconds? If the enemy army has not been seen, assume it is at least as strong as "
            "mine.",
            {name: d.when for name, d in DOCTRINES.items()},
        )
        choice(
            "army_stage",
            "stage",
            "Given my economy, army, the threat in `battle` and the clock: where should my money go right now?",
            dict(STAGES),
        )
        if "threat" in plan.jev_slots:
            questions["threat"] = {
                "type": "score",
                "instructions": "How serious is the enemy threat to my base and economy right now?",
                "criteria": list(THREAT_LEVELS),
            }
            meta["threat"] = "threat"
        seen_any = (
            bool(self._counts(obs.get("visible_enemies")))
            or bool(self._counts(obs.get("visible_enemy_buildings")))
            or (
                isinstance((cands.battle or {}).get("enemy_army_value_seen_last_90s"), (int, float))
                and (cands.battle or {}).get("enemy_army_value_seen_last_90s", 0) > 0
            )
        )
        if seen_any and "style" in plan.jev_slots:
            # Reading the opponent: the answer paces the flank runs and sizes the home guard in the executor.
            choice(
                "enemy_style",
                "style",
                "What is the enemy doing with their army, judging by enemy_army_detail, enemy_buildings_seen "
                "and recent_events? This decides how I defend and where I harass.",
                {
                    "mass_infantry": "many riflemen/rocketeers, few vehicles: slow, strong in numbers, weak "
                    "against vehicles and shrugs off nothing",
                    "mass_armor": "mostly tanks: a strong frontal force that leaves its economy exposed",
                    "mixed": "a balanced combined force",
                    "turtle": "holding ground near their base or a chokepoint behind defences, not pushing",
                    "harass": "small packs raiding my harvesters and expansions from different sides",
                    "unknown": "not enough seen yet to tell",
                },
            )
        if cands.idle_combat >= 3:
            choice(
                "attack_target", "target", "Where should the next attack wave go?", dict(TARGETS)
            )
            if "wave" in plan.jev_slots:
                questions["wave_now"] = {
                    "type": "noul",
                    "instructions": "Should I send my idle fighters as an attack wave right now?",
                    "criteria": {
                        "true": "The idle force is big enough to beat what defends the target, and the clock "
                        "(clock.remaining_s) makes attacking now worth more than saving the force",
                        "false": "Wait: the force is too small or still assembling, the target too strong or "
                        "defended, or the army is needed at home",
                    },
                }
                meta["wave_now"] = "wave"
        if cands.combat_roster and "focus" in plan.jev_slots:
            choice(
                "army_focus",
                "focus",
                "Which of my buildable unit types should dominate my army, given what the enemy fields and the "
                "clock? Composition beats count.",
                {i: cands.describe.get(i) for i in cands.combat_roster},
            )
        if cands.has_harvester_item and "harvester" in plan.jev_slots:
            questions["need_harvester"] = {
                "type": "noul",
                "instructions": "Do I need another harvester to make the most of my refineries and ore, given my cash flow and what I am already building?",
                "criteria": {
                    "true": "My harvesters are fewer than about two per refinery (or income is far below what my refineries could support) and the credits are not desperately needed elsewhere",
                    "false": "I already have enough harvesters for my refineries, or defence/army needs the money more right now",
                },
            }
            meta["need_harvester"] = "harvester"
        if cands.has_power_item and "power" in plan.jev_slots:
            questions["need_power"] = {
                "type": "noul",
                "instructions": "Do I need another power plant soon: is power thin or about to be, given what is queued?",
                "criteria": {
                    "true": "Power headroom is thin or negative now, or queued buildings will push drain past supply",
                    "false": "Power is comfortable with room for what is queued",
                },
            }
            meta["need_power"] = "power"
        # The SystemOne API answers at most ten questions per request; the order above is the priority.
        questions = dict(list(questions.items())[:10])
        meta = {n: s for n, s in meta.items() if n in questions}
        return {"state": state, "questions": questions}, meta

    # -- refresh -------------------------------------------------------------------

    def maybe_refresh(
        self, raw: dict[str, Any], plan: Plan, cands: Candidates, *, sync: bool = False
    ) -> None:
        """Start a background Jev request if one is due. Never blocks the 5 Hz loop."""
        if self.client is None or not isinstance(raw.get("observation"), dict):
            return
        now = self._clock()
        events = [e for e in (raw["observation"].get("events") or []) if isinstance(e, dict)]
        with self._lock:
            if not self._announced:
                self._announced = True
                self._log(
                    "[jev] tactical advisor on: Jev decides fast tactics, the LLM sets the strategy"
                )
            seqs = [e["seq"] for e in events if isinstance(e.get("seq"), int)]
            if seqs:
                fresh = [
                    e for e in events if isinstance(e.get("seq"), int) and e["seq"] > self._last_seq
                ]
                if self._last_seq >= 0 and any(str(e.get("kind")) in URGENT_EVENTS for e in fresh):
                    self._urgent = True
                self._last_seq = max(self._last_seq, max(seqs))
            gap = REFRESH_URGENT_S if self._urgent else REFRESH_S
            if (
                self._busy
                or self._calls >= self._max_calls
                or now < self._backoff_until
                or now - self._last_start < gap
            ):
                return
            request, meta = self.build_request(raw, plan, cands)
            if not request["questions"]:
                return
            self._busy, self._urgent = True, False
            self._last_start = now
            self._calls += 1
        tick = raw["observation"].get("tick")
        if sync:
            self._run(request, meta, tick, plan)
        else:
            threading.Thread(
                target=self._run, args=(request, meta, tick, plan), daemon=True
            ).start()

    def _run(self, request: dict[str, Any], meta: dict[str, Any], tick: Any, plan: Plan) -> None:
        started = self._clock()
        answers = None
        try:
            answers = self.client.ask(request["state"], request["questions"])  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            answers = None
        done = self._clock()
        advice = self._to_advice(answers, meta, tick, done, done - started) if answers else None
        with self._lock:
            self._busy = False
            if advice is None:
                self._errors += 1
                self._backoff_until = done + min(
                    BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** min(self._errors - 1, 4)
                )
                if self._errors in (1, 5):
                    self._log(
                        f"[jev] no answer ({getattr(self.client, 'last_error', None)}); using the rules"
                    )
                return
            self._errors = 0
            self._advice = advice
            self._escalate(advice, plan, done)
        self._record(request, advice)

    def _to_advice(
        self, answers: dict[str, Answer], meta: dict[str, Any], tick: Any, at: float, latency: float
    ) -> Advice | None:
        values: dict[str, Any] = {}
        conf: dict[str, float] = {}
        probs: dict[str, dict[str, float]] = {}
        for name, slot in meta.items():
            a = answers.get(name)
            if a is None:
                continue
            if a.type == "choice" and a.choice:
                values[slot] = a.choice
            elif a.type == "score":
                level = _level(a.score, len(THREAT_LEVELS) - 1)
                if level is None:
                    continue
                values[slot] = level
            elif a.type == "noul" and a.noul is not None:
                values[slot] = a.noul
            else:
                continue
            if a.confidence is not None:
                conf[slot] = a.confidence
            if a.probabilities:
                probs[slot] = a.probabilities
        return (
            Advice(
                at, tick if isinstance(tick, int) else None, round(latency, 2), values, conf, probs
            )
            if values
            else None
        )

    def _escalate(self, advice: Advice, plan: Plan, now: float) -> None:
        """Wake the LLM strategist when Jev is unsure on a big call or sees a serious threat."""

        def push(kind: str, cooldown: float) -> None:
            if now - self._esc_at.get(kind, -1e9) >= cooldown:
                self._esc_at[kind] = now
                self._escalations.append(kind)

        threat = advice.values.get("threat")
        if (
            isinstance(threat, int)
            and threat >= ESCALATE_LEVEL
            and advice.confidence.get("threat", 1.0) >= plan.min_confidence
        ):
            push(f"jev_threat:{threat}", THREAT_COOLDOWN_S)
        style = advice.values.get("style")
        if isinstance(style, str) and style not in ("", "unknown") and style != self._style_last:
            self._style_last = style
            push(
                f"enemy_style:{style}", STYLE_COOLDOWN_S
            )  # the enemy switched posture: re-plan for it
        for slot in ("doctrine",):
            c = advice.confidence.get(slot)
            if slot in plan.jev_slots and c is not None and c < plan.min_confidence:
                push(f"jev_uncertain:{slot}", UNCERTAIN_COOLDOWN_S)

    def _record(self, request: dict[str, Any], advice: Advice) -> None:
        trace.write(
            "jev",
            tick=advice.tick,
            latency_s=advice.latency,
            values=advice.values,
            confidence=advice.confidence,
            asked=list(request["questions"]),
        )
        if not self._telemetry:
            return
        try:
            with open(self._telemetry, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "t": round(time.time(), 2),
                            "tick": advice.tick,
                            "latency_s": advice.latency,
                            "values": advice.values,
                            "confidence": advice.confidence,
                            "questions": list(request["questions"]),
                        },
                        default=str,
                    )
                    + "\n"
                )
        except OSError:
            pass
