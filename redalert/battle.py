"""The fight model: who wins an engagement, at what cost, and how sure we are.

A commander does not compare two army values and attack above a ratio. They ask what each side's units can do to
each other, how many of the enemy there probably are (including the ones out of sight), what the ground gives the
defender, and then they watch the first seconds of the fight to see whether it is going as they expected.

* **Matchups** (:data:`MATCHUP`): damage each kind of unit does to each other kind, per credit of force. Rifle
  infantry mows infantry and is useless against armour; rocket infantry kills armour; tanks crush infantry and
  artillery; static defences shred infantry. This is domain knowledge, written once and visible here, not a pile of
  thresholds.
* **Lanchester exchange**: both sides destroy each other's value at rates set by those matchups and by how much of
  the opponent's force each class makes up. It is the square law: concentration is rewarded automatically, and
  splitting a force costs more than the numbers suggest.
* **Belief, not sight**: the enemy side of the sum is what we see plus the share of the unseen force
  (:class:`redalert.memory.Belief`) that can reach the fight, spread over three scenarios (less, as believed,
  more). The answer is an expectation and a pessimistic bound, so uncertainty costs something.
* **Feedback**: the model's expected exchange is recorded; the fight ledger compares it with what really
  happened. Fights that keep going worse than predicted pull later predictions down, and a fight that is going
  badly right now is a reason to leave it (:meth:`Judge.contact`).

The numbers are the model's assumptions, calibrated online; the decision is always "does this fight improve the
exchange", not "is the ratio above N".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from redalert.memory import MatchMemory

# -- unit classes ----------------------------------------------------------------------------------------------

RIF, ROC, TANK, ART, FAST, TOW = "rifle", "rocket", "armour", "artillery", "fast", "static"
CLASSES = (RIF, ROC, TANK, ART, FAST, TOW)

UNIT_CLASS = {
    "e1": RIF,
    "e2": RIF,
    "e4": RIF,
    "e7": RIF,
    "shok": RIF,
    "medi": RIF,
    "e6": RIF,
    "e3": ROC,
    "1tnk": TANK,
    "2tnk": TANK,
    "3tnk": TANK,
    "4tnk": TANK,
    "ttnk": TANK,
    "apc": TANK,
    "ftrk": TANK,
    "mgg": TANK,
    "jeep": FAST,
    "ftrk_aa": TANK,
    "arty": ART,
    "v2rl": ART,
    "mrj": ART,
    "ctnk": ART,
    "dog": FAST,
    "mnly": FAST,
}

# What a unit of the row's class does to the column's class, per credit of force, relative to rifle-on-rifle (1.0).
MATCHUP: dict[str, dict[str, float]] = {
    RIF: {RIF: 1.0, ROC: 1.0, TANK: 0.35, ART: 0.8, FAST: 0.9, TOW: 0.2},
    ROC: {RIF: 0.5, ROC: 0.6, TANK: 1.5, ART: 1.2, FAST: 0.6, TOW: 0.9},
    TANK: {RIF: 1.4, ROC: 1.2, TANK: 1.0, ART: 1.6, FAST: 1.1, TOW: 1.0},
    ART: {RIF: 1.3, ROC: 1.3, TANK: 0.9, ART: 1.2, FAST: 0.5, TOW: 1.2},
    FAST: {RIF: 1.0, ROC: 0.7, TANK: 0.2, ART: 0.8, FAST: 0.8, TOW: 0.1},
    TOW: {RIF: 1.8, ROC: 1.5, TANK: 1.4, ART: 1.5, FAST: 1.3, TOW: 0.0},
}
DEFAULT_MIX = {
    TANK: 0.45,
    RIF: 0.35,
    ROC: 0.2,
}  # what an unseen enemy force is assumed to be made of

NO_FIGHT_RATIO = 10.0
SAFE_RATIO = 4.0
RATE = 0.035  # fraction of an opposing value one credit of force destroys per second at matchup 1.0 (half-life ~20 s)
HORIZON_S = 50
ATTACK_COHESION = 0.9  # a force that must walk into the fight loses some of its punch
DEFENDER_GROUND = 1.15  # fighting at its own base, with repair and reinforcements next door
HOME_GROUND = 1.1
SPEED = 2.6  # cells per second a fighting force covers (infantry pace), for how soon unseen force can arrive
STATIC_RANGE_S = 3  # seconds a retreating force stays inside a static defence's range
PURSUIT_S = 14  # seconds mobile enemies keep chasing before they turn back to their own ground
RETREAT_FIRE = 0.3  # a force moving away still returns this share of its fire
SCENARIO_WEIGHTS = (
    (-1.0, 0.25),
    (0.0, 0.5),
    (1.0, 0.25),
)  # (sigma multiples, weight) over the hidden force


def classify(kind: str, cost: float, tower_types: frozenset[str] | set[str] = frozenset()) -> str:
    if kind in tower_types:
        return TOW
    if kind in UNIT_CLASS:
        return UNIT_CLASS[kind]
    return TANK if cost >= 600 else RIF


@dataclass
class Verdict:
    """The model's reading of one engagement."""

    ratio: float  # their value destroyed per ours lost, expected (calibrated against past fights)
    pessimistic: float  # the same if the unseen force is at the high end of the belief
    our_loss: float
    their_loss: float
    our_survive: float  # fraction of our force still standing at the end (expected)
    their_survive: float
    our_value: float
    their_value: float
    hidden_counted: float  # how much unseen enemy force was counted into this fight
    a0: dict[str, float] = field(default_factory=dict)  # our force by class at the start
    b0: dict[str, float] = field(default_factory=dict)  # theirs, with the unseen part counted
    notes: list[str] = field(default_factory=list)

    def as_report(self) -> dict[str, Any]:
        return {
            "expected_exchange": round(self.ratio, 2),
            "pessimistic_exchange": round(self.pessimistic, 2),
            "we_lose": round(self.our_loss),
            "they_lose": round(self.their_loss),
            "we_survive_pct": round(100 * self.our_survive),
            "they_survive_pct": round(100 * self.their_survive),
            "enemy_value_counted": round(self.their_value),
            "of_which_unseen": round(self.hidden_counted),
        }


def _exchange(
    a: dict[str, float],
    b: dict[str, float],
    a_mult: float,
    b_mult: float,
    horizon: int = HORIZON_S,
    b_exposure: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Both sides destroy each other's value for ``horizon`` seconds.

    ``b_exposure`` limits how long a class of theirs can keep firing at us (seconds): a force walking away is out
    of a static defence's range within seconds, while pursuers keep shooting for as long as they chase."""
    a, b = dict(a), dict(b)
    for step in range(horizon):
        ta, tb = sum(a.values()), sum(b.values())
        if ta < 1.0 or tb < 1.0:
            break
        if min(ta, tb) < 0.04 * max(ta, tb):
            break  # one side is gone for practical purposes
        hit_b = {d: 0.0 for d in CLASSES}
        hit_a = {c: 0.0 for c in CLASSES}
        for c, va in a.items():
            if va <= 0:
                continue
            for d, vb in b.items():
                if vb <= 0:
                    continue
                hit_b[d] += va * MATCHUP[c][d] * a_mult * RATE * (vb / tb)
        for c, vb in b.items():
            if vb <= 0:
                continue
            if b_exposure is not None and step >= b_exposure.get(c, horizon):
                continue
            for d, va in a.items():
                if va <= 0:
                    continue
                hit_a[d] += vb * MATCHUP[c][d] * b_mult * RATE * (va / ta)
        for d in CLASSES:
            if d in b:
                b[d] = max(0.0, b[d] - hit_b[d])
            if d in a:
                a[d] = max(0.0, a[d] - hit_a[d])
    return a, b


class Judge:
    """Reads engagements for the officers, against the match memory."""

    def __init__(self, memory: MatchMemory) -> None:
        self.memory = memory
        self.tower_types = memory.tower_types

    # -- the core ---------------------------------------------------------------------------------------------

    def known_towers(self, where: tuple[float, float] | None, radius: float, visible: list[str]) -> list[str]:
        """Defences near a point: what is in view plus what the memory has seen there and not seen destroyed."""
        if where is None:
            return list(visible)
        remembered = [
            s.kind
            for s in self.memory.buildings.values()
            if s.kind in self.tower_types and s.cell and math.dist(s.cell, where) <= radius
        ]
        out: list[str] = []
        for kind in {*remembered, *visible}:
            out += [kind] * max(remembered.count(kind), visible.count(kind))
        return out

    def _forces(
        self,
        ours: list[tuple[str, float]],
        enemies: list[tuple[str, float]],
        towers: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        a = {c: 0.0 for c in CLASSES}
        for kind, hp in ours:
            cost = float(self.memory.cost_of(kind))
            a[classify(kind, cost, self.tower_types)] += cost * max(0.1, min(1.0, hp))
        b = {c: 0.0 for c in CLASSES}
        for kind, hp in enemies:
            cost = float(self.memory.cost_of(kind))
            b[classify(kind, cost, self.tower_types)] += cost * max(0.1, min(1.0, hp))
        for kind in towers:
            b[TOW] += float(self.memory.cost_of(kind))
        return a, b

    def _hidden(
        self, game_s: float, visible_value: float, reach: float, scale: float
    ) -> dict[str, float]:
        """The unseen part of the enemy's army that can be at this fight, by class."""
        belief = self.memory.belief(game_s)
        counted = max(0.0, belief.hidden_value) * max(0.0, min(1.0, reach)) * scale
        comp = belief.composition
        mix: dict[str, float] = {}
        total = sum(comp.values())
        if total >= 4:
            for kind, n in comp.items():
                c = classify(kind, float(self.memory.cost_of(kind)), self.tower_types)
                mix[c] = mix.get(c, 0.0) + n * float(self.memory.cost_of(kind))
            norm = sum(mix.values()) or 1.0
            mix = {c: v / norm for c, v in mix.items()}
        else:
            mix = dict(DEFAULT_MIX)
        return {c: counted * mix.get(c, 0.0) for c in CLASSES}

    def assess(
        self,
        ours: list[tuple[str, float]],
        enemies: list[tuple[str, float]],
        towers: list[str],
        *,
        reach: float,
        attacking: bool,
    ) -> Verdict:
        game_s = self.memory._last_t or 0.0
        a, b_seen = self._forces(ours, enemies, towers)
        belief = self.memory.belief(game_s)
        sigma = 0.25 + 0.4 * min(1.0, belief.since_seen_s / 120.0)
        a_mult = ATTACK_COHESION if attacking else HOME_GROUND
        b_mult = DEFENDER_GROUND if attacking else 1.0
        ratios: list[tuple[float, float, dict[str, float], dict[str, float], dict[str, float]]] = []
        for s, w in SCENARIO_WEIGHTS:
            hidden = self._hidden(
                game_s, sum(b_seen.values()), reach, max(0.0, 1.0 + s * sigma * 1.6)
            )
            b = {c: b_seen[c] + hidden[c] for c in CLASSES}
            a_end, b_end = _exchange(a, b, a_mult, b_mult)
            ours_lost = sum(a.values()) - sum(a_end.values())
            theirs_lost = sum(b.values()) - sum(b_end.values())
            eps = 0.02 * sum(a.values()) + 50.0
            ratio = (theirs_lost + eps) / (ours_lost + eps)
            if sum(b.values()) < 1.0:
                ratio = NO_FIGHT_RATIO  # nothing there to fight: not an even fight, an empty field
            elif ours_lost < 0.02 * sum(a.values()):
                ratio = max(ratio, SAFE_RATIO)  # we lose next to nothing: the exchange cannot be bad
            ratios.append((w, ratio, a_end, b_end, b))
        # expectation over the scenarios, in the log domain so a lopsided tail does not swamp the rest
        log_r = sum(w * math.log(r) for w, r, *_ in ratios)
        mid = ratios[1]
        a_end, b_end, b_all = mid[2], mid[3], mid[4]
        our_v, their_v = sum(a.values()), sum(b_all.values())
        calibrated = math.exp(log_r) * self.memory.exchange_calibration
        pessimistic = ratios[2][1] * self.memory.exchange_calibration
        hidden_counted = sum(b_all.values()) - sum(b_seen.values())
        return Verdict(
            ratio=calibrated,
            pessimistic=pessimistic,
            our_loss=our_v - sum(a_end.values()),
            their_loss=their_v - sum(b_end.values()),
            our_survive=(sum(a_end.values()) / our_v) if our_v else 0.0,
            their_survive=(sum(b_end.values()) / their_v) if their_v else 1.0,
            our_value=our_v,
            their_value=their_v,
            hidden_counted=max(0.0, hidden_counted),
            a0=a,
            b0=b_all,
        )

    # -- what the officers ask -----------------------------------------------------------------------------------

    def reach(self, where: tuple[float, float] | None, *, at_their_base: bool) -> float:
        """What share of the unseen enemy force can be in a fight here within the fight's span.

        At their base most of it is already there. Anywhere else it has to travel from there: the farther the
        fight is from their base, the less of it arrives in time."""
        if at_their_base:
            return 0.9
        base = self.memory.enemy_base
        if where is None or base is None:
            return 0.5
        d = math.dist(where, base)
        return max(0.1, min(0.9, 1.0 - d / (SPEED * HORIZON_S)))

    def advance(
        self,
        ours: list[tuple[str, float]],
        enemies: list[tuple[str, float]],
        towers: list[str],
        *,
        at_base: bool,
        where: tuple[float, float] | None = None,
    ) -> Verdict:
        """Should this force walk into that fight? The unseen army is mostly at their base; elsewhere less of it."""
        towers = self.known_towers(where, 16.0, towers)
        v = self.assess(ours, enemies, towers, reach=self.reach(where, at_their_base=at_base), attacking=True)
        self.memory.note_prediction(v.ratio, self.memory._last_t or 0.0)
        return v

    def contact(
        self,
        ours: list[tuple[str, float]],
        enemies: list[tuple[str, float]],
        towers: list[str],
        *,
        where: tuple[float, float] | None = None,
    ) -> Verdict:
        """A fight is on: what does staying in it look like from here, counting only the unseen force that can arrive?"""
        towers = self.known_towers(where, 16.0, towers)
        return self.assess(ours, enemies, towers, reach=self.reach(where, at_their_base=False), attacking=False)

    def break_off(
        self,
        ours: list[tuple[str, float]],
        enemies: list[tuple[str, float]],
        towers: list[str],
        *,
        where: tuple[float, float] | None,
        safe: tuple[float, float] | None,
        abort_ratio: float,
    ) -> tuple[bool, Verdict]:
        """Leave this fight, or stay? Leaving is only better if the fight is going badly AND getting out costs less
        than staying: a force that retreats takes the enemy's fire for the whole walk and returns little, so
        retreating from a fight that cannot be left is a way of losing the army without trading."""
        v = self.contact(ours, enemies, towers, where=where)
        bad = v.ratio < abort_ratio or self.going_badly(abort_ratio)
        if not bad or where is None or safe is None:
            return False, v
        d = math.dist(where, safe)
        if d <= 8.0:
            return False, v  # already at the safe ground: there is nowhere better to go
        steps = int(max(1.0, min(25.0, d / SPEED)))
        exposure = {c: PURSUIT_S for c in CLASSES}
        exposure[TOW] = STATIC_RANGE_S
        a_end, _ = _exchange(v.a0, v.b0, RETREAT_FIRE, 1.0, steps, exposure)
        leave_loss = sum(v.a0.values()) - sum(a_end.values())
        return leave_loss < 0.85 * v.our_loss, v

    def going_badly(self, abort_ratio: float) -> bool:
        """The feedback: the real exchange over the last 15 seconds of this fight against the line the General set."""
        fight = self.memory.open_fight
        now = self.memory._last_t or 0.0
        if fight is None or now - fight.t0 < 8.0:
            return False
        if fight.our_lost + fight.their_lost < 400:
            return False
        return fight.ratio < abort_ratio * 0.8
