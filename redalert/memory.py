"""Match memory: the whole game, kept.

The engine shows only what is in view this tick. A commander does not play that way: they remember where the enemy
was seen and with what, what their economy and tech look like, every fight so far and why it went as it did, and
what each earlier plan actually achieved. This module is that memory. It is updated every decision, never reset,
and read by two consumers:

* the General (the strategist model), through :meth:`MatchMemory.briefing`: the long-run picture it plans from;
* the fight model (:mod:`redalert.battle`), through :meth:`MatchMemory.belief` and the fight ledger's realised
  exchange rate, which it uses to judge an engagement and to correct itself.

Only engine data goes in: types, ids, cells, hp, costs and the match's own counters.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

START_CASH = 5000.0
UNIT_COST_HARVESTER = 1100.0
CONTACT_R = 12.0  # an enemy combat unit this close to one of ours is a fight
FIGHT_RADIUS = 22.0  # what counts as "in this fight" around its centre
CALIBRATION_PRIOR = 2.0  # pseudo-fights that went as predicted
FIGHT_QUIET_S = 20.0  # no contact and no losses for this long ends a fight
SNAPSHOT_EVERY_S = 15.0
MAX_FIGHTS = 40
MAX_PLANS = 12
PRODUCTION = {"barr", "tent", "weap", "afld", "hpad", "syrd", "spen"}
TECH = {"dome", "atek", "stek", "fix", "apwr", "mslo", "iron", "pdox"}
ECONOMY = {"proc", "silo"}


def _cell(v: Any) -> tuple[float, float] | None:
    if (
        isinstance(v, dict)
        and isinstance(v.get("x"), (int, float))
        and isinstance(v.get("y"), (int, float))
    ):
        return float(v["x"]), float(v["y"])
    if (
        isinstance(v, (list, tuple))
        and len(v) >= 2
        and all(isinstance(c, (int, float)) for c in v[:2])
    ):
        return float(v[0]), float(v[1])
    return None


def _centroid(cells: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not cells:
        return None
    return sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells)


@dataclass
class Sighting:
    kind: str
    cell: tuple[float, float] | None
    hp: float | None
    first: float
    last: float


@dataclass
class Fight:
    """One engagement: who was there, what it cost each side, and what the model expected."""

    t0: float
    where: tuple[float, float] | None
    ours: dict[str, int]
    t1: float | None = None
    theirs_peak: dict[str, int] = field(default_factory=dict)
    theirs_peak_value: float = 0.0
    towers_near: int = 0
    our_lost: float = 0.0
    their_lost: float = 0.0
    units_lost: int = 0
    units_killed: int = 0
    buildings_lost: int = 0
    buildings_killed: int = 0
    predicted_ratio: float | None = None  # what the fight model expected this exchange to be
    result: str = "open"
    deaths: list[tuple[float, float]] = field(default_factory=list)  # where our units actually fell

    @property
    def ratio(self) -> float:
        """Realised exchange: value of theirs destroyed per value of ours lost (smoothed)."""
        return (self.their_lost + 100.0) / (self.our_lost + 100.0)

    def as_brief(self) -> dict[str, Any]:
        return {
            "t": round(self.t0),
            "secs": round((self.t1 or self.t0) - self.t0),
            "where": [round(self.where[0]), round(self.where[1])] if self.where else None,
            "we_had": self.ours,
            "they_had_peak": self.theirs_peak,
            "towers_near": self.towers_near,
            "we_lost_value": round(self.our_lost),
            "they_lost_value": round(self.their_lost),
            "exchange": round(self.ratio, 2),
            "expected_exchange": round(self.predicted_ratio, 2) if self.predicted_ratio else None,
            "result": self.result,
            "our_units_fell_at": (
                [round(c[0]), round(c[1])] if (c := _centroid(self.deaths)) else None
            ),
            "no_enemy_unit_in_view_while_we_lost": bool(self.our_lost > 0 and not self.theirs_peak_value),
        }


@dataclass
class PlanRecord:
    version: int
    t: float
    assessment: str
    missions: Any
    start: dict[str, float]
    outcome: dict[str, float] | None = None


@dataclass
class Belief:
    """What we believe the enemy has right now, with the doubt attached."""

    visible_value: float  # in view this tick
    remembered_value: float  # seen earlier, out of view, probably still alive
    army_value: float  # best single estimate of their whole field army
    low: float
    high: float
    hidden_value: float  # army_value not accounted for by sightings
    income: float  # their estimated credits per second
    since_seen_s: float  # how long since any enemy combat unit was in view
    towers: dict[str, int]  # defensive structures known, by type
    composition: dict[str, int]  # best guess of their army by type (seen + remembered)


class MatchMemory:
    def __init__(
        self,
        cost_of: Callable[[str], float],
        *,
        tower_types: frozenset[str] = frozenset(),
        non_combat: frozenset[str] = frozenset(),
    ) -> None:
        self.cost_of = cost_of
        self.tower_types = tower_types
        self.non_combat = non_combat
        # what the enemy has been seen to have, for the whole match
        self.units: dict[int, Sighting] = {}
        self.buildings: dict[int, Sighting] = {}
        self.peak_army: dict[str, int] = {}
        self.peak_value = 0.0
        self.harvesters_seen = 0
        self.last_combat_seen = -1e9
        self.enemy_base: tuple[float, float] | None = None
        # money: our income integrated, the enemy's inferred from it
        self.income_cum = 0.0
        self._last_t: float | None = None
        self._income_now = 0.0
        self._my_harvesters = 1
        self._my_infra = 0.0
        self._my_army_value = 0.0
        self._my_losses = 0.0
        self._base_view_at = -1e9
        self._mine_prev: dict[int, tuple[str, tuple[float, float] | None, float | None]] = {}
        self.harv_lost: deque[dict[str, Any]] = deque(maxlen=12)
        # the fights
        self.fights: deque[Fight] = deque(maxlen=MAX_FIGHTS)
        self._open: Fight | None = None
        self._quiet_since: float | None = None
        self._counters: tuple[float, ...] | None = None
        self._pending_prediction: tuple[float, float] | None = None  # (time, predicted ratio)
        self._calib: float = 0.0  # ln of the correction: weighted mean of ln(realised / predicted) with a prior
        self._calib_sum = 0.0
        self._calib_w = 0.0
        # the plans and the standing
        self.plans: deque[PlanRecord] = deque(maxlen=MAX_PLANS)
        self._plan_version = -1
        self.standing: deque[dict[str, float]] = deque(maxlen=200)
        self._snap_at = -1e9
        self._bias = 1.0  # learned: how far the force estimate ran above/below what fights revealed

    # -- the feed ---------------------------------------------------------------------------------------------

    def update(
        self,
        obs: dict[str, Any],
        game_s: float,
        *,
        income: float,
        home: tuple[float, float] | None,
        enemy_base: tuple[float, float] | None,
        fighters: list[dict[str, Any]],
        harvesters: int,
    ) -> None:
        dt = 0.0 if self._last_t is None else max(0.0, min(2.0, game_s - self._last_t))
        self._last_t = game_s
        self._income_now = max(0.0, income)
        self.income_cum += self._income_now * dt
        self._my_harvesters = max(1, harvesters)
        mil = obs.get("military") or {}
        self._my_army_value = float(mil.get("army_value") or 0)
        self._my_losses = float(mil.get("deaths_cost") or 0)
        self._my_infra = sum(
            self.cost_of(str(b.get("type")).lower())
            for b in (obs.get("buildings") or [])
            if isinstance(b, dict) and str(b.get("type")).lower() not in self.tower_types
        )
        if enemy_base:
            self.enemy_base = (float(enemy_base[0]), float(enemy_base[1]))
        visible = self._see(obs, game_s)
        died = self._casualties(obs, game_s)
        if self.enemy_base and any(
            (c := _cell(u)) and math.dist(c, self.enemy_base) <= 15.0
            for u in (obs.get("units") or [])
            if isinstance(u, dict)
        ):
            self._base_view_at = game_s
        self._fights(obs, game_s, fighters, visible, died)
        if game_s - self._snap_at >= SNAPSHOT_EVERY_S:
            self._snap_at = game_s
            m = obs.get("military") or {}
            b = self.belief(game_s)
            self.standing.append(
                {
                    "t": round(game_s),
                    "income": round(self._income_now, 1),
                    "cum": round(self.income_cum),
                    "harvesters": harvesters,
                    "my_army": m.get("army_value") or 0,
                    "assets": m.get("assets_value") or 0,
                    "kills": m.get("kills_cost") or 0,
                    "losses": m.get("deaths_cost") or 0,
                    "enemy_army_est": round(b.army_value),
                }
            )

    def _casualties(self, obs: dict[str, Any], game_s: float) -> list[tuple[str, tuple[float, float]]]:
        """Our units that were here a moment ago and are gone: where they fell, and for harvesters, what was near."""
        now: dict[int, tuple[str, tuple[float, float] | None, float | None]] = {}
        for u in obs.get("units") or []:
            if isinstance(u, dict) and isinstance(u.get("id"), int) and not u.get("wreck"):
                now[u["id"]] = (str(u.get("type")).lower(), _cell(u), u.get("hp"))
        died: list[tuple[str, tuple[float, float]]] = []
        if self._mine_prev:
            enemies = [
                (str(e.get("type")).lower(), c)
                for e in obs.get("visible_enemies") or []
                if isinstance(e, dict) and not e.get("wreck") and (c := _cell(e))
            ]
            for uid, (kind, cell, hp) in self._mine_prev.items():
                if uid in now or cell is None:
                    continue
                died.append((kind, cell))
                if kind == "harv":
                    near = sorted(
                        {k for k, c in enemies if math.dist(c, cell) <= 12.0 and k not in self.non_combat}
                    )
                    self.harv_lost.append(
                        {
                            "t": round(game_s),
                            "where": [round(cell[0]), round(cell[1])],
                            "hp_before": hp,
                            "enemy_units_near": near,
                        }
                    )
        self._mine_prev = now
        return died

    def economy_picture(self, obs: dict[str, Any], game_s: float) -> dict[str, Any]:
        """Our own economy as measurements: what the harvesters are doing, what has killed them, what the income is."""
        units = [u for u in obs.get("units") or [] if isinstance(u, dict) and not u.get("wreck")]
        harv = [u for u in units if str(u.get("type")).lower() == "harv"]
        acts: dict[str, int] = {}
        for u in harv:
            key = str(u.get("act") or ("idle" if u.get("idle") else "none"))
            acts[key] = acts.get(key, 0) + 1
        cells = [c for u in harv if (c := _cell(u))]
        hostile = [
            str(e.get("type")).lower()
            for e in obs.get("visible_enemies") or []
            if isinstance(e, dict)
            and not e.get("wreck")
            and str(e.get("type")).lower() not in self.non_combat
            and (c := _cell(e))
            and any(math.dist(c, h) <= 14.0 for h in cells)
        ]
        refineries = [
            b for b in obs.get("buildings") or [] if isinstance(b, dict) and str(b.get("type")).lower() == "proc"
        ]
        econ = obs.get("economy") or {}
        y = self.harvester_yield()
        return {
            "income_per_s": round(self._income_now, 1),
            "income_per_harvester_per_s": round(self._income_now / len(harv), 1) if harv else None,
            "best_income_per_harvester_so_far": round(y[1], 1) if y else None,
            "harvesters_alive": len(harv),
            "harvesters_by_activity": acts,
            "harvesters_idle": sum(1 for u in harv if u.get("idle")),
            "harvester_hp_pct": [u.get("hp") for u in harv][:8],
            "enemy_combat_units_within_14_of_a_harvester": hostile,
            "harvesters_lost_recently": list(self.harv_lost)[-4:],
            "refineries": len(refineries),
            "refinery_hp_pct": [b.get("hp") for b in refineries][:4],
            "cash": econ.get("cash"),
            "ore_in_silos": econ.get("ore"),
        }

    def _see(
        self, obs: dict[str, Any], game_s: float
    ) -> list[tuple[str, tuple[float, float] | None]]:
        seen: list[tuple[str, tuple[float, float] | None]] = []
        in_view: set[int] = set()
        for e in obs.get("visible_enemies") or []:
            if not isinstance(e, dict) or not isinstance(e.get("id"), int):
                continue
            if e.get("wreck"):
                self.units.pop(e["id"], None)
                continue
            kind = str(e.get("type")).lower()
            cell = _cell(e)
            prev = self.units.get(e["id"])
            self.units[e["id"]] = Sighting(
                kind, cell, e.get("hp"), prev.first if prev else game_s, game_s
            )
            in_view.add(e["id"])
            if kind not in self.non_combat:
                self.last_combat_seen = game_s
                seen.append((kind, cell))
        for b in obs.get("visible_enemy_buildings") or []:
            if not isinstance(b, dict) or not isinstance(b.get("id"), int):
                continue
            if b.get("wreck"):
                self.buildings.pop(b["id"], None)
                continue
            prev = self.buildings.get(b["id"])
            self.buildings[b["id"]] = Sighting(
                str(b.get("type")).lower(),
                _cell(b),
                b.get("hp"),
                prev.first if prev else game_s,
                game_s,
            )
        # the enemy's base is where its buildings cluster, which beats the map's guess
        cells = [s.cell for s in self.buildings.values() if s.cell]
        if cells:
            self.enemy_base = _centroid(cells)
        # an enemy unit we last saw where one of ours now stands, but cannot see, is gone from there
        ours = [c for c in (_cell(u) for u in (obs.get("units") or []) if isinstance(u, dict)) if c]
        for uid in [
            i
            for i, s in self.units.items()
            if i not in in_view and s.cell and any(math.dist(s.cell, c) <= 4 for c in ours)
        ]:
            self.units.pop(uid, None)
        now_army: dict[str, int] = {}
        for s in self.units.values():
            if s.kind not in self.non_combat and game_s - s.last <= 12.0:
                now_army[s.kind] = now_army.get(s.kind, 0) + 1
        value = sum(self.cost_of(k) * n for k, n in now_army.items())
        if value > self.peak_value:
            self.peak_value, self.peak_army = value, dict(now_army)
        self.harvesters_seen = max(
            self.harvesters_seen,
            sum(1 for s in self.units.values() if s.kind == "harv" and game_s - s.last <= 30.0),
        )
        return seen

    # -- the fights -------------------------------------------------------------------------------------------

    def note_prediction(self, ratio: float, game_s: float) -> None:
        """The fight model says what it expects before it commits; the ledger checks it afterwards."""
        self._pending_prediction = (game_s, float(ratio))

    def _fights(
        self,
        obs: dict[str, Any],
        game_s: float,
        fighters: list[dict[str, Any]],
        visible: list[tuple[str, tuple[float, float] | None]],
        died: list[tuple[str, tuple[float, float]]] | None = None,
    ) -> None:
        m = obs.get("military") or {}
        counters = (
            float(m.get("kills_cost") or 0),
            float(m.get("deaths_cost") or 0),
            float(m.get("units_killed") or 0),
            float(m.get("units_lost") or 0),
            float(m.get("buildings_killed") or 0),
            float(m.get("buildings_lost") or 0),
        )
        prev = self._counters or counters
        self._counters = counters
        delta = tuple(max(0.0, a - b) for a, b in zip(counters, prev))
        mine = [(u, c) for u in fighters if (c := _cell(u))]
        contact = any(
            kind not in self.non_combat
            and cell
            and any(math.dist(cell, c) <= CONTACT_R for _, c in mine)
            for kind, cell in visible
        )
        changed = any(d > 0 for d in delta)
        if self._open is None:
            if not (changed or contact):
                return
            centre = _centroid([c for _, c in mine])
            near = [
                u for u, c in mine if centre and math.dist(c, centre) <= FIGHT_RADIUS * 1.5
            ] or [u for u, _ in mine]
            ours: dict[str, int] = {}
            for u in near:
                k = str(u.get("type")).lower()
                ours[k] = ours.get(k, 0) + 1
            self._open = Fight(t0=game_s, where=centre, ours=ours)
            if self._pending_prediction and game_s - self._pending_prediction[0] <= 40.0:
                self._open.predicted_ratio = self._pending_prediction[1]
            self._quiet_since = None
        fight = self._open
        fight.their_lost += delta[0]
        fight.our_lost += delta[1]
        fight.units_killed += int(delta[2])
        fight.units_lost += int(delta[3])
        fight.buildings_killed += int(delta[4])
        fight.buildings_lost += int(delta[5])
        centre = _centroid([c for _, c in mine]) or fight.where
        fight.deaths += [c for k, c in (died or []) if k not in self.non_combat]
        # A stretched-out group's centre is not where the fight is: losses fell where the enemy was. The enemy and
        # the defences count around both the group and the places our units fell.
        spots = [c for c in (centre, _centroid(fight.deaths[-8:])) if c]
        around: dict[str, int] = {}
        for kind, cell in visible:
            if kind in self.non_combat or not cell or not spots:
                continue
            if any(math.dist(cell, c) <= FIGHT_RADIUS * 1.5 for c in spots):
                around[kind] = around.get(kind, 0) + 1
        value = sum(self.cost_of(k) * n for k, n in around.items())
        if value > fight.theirs_peak_value:
            fight.theirs_peak_value, fight.theirs_peak = value, around
        if centre:
            fight.where = centre
        if spots:
            fight.towers_near = max(
                fight.towers_near,
                max(
                    sum(
                        1
                        for s in self.buildings.values()
                        if s.kind in self.tower_types and s.cell and math.dist(s.cell, c) <= 16
                    )
                    for c in spots
                ),
            )
        if changed or contact:
            self._quiet_since = None
            return
        if self._quiet_since is None:
            self._quiet_since = game_s
        if game_s - self._quiet_since >= FIGHT_QUIET_S:
            self._close(fight, game_s)

    def _close(self, fight: Fight, game_s: float) -> None:
        fight.t1 = game_s - FIGHT_QUIET_S
        r = fight.ratio
        fight.result = "won" if r >= 1.3 else "lost" if r <= 0.77 else "even"
        if fight.our_lost + fight.their_lost >= 300:
            self.fights.append(fight)
            if fight.predicted_ratio:
                # Past fights correct the model, but one skirmish must not rewrite it: each fight counts in
                # proportion to its size, against a prior of two fights that went exactly as predicted.
                err = math.log(max(0.1, r) / max(0.1, fight.predicted_ratio))
                w = min(1.0, (fight.our_lost + fight.their_lost) / 3000.0)
                self._calib_sum += w * err
                self._calib_w += w
                self._calib = self._calib_sum / (self._calib_w + CALIBRATION_PRIOR)
            self._learn_force(fight)
        self._open = None
        self._quiet_since = None

    def _learn_force(self, fight: Fight) -> None:
        """A fight reveals how much force the enemy really had at that moment; correct the economic estimate."""
        if fight.theirs_peak_value <= 0:
            return
        est = self._raw_army_estimate(fight.t0)
        if est > 0:
            self._bias = max(
                0.5, min(2.0, 0.7 * self._bias + 0.3 * (fight.theirs_peak_value / est))
            )

    @property
    def open_fight(self) -> Fight | None:
        return self._open

    @property
    def exchange_calibration(self) -> float:
        """Multiplier on predicted exchange ratios: below 1 means fights have gone worse than the model said."""
        return math.exp(self._calib)

    def realised_exchange(self, window_s: float, now: float) -> float | None:
        """Value we destroyed per value we lost across fights that ended in the last ``window_s`` (and the open one)."""
        lost = killed = 0.0
        for f in [*self.fights, *([self._open] if self._open else [])]:
            if f is None:
                continue
            end = f.t1 if f.t1 is not None else now
            if now - end <= window_s:
                lost += f.our_lost
                killed += f.their_lost
        if lost + killed < 300:
            return None
        return (killed + 100.0) / (lost + 100.0)

    # -- what we believe the enemy has --------------------------------------------------------------------------

    def _infrastructure(self) -> float:
        total = 0.0
        for s in self.buildings.values():
            if s.kind not in self.tower_types:
                total += self.cost_of(s.kind)
        return total

    def _my_efficiency(self) -> float:
        """What share of our non-infrastructure budget has become army: the same economy runs on their side."""
        spent_on_army = self._my_army_value + self._my_losses
        free = START_CASH + self.income_cum - self._my_infra - UNIT_COST_HARVESTER * self._my_harvesters
        if free < 1500:
            return 0.8
        return max(0.35, min(1.0, spent_on_army / free))

    def _raw_army_estimate(self, game_s: float) -> float:
        # Both sides start with the same cash and run the same economy, so what they could have spent is what we
        # could have: our income, scaled by how many harvesters they were seen to field against ours. Their
        # infrastructure is what we have seen of it, and never less than most of what ours is (they need power,
        # production, a refinery and tech too). The share of the remainder that became army is our own share
        # (:meth:`_my_efficiency`): it is measured, not assumed. Our kills are army they no longer have.
        ratio = 1.0
        if self.harvesters_seen:
            ratio = max(0.5, min(1.5, self.harvesters_seen / self._my_harvesters))
        budget = START_CASH + self.income_cum * ratio
        infra = max(self._infrastructure(), 0.85 * self._my_infra)
        harvs = UNIT_COST_HARVESTER * max(self.harvesters_seen, self._my_harvesters, 2)
        defences = sum(
            self.cost_of(s.kind) for s in self.buildings.values() if s.kind in self.tower_types
        )
        killed = (self._counters or (0,))[0]
        free = max(0.0, budget - infra - harvs - defences)
        return max(0.0, free * self._my_efficiency() - killed)

    def _base_in_view(self, game_s: float) -> bool:
        return self.enemy_base is not None and game_s - self._base_view_at <= 10.0

    def belief(self, game_s: float) -> Belief:
        visible_units = {i: s for i, s in self.units.items() if s.kind not in self.non_combat}
        vis = {i: s for i, s in visible_units.items() if game_s - s.last <= 6.0}
        gone = {i: s for i, s in visible_units.items() if i not in vis and game_s - s.last <= 120.0}
        visible_value = sum(self.cost_of(s.kind) for s in vis.values())
        remembered = sum(self.cost_of(s.kind) * 0.6 for s in gone.values())
        sighted = visible_value + remembered
        economic = self._raw_army_estimate(game_s) * self._bias
        since = game_s - self.last_combat_seen if self.last_combat_seen > -1e8 else game_s
        # What is in view is a floor, never the whole: units we cannot see are still there. Where we can see their
        # base the economy's estimate counts for less (we would see the army if it were home), elsewhere in full.
        cover = 0.7 if self._base_in_view(game_s) else 0.0
        hidden = max(0.0, economic - sighted) * (1.0 - cover)
        army = sighted + hidden
        if game_s < 150:  # the opening: nothing real can exist yet beyond what we have seen
            army = max(sighted, min(army, 1500.0))
        sigma = 0.25 + 0.4 * min(1.0, since / 120.0)
        income = self._income_now * (
            max(0.5, min(1.5, self.harvesters_seen / self._my_harvesters))
            if self.harvesters_seen
            else 1.0
        )
        towers: dict[str, int] = {}
        for s in self.buildings.values():
            if s.kind in self.tower_types:
                towers[s.kind] = towers.get(s.kind, 0) + 1
        comp: dict[str, int] = {}
        for s in [*vis.values(), *gone.values()]:
            comp[s.kind] = comp.get(s.kind, 0) + 1
        return Belief(
            visible_value=visible_value,
            remembered_value=remembered,
            army_value=army,
            low=max(sighted, army * (1 - sigma)),
            high=army * (1 + sigma) + 400,
            hidden_value=max(0.0, army - sighted),
            income=income,
            since_seen_s=since,
            towers=towers,
            composition=comp,
        )

    # -- the economy ------------------------------------------------------------------------------------------

    def harvester_yield(self, window_s: float = 60.0) -> tuple[float, float] | None:
        """(credits per second per harvester over the recent window, the best such figure seen so far).

        Income is integrated, not read off the instantaneous meter, so refinery drop-offs do not whipsaw it.
        ``None`` until there is a window of data."""
        snaps = list(self.standing)
        if len(snaps) < 3:
            return None
        end = snaps[-1]
        start = next((x for x in snaps if end["t"] - x["t"] <= window_s), snaps[0])
        span = end["t"] - start["t"]
        if span < 30:
            return None
        harv = [x["harvesters"] for x in snaps if start["t"] <= x["t"] <= end["t"]]
        mean_h = sum(harv) / len(harv) if harv else 0
        if mean_h < 1:
            return None
        recent = (end["cum"] - start["cum"]) / span / mean_h
        best = recent
        for i in range(1, len(snaps)):
            a, z = snaps[i - 1], snaps[i]
            dt = z["t"] - a["t"]
            if dt > 0 and a["harvesters"] >= 1 and a["t"] >= 150:
                best = max(best, min(60.0, (z["cum"] - a["cum"]) / dt / a["harvesters"]) * 0.8)
        return recent, best

    def harvester_extra(
        self,
        game_s: float,
        remaining_s: float | None,
        harvesters: int,
        refineries: int,
        cost: float,
        under_threat: bool,
        cap: int,
    ) -> int:
        """How many harvesters beyond the standing target are worth buying: an investment, judged on its payback.

        A harvester costs ``cost`` and earns what the ones we have are earning now. It is worth buying when that
        pays back well inside the time left, the field is not already saturated (the yield per harvester has not
        fallen away as the count grew), the refineries can unload it, and the base is not under attack. The enemy
        out-harvesting us (more harvesters seen than we own) is evidence the field supports more."""
        if under_threat or harvesters < 2 or refineries < 1 or harvesters >= cap:
            return 0
        y = self.harvester_yield()
        if y is None:
            return 0
        recent, best = y
        if recent <= 1.0:
            return 0
        left = (remaining_s if isinstance(remaining_s, (int, float)) else max(0.0, 1200.0 - game_s)) - 90.0
        payback = cost / recent
        saturated = recent < 0.6 * best
        room = harvesters < 3 * refineries
        extra = 0
        if payback <= left / 2.5 and not saturated and room:
            extra = 1
            if payback <= left / 4.0 and self.harvesters_seen > harvesters:
                extra = 2
        return min(extra, cap - harvesters)

    # -- plans ------------------------------------------------------------------------------------------------

    def _plan_state(self, obs: dict[str, Any]) -> dict[str, float]:
        m = obs.get("military") or {}
        return {
            "kills": float(m.get("kills_cost") or 0),
            "losses": float(m.get("deaths_cost") or 0),
            "army": float(m.get("army_value") or 0),
            "income": self._income_now,
        }

    def note_plan(
        self, version: int, game_s: float, assessment: str, missions: Any, obs: dict[str, Any]
    ) -> None:
        if version == self._plan_version:
            return
        now_state = self._plan_state(obs)
        if self.plans and self.plans[-1].outcome is None:
            last = self.plans[-1]
            last.outcome = {
                "secs": round(game_s - last.t),
                "exchange_gained": round(
                    (now_state["kills"] - last.start["kills"])
                    - (now_state["losses"] - last.start["losses"])
                ),
                "army_change": round(now_state["army"] - last.start["army"]),
                "income_change": round(now_state["income"] - last.start["income"], 1),
            }
        self._plan_version = version
        self.plans.append(PlanRecord(version, game_s, str(assessment)[:300], missions, now_state))

    # -- what the General reads -------------------------------------------------------------------------------

    def briefing(
        self, game_s: float, obs: dict[str, Any], remaining_s: float | None
    ) -> dict[str, Any]:
        b = self.belief(game_s)
        m = obs.get("military") or {}
        kills, losses = float(m.get("kills_cost") or 0), float(m.get("deaths_cost") or 0)
        e_buildings: dict[str, int] = {}
        for s in self.buildings.values():
            e_buildings[s.kind] = e_buildings.get(s.kind, 0) + 1
        prod = {k: n for k, n in e_buildings.items() if k in PRODUCTION}
        tech = sorted(k for k in e_buildings if k in TECH)
        last_seen = max(
            (s.last for s in self.units.values() if s.kind not in self.non_combat), default=None
        )
        where = None
        if last_seen is not None:
            cells = [
                s.cell
                for s in self.units.values()
                if s.kind not in self.non_combat and s.last == last_seen and s.cell
            ]
            c = _centroid(cells)
            where = [round(c[0]), round(c[1])] if c else None
        fights = [f.as_brief() for f in list(self.fights)[-6:]]
        if self._open:
            fights.append({**self._open.as_brief(), "result": "in progress"})
        recent = self.realised_exchange(120.0, game_s)
        standing = list(self.standing)
        trend = None
        if len(standing) >= 5:
            a, z = standing[-5], standing[-1]
            trend = {
                "over_s": z["t"] - a["t"],
                "my_army_change": round(z["my_army"] - a["my_army"]),
                "enemy_army_estimate_change": round(z["enemy_army_est"] - a["enemy_army_est"]),
                "income_change": round(z["income"] - a["income"], 1),
            }
        return {
            "about": "your own long-run memory of this match: read it before every plan",
            "clock": {
                "game_s": round(game_s),
                "remaining_s": round(remaining_s)
                if isinstance(remaining_s, (int, float))
                else None,
                "trade_balance": round(kills - losses),
                "our_kills_value": round(kills),
                "our_losses_value": round(losses),
                "if_the_clock_ran_out_now": "ahead on trades"
                if kills > losses
                else "behind on trades"
                if kills < losses
                else "level on trades",
            },
            "enemy": {
                "army_belief_value": round(b.army_value),
                "army_belief_range": [round(b.low), round(b.high)],
                "in_view_now": round(b.visible_value),
                "seen_recently_out_of_view": round(b.remembered_value),
                "unseen_estimate": round(b.hidden_value),
                "seconds_since_any_army_unit_in_view": round(b.since_seen_s),
                "last_seen_army_at": where,
                "peak_army_ever_seen": {"value": round(self.peak_value), "by_type": self.peak_army},
                "likely_composition": b.composition,
                "income_estimate_per_s": round(b.income, 1),
                "harvesters_seen": self.harvesters_seen,
                "base_cell": [round(self.enemy_base[0]), round(self.enemy_base[1])]
                if self.enemy_base
                else None,
                "buildings_known": e_buildings,
                "production_structures": prod,
                "tech_seen": tech,
                "defences_known": b.towers,
            },
            "fights_so_far": {
                "recent": fights,
                "exchange_last_2min": round(recent, 2) if recent else None,
                "model_calibration": round(self.exchange_calibration, 2),
                "total": len(self.fights),
            },
            "plans_so_far": [
                {"v": p.version, "t": round(p.t), "assessment": p.assessment, "result": p.outcome}
                for p in list(self.plans)[-5:]
            ],
            "trend": trend,
            "our_economy": self.economy_picture(obs, game_s),
        }
