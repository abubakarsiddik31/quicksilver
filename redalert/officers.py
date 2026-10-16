"""Officers and missions: the General (the strategist) commands groups, not a single "army mode".

The strategist assigns each officer a *mission* (a kind, a share of the army, a target and a few judgment
parameters). The :class:`Fleet` keeps each officer's soldiers sticky across ticks and turns mission plus live
observation into orders, so the model's seconds-long latency commands manoeuvres that the executor carries out
at five decisions a second.

Missions:

* ``HOLD_LINE`` - the home guard: engages in range on ``defend`` stance, meets anything that comes near the
  base, never chases.
* ``STRIKE`` - the main body: musters, advances together when the local odds are good, fights, and breaks off to
  a fall-back when the enemy around it outweighs it; regroups, then comes again.
* ``FLANK`` - a detachment that swings wide of the target and hits from the side, then withdraws.
* ``RAID`` - a fast group that goes for an economic target (harvesters, refinery, power) and withdraws when
  something bigger arrives.
* ``SCREEN`` - a forward picket that gives vision and falls back when the enemy comes, so the army sees it coming.
* ``FALL_BACK`` - retreat to the tower line (or the base), fighting while it goes.

What the code owns is mechanics, not strategy: units fight on attack-move (a unit on a plain move does not
shoot), a retreat in contact keeps firing, soldiers do not chase past their leash, and every officer reports its
state back so the strategist can react to what actually happened. One invariant stays: the base is never left
without a guard while there is an army to guard it.

Pure over plain data (units and enemies are the observation's dicts), so every decision is unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

HOLD_LINE, STRIKE, FLANK, RAID, SCREEN, FALL_BACK = (
    "HOLD_LINE",
    "STRIKE",
    "FLANK",
    "RAID",
    "SCREEN",
    "FALL_BACK",
)
KINDS = (HOLD_LINE, STRIKE, FLANK, RAID, SCREEN, FALL_BACK)
SIDES = ("left", "right", "alternate")
MAX_MISSIONS = 6

# Judgment parameters live on the mission; these are only the defaults when the strategist leaves them out.
DEFAULT_ENGAGE_RATIO = (
    1.0  # STRIKE advances when its value is at least this multiple of the enemy near the target
)
DEFAULT_ABORT_RATIO = 0.7  # ...and breaks off in contact below this multiple of the enemy around it
DEFAULT_REGROUP_S = 30.0  # how long a broken group stays back before it may advance again
MIN_ADVANCE_UNITS = 5  # a strike advances with at least this many bodies: one rifleman 'advancing' at 137 s died alone
BLIND_ADVANCE_MIN = 9000.0  # a strike that has never seen what guards its target advances blind only when this strong
SIGHTING_MEMORY_S = 150.0  # a sighting at the target still counts this long after the scout lost sight of it
DEFAULT_RAID_S = 50.0  # a flank or raid run ends this long after it leaves
DEFAULT_LEASH = 22.0  # how far from home a screen or the line stands
LOCAL_RADIUS = 16.0  # enemies this close to a group count as the fight it is in
CONTACT_RADIUS = 10.0
TARGET_RADIUS = 14.0  # enemy strength "at the target" is what stands within this of it
LINE_THREAT_RADIUS = 26.0  # enemy units this close to the base are met by the home line
FLANK_OFFSET = 14.0
MIN_GUARD = 2  # the one invariant: the base always keeps a guard
MIN_ARMY_FOR_GUARD = 5
ORDER_GAP_S = 2.5
ASSEMBLED_SHARE = 0.7
ASSEMBLE_RADIUS = 12.0
MIN_GROUP = 3  # below this a group retreats instead of judging a fight
GUARD_STANCE = "defend"
FIGHT_STANCE = "attack_anything"

ECONOMIC_TARGETS = {
    "harvesters": {"harv"},
    "economy": {"harv", "proc", "powr", "apwr"},
    "power": {"powr", "apwr"},
    "refinery": {"proc"},
    "production": {"weap", "tent", "barr", "afld", "fact"},
}


@dataclass(frozen=True)
class Mission:
    id: str  # the officer's name: stable across plans, so a group keeps its soldiers when the plan is restated
    kind: str
    share: float = 0.0  # fraction of the fighters; STRIKE with no share takes whoever is left
    count: int | None = None  # an exact number instead of a share
    target: Any = (
        "nearest"  # "nearest" | "base" | "building" | an ECONOMIC_TARGETS key | "towers" | [x, y]
    )
    side: str = "alternate"
    engage_ratio: float = DEFAULT_ENGAGE_RATIO
    abort_ratio: float = DEFAULT_ABORT_RATIO
    regroup_s: float = DEFAULT_REGROUP_S
    duration_s: float = DEFAULT_RAID_S
    leash: float = DEFAULT_LEASH
    note: str = ""

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "note": self.note,
        }
        if self.count is not None:
            out["count"] = self.count
        else:
            out["share"] = self.share
        out.update(
            side=self.side,
            engage_ratio=self.engage_ratio,
            abort_ratio=self.abort_ratio,
            regroup_s=self.regroup_s,
            duration_s=self.duration_s,
            leash=self.leash,
        )
        return out


def _num(value: Any, low: float, high: float, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(low, min(high, float(value)))


def parse_missions(value: Any) -> tuple[Mission, ...]:
    """The strategist's ``missions`` list, validated: unknown kinds and malformed rows are dropped, never raised."""
    if not isinstance(value, list):
        return ()
    out: list[Mission] = []
    seen: set[str] = set()
    for row in value[:MAX_MISSIONS]:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", "")).upper().replace("-", "_").strip()
        if kind not in KINDS:
            continue
        mid = str(row.get("id") or kind.lower())[:24] or kind.lower()
        if mid in seen:
            mid = f"{mid}{len(seen)}"
        seen.add(mid)
        target = row.get("target", "nearest")
        if (
            isinstance(target, (list, tuple))
            and len(target) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in target)
        ):
            target = [int(target[0]), int(target[1])]
        elif not isinstance(target, str):
            target = "nearest"
        count = row.get("count")
        count = (
            int(count)
            if isinstance(count, int) and not isinstance(count, bool) and count > 0
            else None
        )
        side = row.get("side") if row.get("side") in SIDES else "alternate"
        out.append(
            Mission(
                id=mid,
                kind=kind,
                share=_num(row.get("share"), 0.0, 1.0, 0.0),
                count=count,
                target=target,
                side=side,
                engage_ratio=_num(row.get("engage_ratio"), 0.0, 10.0, DEFAULT_ENGAGE_RATIO),
                abort_ratio=_num(row.get("abort_ratio"), 0.0, 5.0, DEFAULT_ABORT_RATIO),
                regroup_s=_num(row.get("regroup_s"), 5.0, 180.0, DEFAULT_REGROUP_S),
                duration_s=_num(row.get("duration_s"), 15.0, 180.0, DEFAULT_RAID_S),
                leash=_num(row.get("leash"), 6.0, 60.0, DEFAULT_LEASH),
                note=str(row.get("note", ""))[:160],
            )
        )
    return tuple(out)


def default_missions(*, hold: bool = False) -> tuple[Mission, ...]:
    """What the legacy army fields amount to: a home line and one main body (all in on the base when attacking)."""
    line = Mission("line", HOLD_LINE, share=0.25)
    if hold:
        return (Mission("line", HOLD_LINE, share=1.0),)
    return (line, Mission("strike", STRIKE, target="nearest"))


# -- geometry and the picture of the fight ------------------------------------------------------------------------


def _cell(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(v, (int, float)) for v in value[:2])
    ):
        return float(value[0]), float(value[1])
    if (
        isinstance(value, dict)
        and isinstance(value.get("x"), (int, float))
        and isinstance(value.get("y"), (int, float))
    ):
        return float(value["x"]), float(value["y"])
    return None


def _center(cells: Iterable[tuple[float, float]]) -> tuple[float, float] | None:
    pts = list(cells)
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _toward(a: tuple[float, float], b: tuple[float, float], fraction: float) -> list[int]:
    return [int(a[0] + (b[0] - a[0]) * fraction), int(a[1] + (b[1] - a[1]) * fraction)]


def _clamp(cell: Any, bounds: Any) -> list[int] | None:
    if cell is None:
        return None
    x, y = int(cell[0]), int(cell[1])
    if isinstance(bounds, list) and len(bounds) == 4:
        bx, by, bw, bh = bounds
        x, y = max(bx, min(bx + bw - 1, x)), max(by, min(by + bh - 1, y))
    return [x, y]


@dataclass
class Picture:
    """The enemy as the officers see it this tick."""

    units: list[tuple[tuple[float, float], str, float, Any]] = field(
        default_factory=list
    )  # (cell, type, cost, id)
    buildings: list[tuple[tuple[float, float], str, float, Any]] = field(default_factory=list)
    towers: set[str] = field(default_factory=set)

    def near(
        self, point: tuple[float, float] | None, radius: float, *, with_towers: bool = True
    ) -> float:
        """Enemy value (credits) within ``radius`` of a point: units, plus defensive structures."""
        if point is None:
            return 0.0
        total = sum(c for cell, _, c, _ in self.units if math.dist(cell, point) <= radius)
        if with_towers:
            total += sum(
                c
                for cell, kind, c, _ in self.buildings
                if kind in self.towers and math.dist(cell, point) <= radius
            )
        return total

    def nearest(
        self, point: tuple[float, float], kinds: set[str] | None = None, *, buildings: bool = False
    ):
        pool = self.buildings if buildings else self.units
        best = None
        for cell, kind, _, ident in pool:
            if kinds is not None and kind not in kinds:
                continue
            d = math.dist(cell, point)
            if best is None or d < best[0]:
                best = (d, cell, kind, ident)
        return best

    def any_near(self, cells: list[tuple[float, float]], radius: float) -> bool:
        return any(math.dist(c, cell) <= radius for c in cells for cell, _, _, _ in self.units)


@dataclass
class Report:
    id: str
    kind: str
    n: int
    value: float
    center: list[int] | None
    state: str
    enemy_near: float
    ratio: float | None
    target: Any

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "soldiers": self.n,
            "value": round(self.value),
            "at": self.center,
            "state": self.state,
            "enemy_near_value": round(self.enemy_near),
            "odds": None if self.ratio is None else round(self.ratio, 2),
            "target": self.target,
        }


@dataclass
class Result:
    orders: list[dict[str, Any]] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class Fleet:
    """The standing officers: who belongs to whom, what state each is in, and the orders this tick."""

    def __init__(
        self,
        cost_of: Callable[[str], float],
        clock: Callable[[], float],
        *,
        non_combat: frozenset[str] = frozenset(),
        tower_types: frozenset[str] = frozenset(),
    ) -> None:
        self.cost_of = cost_of
        self.clock = clock
        self.non_combat = non_combat
        self.tower_types = tower_types
        self.members: dict[str, set[int]] = {}
        self.state: dict[str, str] = {}
        self._since: dict[str, float] = {}
        self._until: dict[str, float] = {}  # broken / run-ends-at timers
        self._target_seen: tuple[float, float] | None = None  # (enemy value seen at a strike target, when)
        self._last_order: dict[str, float] = {}
        self._stance: dict[str, str] = {}
        self._stanced: dict[str, set[int]] = {}
        self._side_flip = 1
        self._run_side: dict[str, int] = {}
        self._engaged_reported: set[str] = set()

    # -- the roster -------------------------------------------------------------------------------------------

    def assign(
        self,
        missions: tuple[Mission, ...],
        fighters: list[dict[str, Any]],
        home: tuple[float, float] | None,
    ):
        """Give every fighter to exactly one officer; membership sticks while the officer's quota holds."""
        alive = {u["id"]: u for u in fighters if isinstance(u.get("id"), int)}
        n = len(alive)
        missions = list(missions)
        guard_needed = n >= MIN_ARMY_FOR_GUARD
        line = next((m for m in missions if m.kind == HOLD_LINE), None)
        if guard_needed and line is None:
            missions.insert(0, Mission("line", HOLD_LINE, count=MIN_GUARD))
        order = {HOLD_LINE: 0, SCREEN: 1, RAID: 2, FLANK: 3, FALL_BACK: 4, STRIKE: 5}
        ranked = sorted(missions, key=lambda m: order[m.kind])

        quotas: dict[str, int] = {}
        left = n
        for m in ranked:
            if m.kind == STRIKE and m.count is None and m.share == 0.0:
                continue
            want = m.count if m.count is not None else round(m.share * n)
            if m.kind == HOLD_LINE and guard_needed:
                want = max(want, MIN_GUARD)
            quotas[m.id] = max(0, min(left, want))
            left -= quotas[m.id]
        strike = next((m for m in ranked if m.kind == STRIKE and m.id not in quotas), None)
        if strike is not None:
            quotas[strike.id] = left
            left = 0

        # Stale officers disappear; survivors keep their soldiers.
        names = {m.id for m in ranked}
        for gone in [k for k in self.members if k not in names]:
            self.members.pop(gone, None)
            self.state.pop(gone, None)
        taken: set[int] = set()
        for m in ranked:
            kept = [u for u in self.members.get(m.id, set()) if u in alive and u not in taken]
            kept = kept[: quotas.get(m.id, 0)]
            self.members[m.id] = set(kept)
            taken.update(kept)
        free = [u for u in alive if u not in taken]
        by_cost = lambda u: self.cost_of(str(alive[u].get("type")).lower())  # noqa: E731
        for m in ranked:
            need = quotas.get(m.id, 0) - len(self.members[m.id])
            if need <= 0 or not free:
                continue
            if m.kind == HOLD_LINE and home is not None:
                pool = sorted(free, key=lambda u: math.dist(_cell(alive[u]) or home, home))
            elif m.kind in (RAID, FLANK, SCREEN):
                pool = sorted(
                    free, key=by_cost
                )  # the cheapest and fastest go out; the heavy units hold the front
            else:
                pool = sorted(free, key=lambda u: -by_cost(u))
            for u in pool[:need]:
                self.members[m.id].add(u)
                taken.add(u)
            free = [u for u in free if u not in taken]
        if free:  # shares that do not add up to the army: the rest stands with the home line
            anchor = next(
                (m.id for m in ranked if m.kind == HOLD_LINE), ranked[0].id if ranked else None
            )
            if anchor is not None:
                self.members.setdefault(anchor, set()).update(free)
        return ranked

    # -- the picture ------------------------------------------------------------------------------------------

    def picture(self, obs: dict[str, Any]) -> Picture:
        pic = Picture(towers=set(self.tower_types))
        for key, dest in (
            ("visible_enemies", pic.units),
            ("visible_enemy_buildings", pic.buildings),
        ):
            for e in obs.get(key) or []:
                if not isinstance(e, dict) or e.get("wreck"):
                    continue
                cell = _cell(e)
                kind = str(e.get("type")).lower()
                if cell is None or (
                    key == "visible_enemies" and kind in self.non_combat and kind != "harv"
                ):
                    continue
                dest.append((cell, kind, float(self.cost_of(kind)), e.get("id")))
        return pic

    # -- one tick ---------------------------------------------------------------------------------------------

    def step(
        self,
        missions: tuple[Mission, ...],
        fighters: list[dict[str, Any]],
        obs: dict[str, Any],
        *,
        home: tuple[float, float] | None,
        enemy_base: tuple[float, float] | None,
        bounds: Any = None,
        own_towers: list[tuple[float, float]] | None = None,
    ) -> Result:
        res = Result()
        now = self.clock()
        if not fighters or home is None:
            return res
        ranked = self.assign(missions, fighters, home)
        units = {u["id"]: u for u in fighters if isinstance(u.get("id"), int)}
        pic = self.picture(obs)
        ctx = _Ctx(self, pic, units, home, enemy_base, bounds, own_towers or [], now)
        for m in ranked:
            ids = sorted(self.members.get(m.id, ()))
            if not ids:
                continue
            handler = {
                HOLD_LINE: self._hold_line,
                STRIKE: self._strike,
                FLANK: self._run,
                RAID: self._run,
                SCREEN: self._screen,
                FALL_BACK: self._fall_back,
            }[m.kind]
            handler(m, ids, ctx, res)
        return res

    # -- helpers shared by the missions -----------------------------------------------------------------------

    def _set(
        self, mid: str, state: str, now: float, res: Result, *, event: str | None = None
    ) -> None:
        if self.state.get(mid) != state:
            self.state[mid] = state
            self._since[mid] = now
            if event:
                res.events.append(f"{event}:{mid}")

    def _stance_order(self, mid: str, ids: list[int], stance: str, res: Result) -> None:
        had = self._stanced.setdefault(mid, set())
        if self._stance.get(mid) != stance:
            had.clear()
            self._stance[mid] = stance
        new = [u for u in ids if u not in had]
        if new:
            had.update(new)
            res.orders.append({"cmd": "set_stance", "units": new[:50], "stance": stance})

    def _go(
        self,
        mid: str,
        ids: list[int],
        to: Any,
        ctx: _Ctx,
        res: Result,
        *,
        fight: bool,
        gap: float = ORDER_GAP_S,
    ) -> None:
        """Move an officer's soldiers. ``fight`` means attack-move: soldiers stop and shoot what they meet; a plain
        move is only for a walk with no enemy in contact."""
        if to is None or ctx.now - self._last_order.get(mid, -1e9) < gap:
            return
        self._last_order[mid] = ctx.now
        for i in range(0, len(ids), 50):
            res.orders.append(
                {
                    "cmd": "attack_move" if fight else "move",
                    "units": ids[i : i + 50],
                    "to": list(to),
                }
            )

    def _in_contact(self, ids: list[int], ctx: _Ctx) -> bool:
        cells = [c for c in (_cell(ctx.units.get(u)) for u in ids) if c]
        return ctx.pic.any_near(cells, CONTACT_RADIUS)

    def _value(self, ids: list[int], ctx: _Ctx) -> float:
        return sum(
            self.cost_of(str(ctx.units[u].get("type")).lower()) for u in ids if u in ctx.units
        )

    def _center(self, ids: list[int], ctx: _Ctx) -> tuple[float, float] | None:
        return _center(c for c in (_cell(ctx.units.get(u)) for u in ids) if c)

    def _report(
        self,
        m: Mission,
        ids: list[int],
        ctx: _Ctx,
        res: Result,
        *,
        near: float,
        ratio: float | None,
        target: Any,
    ) -> None:
        c = self._center(ids, ctx)
        res.reports.append(
            Report(
                m.id,
                m.kind,
                len(ids),
                self._value(ids, ctx),
                [int(c[0]), int(c[1])] if c else None,
                self.state.get(m.id, "ready"),
                near,
                ratio,
                target,
            )
        )

    def resolve_target(
        self, target: Any, center: tuple[float, float] | None, ctx: _Ctx
    ) -> tuple[float, float] | None:
        """A mission's target as a cell: a point, the enemy base, the nearest of something, or an economic target."""
        cell = _cell(target) if not isinstance(target, str) else None
        if cell:
            return cell
        anchor = center or ctx.home
        if target == "base":
            return ctx.enemy_base or anchor
        if target == "building":
            hit = ctx.pic.nearest(anchor, buildings=True)
            return hit[1] if hit else ctx.enemy_base
        if target in ECONOMIC_TARGETS:
            kinds = ECONOMIC_TARGETS[target]
            hit = ctx.pic.nearest(anchor, kinds) or ctx.pic.nearest(anchor, kinds, buildings=True)
            return hit[1] if hit else ctx.enemy_base
        if target == "enemy_army":
            c = _center(cell for cell, _, _, _ in ctx.pic.units)
            return c or ctx.enemy_base
        hit = ctx.pic.nearest(anchor)  # "nearest"
        return hit[1] if hit else ctx.enemy_base

    def fall_back_point(
        self, ctx: _Ctx, center: tuple[float, float] | None, target: Any
    ) -> tuple[float, float]:
        cell = _cell(target) if not isinstance(target, str) else None
        if cell:
            return cell
        anchor = center or ctx.home
        if ctx.own_towers and target != "base":
            return min(ctx.own_towers, key=lambda t: math.dist(t, anchor))
        return ctx.home

    # -- the missions -----------------------------------------------------------------------------------------

    def _hold_line(self, m: Mission, ids: list[int], ctx: _Ctx, res: Result) -> None:
        threats = [
            c for c, _, _, _ in ctx.pic.units if math.dist(c, ctx.home) <= LINE_THREAT_RADIUS
        ]
        if threats:
            point = _center(threats)
            self._stance_order(m.id, ids, FIGHT_STANCE, res)
            self._set(m.id, "engaged", ctx.now, res)
            self._go(m.id, ids, [int(point[0]), int(point[1])], ctx, res, fight=True, gap=1.5)
            self._report(
                m,
                ids,
                ctx,
                res,
                near=ctx.pic.near(point, LOCAL_RADIUS, with_towers=False),
                ratio=None,
                target=[int(point[0]), int(point[1])],
            )
            return
        self._stance_order(m.id, ids, GUARD_STANCE, res)
        toward = ctx.enemy_base or ctx.home
        d = math.dist(ctx.home, toward)
        line = _clamp(
            _toward(ctx.home, toward, min(10.0 / d, 0.35)) if d > 1 else ctx.home, ctx.bounds
        )
        center = self._center(ids, ctx)
        self._set(m.id, "holding", ctx.now, res)
        if line and center and math.dist(center, line) > 6:
            self._go(m.id, ids, line, ctx, res, fight=self._in_contact(ids, ctx), gap=4.0)
        self._report(m, ids, ctx, res, near=0.0, ratio=None, target=line)

    def _strike(self, m: Mission, ids: list[int], ctx: _Ctx, res: Result) -> None:
        mid, now = m.id, ctx.now
        center = self._center(ids, ctx)
        value = self._value(ids, ctx)
        target = self.resolve_target(m.target, center, ctx)
        near_me = ctx.pic.near(center, LOCAL_RADIUS)
        near_target = ctx.pic.near(target, TARGET_RADIUS) if target else 0.0
        in_contact = self._in_contact(ids, ctx)

        # Recovering from a broken fight: fall back fighting, hold there, wait out the regroup.
        if now < self._until.get(mid, 0.0):
            dest = self.fall_back_point(ctx, center, "towers")
            self._stance_order(mid, ids, FIGHT_STANCE, res)
            self._set(mid, "broken", now, res)
            if center is None or math.dist(center, dest) > 6 or in_contact:
                self._go(mid, ids, [int(dest[0]), int(dest[1])], ctx, res, fight=in_contact)
            self._report(
                m,
                ids,
                ctx,
                res,
                near=near_me,
                ratio=(value / near_me) if near_me else None,
                target=target,
            )
            return

        # In a fight: keep going while the odds hold, break off when the enemy around us outweighs us.
        if in_contact:
            ratio = value / near_me if near_me > 0 else None
            if len(ids) >= MIN_GROUP and ratio is not None and ratio < m.abort_ratio:
                self._until[mid] = now + m.regroup_s
                self._set(mid, "broken", now, res, event="officer_broken")
                dest = self.fall_back_point(ctx, center, "towers")
                self._stance_order(mid, ids, FIGHT_STANCE, res)
                self._last_order.pop(mid, None)
                self._go(mid, ids, [int(dest[0]), int(dest[1])], ctx, res, fight=True)
                self._report(m, ids, ctx, res, near=near_me, ratio=ratio, target=target)
                return
            self._stance_order(mid, ids, FIGHT_STANCE, res)
            self._set(mid, "engaged", now, res, event="officer_engaged")
            self._go(
                mid, ids, [int(target[0]), int(target[1])] if target else None, ctx, res, fight=True
            )
            self._report(m, ids, ctx, res, near=near_me, ratio=ratio, target=target)
            return

        # Not in contact: muster, then advance together when the odds at the target are good.
        staging = None
        if target is not None:
            d = math.dist(ctx.home, target)
            staging = _clamp(_toward(ctx.home, target, min(0.2, 24.0 / max(d, 1.0))), ctx.bounds)
        assembled = False
        if staging and center:
            near_stage = sum(
                1
                for u in ids
                if (c := _cell(ctx.units.get(u))) and math.dist(c, staging) <= ASSEMBLE_RADIUS
            )
            assembled = near_stage >= ASSEMBLED_SHARE * len(ids) or self.state.get(mid) in (
                "advancing",
                "engaged",
            )
        if near_target > 0:
            self._target_seen = (near_target, now)
        elif self._target_seen and now - self._target_seen[1] <= SIGHTING_MEMORY_S:
            near_target = 0.85 * self._target_seen[0]  # out of sight is not gone: the scout saw an army here a moment ago
        ratio = (value / near_target) if near_target > 0 else None
        # Nothing seen at the target is not an empty target: a live match walked eight riflemen into an unseen
        # army (16 lost for one kill). A strike that never saw the defenders waits for scouting or for a big army.
        odds_ok = value >= m.engage_ratio * near_target if near_target > 0 else value >= BLIND_ADVANCE_MIN
        odds_ok = odds_ok and len(ids) >= MIN_ADVANCE_UNITS
        self._stance_order(mid, ids, FIGHT_STANCE, res)
        if assembled and odds_ok and target is not None:
            self._set(mid, "advancing", now, res)
            self._go(mid, ids, [int(target[0]), int(target[1])], ctx, res, fight=True)
        elif staging:
            self._set(mid, "mustering" if not assembled else "waiting", now, res)
            if center is None or math.dist(center, staging) > ASSEMBLE_RADIUS * 0.6:
                self._go(mid, ids, staging, ctx, res, fight=False, gap=4.0)
        self._report(m, ids, ctx, res, near=near_target, ratio=ratio, target=target)

    def _run(self, m: Mission, ids: list[int], ctx: _Ctx, res: Result) -> None:
        """FLANK and RAID: swing out, hit, withdraw. A run ends on its timer, on a bigger enemy, or on the kill."""
        mid, now = m.id, ctx.now
        center = self._center(ids, ctx)
        value = self._value(ids, ctx)
        goal = (
            m.target if m.target != "nearest" else ("harvesters" if m.kind == RAID else "building")
        )
        target = self.resolve_target(goal, center, ctx)
        near_me = ctx.pic.near(center, LOCAL_RADIUS)
        state = self.state.get(mid, "ready")
        in_contact = self._in_contact(ids, ctx)
        home_back = self.fall_back_point(ctx, center, "base")

        if state in ("withdrawing", "ready") and now < self._until.get(mid, 0.0):
            # Cooling down at home after a run.
            self._stance_order(mid, ids, GUARD_STANCE, res)
            if center and math.dist(center, home_back) > 8 or in_contact:
                self._go(
                    mid, ids, [int(home_back[0]), int(home_back[1])], ctx, res, fight=in_contact
                )
            self._report(m, ids, ctx, res, near=near_me, ratio=None, target=target)
            return

        if state not in ("swinging", "hitting"):
            if target is None:
                self._report(m, ids, ctx, res, near=near_me, ratio=None, target=None)
                return
            if m.side == "alternate":
                self._side_flip = -self._side_flip
                side = self._side_flip
            else:
                side = 1 if m.side == "left" else -1
            self._run_side[mid] = side
            self._until[mid] = now + m.duration_s
            self._set(mid, "swinging", now, res, event=None)
            state = "swinging"

        out_of_time = now >= self._until.get(mid, 0.0)
        bigger = len(ids) >= 2 and near_me >= max(value, 1.0) * 1.0 and in_contact
        if out_of_time or bigger:
            self._until[mid] = now + (m.regroup_s if bigger else m.regroup_s * 0.5)
            self._set(mid, "withdrawing", now, res, event="officer_withdrew" if bigger else None)
            self._stance_order(mid, ids, FIGHT_STANCE if in_contact else GUARD_STANCE, res)
            self._last_order.pop(mid, None)
            self._go(mid, ids, [int(home_back[0]), int(home_back[1])], ctx, res, fight=in_contact)
            self._report(
                m,
                ids,
                ctx,
                res,
                near=near_me,
                ratio=(value / near_me) if near_me else None,
                target=target,
            )
            return

        if target is None:
            self._report(m, ids, ctx, res, near=near_me, ratio=None, target=None)
            return
        if state == "swinging":
            d = math.dist(ctx.home, target) or 1.0
            px, py = -(target[1] - ctx.home[1]) / d, (target[0] - ctx.home[0]) / d
            side = self._run_side.get(mid, 1)
            wp = _clamp(
                [target[0] + px * side * FLANK_OFFSET, target[1] + py * side * FLANK_OFFSET],
                ctx.bounds,
            )
            self._stance_order(mid, ids, GUARD_STANCE, res)
            if m.kind == RAID:
                wp = None  # a raid goes straight at the economy
            if wp and center and math.dist(center, wp) > 8:
                self._go(mid, ids, wp, ctx, res, fight=True)
            else:
                self._set(mid, "hitting", now, res)
                state = "hitting"
        if state == "hitting":
            self._stance_order(mid, ids, FIGHT_STANCE, res)
            self._go(mid, ids, [int(target[0]), int(target[1])], ctx, res, fight=True)
        self._report(
            m,
            ids,
            ctx,
            res,
            near=near_me,
            ratio=(value / near_me) if near_me else None,
            target=target,
        )

    def _screen(self, m: Mission, ids: list[int], ctx: _Ctx, res: Result) -> None:
        mid, now = m.id, ctx.now
        center = self._center(ids, ctx)
        value = self._value(ids, ctx)
        toward = (
            self.resolve_target(m.target if m.target != "nearest" else "base", None, ctx)
            or ctx.home
        )
        d = math.dist(ctx.home, toward) or 1.0
        post = _clamp(_toward(ctx.home, toward, min(m.leash / d, 0.9)), ctx.bounds)
        near_me = ctx.pic.near(center, LOCAL_RADIUS, with_towers=False)
        self._stance_order(mid, ids, GUARD_STANCE, res)
        if near_me > 0 and value < 1.5 * near_me:
            self._set(mid, "falling_back", now, res, event="officer_spotted")
            back = _clamp(_toward(ctx.home, toward, min(6.0 / d, 0.3)), ctx.bounds)
            self._go(mid, ids, back, ctx, res, fight=self._in_contact(ids, ctx))
        else:
            self._set(mid, "watching", now, res)
            if post and center and math.dist(center, post) > 6:
                self._go(mid, ids, post, ctx, res, fight=self._in_contact(ids, ctx), gap=4.0)
        self._report(
            m,
            ids,
            ctx,
            res,
            near=near_me,
            ratio=(value / near_me) if near_me else None,
            target=post,
        )

    def _fall_back(self, m: Mission, ids: list[int], ctx: _Ctx, res: Result) -> None:
        mid, now = m.id, ctx.now
        center = self._center(ids, ctx)
        dest = self.fall_back_point(ctx, center, m.target if m.target != "nearest" else "towers")
        in_contact = self._in_contact(ids, ctx)
        self._stance_order(mid, ids, FIGHT_STANCE, res)
        if center is None or math.dist(center, dest) > 6 or in_contact:
            self._set(mid, "falling_back", now, res)
            self._go(mid, ids, [int(dest[0]), int(dest[1])], ctx, res, fight=in_contact)
        else:
            self._set(mid, "holding", now, res)
        self._report(
            m,
            ids,
            ctx,
            res,
            near=ctx.pic.near(center, LOCAL_RADIUS),
            ratio=None,
            target=[int(dest[0]), int(dest[1])],
        )

    def summary(self, reports: list[Report]) -> list[dict[str, Any]]:
        return [r.as_json() for r in reports]


@dataclass
class _Ctx:
    fleet: Fleet
    pic: Picture
    units: dict[int, dict[str, Any]]
    home: tuple[float, float]
    enemy_base: tuple[float, float] | None
    bounds: Any
    own_towers: list[tuple[float, float]]
    now: float
