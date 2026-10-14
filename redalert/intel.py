"""Battle intelligence: the full picture of both sides for Jev's decisions.

The engine shows only what we can see right now (fog of war). This module keeps
a memory: every enemy unit seen in the last ``UNIT_MEMORY_S`` (its type, last
cell, health) and every enemy building seen that has not been destroyed, so an
army that walks out of sight is not forgotten. :meth:`Intel.summary` turns both
sides into plain numbers: forces by type and value, health, where each army is
relative to both bases, losses and kills.

Only engine data goes in (types, ids, cells, hp, costs), never player text.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

UNIT_MEMORY_S = 90.0
NEAR_BASE = 30  # cells: "at my base"
SIGHT = 4  # cells: well inside any unit's vision
NON_COMBAT = {"mcv", "harv", "medi", "mech", "e6", "spy", "thf", "truk"}
TOWERS = {"pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam", "gap", "iron", "pdox", "mslo"}


def _cell(v: Any) -> tuple[int, int] | None:
    if (
        isinstance(v, dict)
        and isinstance(v.get("x"), (int, float))
        and isinstance(v.get("y"), (int, float))
    ):
        return int(v["x"]), int(v["y"])
    if (
        isinstance(v, (list, tuple))
        and len(v) >= 2
        and all(isinstance(c, (int, float)) for c in v[:2])
    ):
        return int(v[0]), int(v[1])
    return None


def _hp(v: dict[str, Any]) -> float | None:
    hp = v.get("hp")
    return float(hp) if isinstance(hp, (int, float)) else None


def _centroid(cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not cells:
        return None
    return round(sum(c[0] for c in cells) / len(cells)), round(
        sum(c[1] for c in cells) / len(cells)
    )


class Intel:
    def __init__(self) -> None:
        self.units: dict[int, dict[str, Any]] = {}  # enemy unit id -> {type, cell, hp, seen}
        self.buildings: dict[
            int, dict[str, Any]
        ] = {}  # enemy building id -> {type, cell, hp, seen}
        self._seq = -1
        self.lost_recent: list[tuple[float, str]] = []  # (time, type) of our units/buildings lost
        self.killed_recent: list[tuple[float, str]] = []

    def update(self, obs: dict[str, Any], now: float) -> None:
        visible: set[int] = set()
        for e in obs.get("visible_enemies") or []:
            if isinstance(e, dict) and isinstance(e.get("id"), int):
                if e.get("wreck"):
                    self.units.pop(e["id"], None)
                    continue
                self.units[e["id"]] = {
                    "type": str(e.get("type")).lower(),
                    "cell": _cell(e),
                    "hp": _hp(e),
                    "seen": now,
                }
                visible.add(e["id"])
        for b in obs.get("visible_enemy_buildings") or []:
            if isinstance(b, dict) and isinstance(b.get("id"), int):
                if b.get("wreck"):
                    self.buildings.pop(b["id"], None)
                    continue
                self.buildings[b["id"]] = {
                    "type": str(b.get("type")).lower(),
                    "cell": _cell(b),
                    "hp": _hp(b),
                    "seen": now,
                }
        for ev in obs.get("events") or []:
            if (
                not isinstance(ev, dict)
                or not isinstance(ev.get("seq"), int)
                or ev["seq"] <= self._seq
            ):
                continue
            self._seq = ev["seq"]
            kind = str(ev.get("kind"))
            what = str(ev.get("type") or ev.get("actor_type") or "?").lower()
            if kind in ("unit_lost", "building_lost"):
                self.lost_recent.append((now, what))
            elif kind == "enemy_killed":
                n = ev.get("count") if isinstance(ev.get("count"), int) else 1
                self.killed_recent.extend((now, what) for _ in range(max(1, min(n, 50))))
        # A remembered enemy whose last cell one of ours is standing on, yet unseen, has moved on or died.
        ours = [
            c
            for c in (
                _cell(r)
                for r in (*(obs.get("units") or []), *(obs.get("buildings") or []))
                if isinstance(r, dict)
            )
            if c
        ]
        self.units = {
            i: u
            for i, u in self.units.items()
            if now - u["seen"] <= UNIT_MEMORY_S
            and (
                i in visible
                or not u["cell"]
                or not any(math.dist(u["cell"], c) <= SIGHT for c in ours)
            )
        }
        self.lost_recent = [x for x in self.lost_recent if now - x[0] <= 60]
        self.killed_recent = [x for x in self.killed_recent if now - x[0] <= 60]

    # -- numbers -------------------------------------------------------------------

    def enemy_army_value(self, cost_of: Callable[[str], int]) -> float:
        return float(
            sum(cost_of(u["type"]) for u in self.units.values() if u["type"] not in NON_COMBAT)
        )

    def enemy_types(self) -> dict[str, int]:
        """Combat units seen in the last ``UNIT_MEMORY_S``, by type: what the enemy actually fields."""
        out: dict[str, int] = {}
        for u in self.units.values():
            if u["type"] not in NON_COMBAT:
                out[u["type"]] = out.get(u["type"], 0) + 1
        return out

    def enemy_tower_count(self) -> int:
        return sum(1 for b in self.buildings.values() if str(b.get("type")).lower() in TOWERS)

    def enemy_defence_value(self, cost_of: Callable[[str], int]) -> float:
        """Credit value of enemy defensive structures seen and not yet destroyed. Towers never show up in
        the army value, but they are exactly what shreds a push into a defended base (a lost mirror match:
        the opponent sat behind guns with zero field army and our waves kept feeding them the tiebreak)."""
        return float(
            sum(
                cost_of(str(b.get("type")).lower())
                for b in self.buildings.values()
                if str(b.get("type")).lower() in TOWERS
            )
        )

    def enemy_near(
        self,
        home: tuple[int, int] | None,
        cost_of: Callable[[str], int],
        now: float,
        fresh_s: float = 10.0,
    ) -> tuple[int, float]:
        """Enemy combat units seen in the last ``fresh_s`` within ``NEAR_BASE`` of ``home``: (count, value)."""
        if home is None:
            return 0, 0.0
        near = [
            u
            for u in self.units.values()
            if u["type"] not in NON_COMBAT
            and u["cell"]
            and now - u["seen"] <= fresh_s
            and math.dist(u["cell"], home) <= NEAR_BASE
        ]
        return len(near), float(sum(cost_of(u["type"]) for u in near))

    def summary(
        self,
        *,
        obs: dict[str, Any],
        now: float,
        home: tuple[int, int] | None,
        enemy_base: tuple[int, int] | None,
        cost_of: Callable[[str], int],
        idle: set[int],
    ) -> dict[str, Any]:
        """Both sides, as plain numbers for a decision model."""

        def dist(c: tuple[int, int] | None, to: tuple[int, int] | None) -> int | None:
            return round(math.dist(c, to)) if c and to else None

        mine = [u for u in (obs.get("units") or []) if isinstance(u, dict) and not u.get("wreck")]
        my_blds = [b for b in (obs.get("buildings") or []) if isinstance(b, dict)]
        fighters = [u for u in mine if str(u.get("type")).lower() not in NON_COMBAT]
        f_cells = [c for c in (_cell(u) for u in fighters) if c]
        f_hp = [h for h in (_hp(u) for u in fighters) if h is not None]
        by_type: dict[str, int] = {}
        for u in fighters:
            t = str(u.get("type")).lower()
            by_type[t] = by_type.get(t, 0) + 1
        bld_types: dict[str, int] = {}
        for b in my_blds:
            t = str(b.get("type")).lower()
            bld_types[t] = bld_types.get(t, 0) + 1
        damaged = sum(1 for b in my_blds if (_hp(b) or 100) < 60)
        my_centre = _centroid(f_cells)

        e_army = [u for u in self.units.values() if u["type"] not in NON_COMBAT]
        e_cells = [u["cell"] for u in e_army if u["cell"]]
        e_by_type: dict[str, int] = {}
        for u in e_army:
            e_by_type[u["type"]] = e_by_type.get(u["type"], 0) + 1
        e_hp = [u["hp"] for u in e_army if u["hp"] is not None]
        e_centre = _centroid(e_cells)
        e_blds: dict[str, int] = {}
        for b in self.buildings.values():
            e_blds[b["type"]] = e_blds.get(b["type"], 0) + 1
        e_harv = sum(1 for u in self.units.values() if u["type"] == "harv")
        near_n, near_v = self.enemy_near(home, cost_of, now)
        visible_now = sum(1 for u in e_army if now - u["seen"] <= 1.0)
        military = obs.get("military") or {}

        def count_types(rows: list[tuple[float, str]]) -> dict[str, int]:
            out: dict[str, int] = {}
            for _, t in rows:
                out[t] = out.get(t, 0) + 1
            return out

        return {
            "me": {
                "army_by_type": by_type,
                "army_value": round(sum(cost_of(t) * n for t, n in by_type.items())),
                "army_units": len(fighters),
                "army_idle": sum(1 for u in fighters if u.get("id") in idle),
                "army_avg_hp_pct": round(sum(f_hp) / len(f_hp)) if f_hp else None,
                "army_distance_from_my_base": dist(my_centre, home),
                "army_distance_to_enemy_base": dist(my_centre, enemy_base),
                "buildings_by_type": bld_types,
                "buildings_damaged": damaged,
                "towers": sum(n for t, n in bld_types.items() if t in TOWERS),
                "harvesters": sum(1 for u in mine if str(u.get("type")).lower() == "harv"),
                "lost_last_60s_by_type": count_types(self.lost_recent),
                "enemies_killed_last_60s": len(self.killed_recent),
                "totals": {
                    k: military.get(k)
                    for k in (
                        "kills_cost",
                        "deaths_cost",
                        "units_killed",
                        "units_lost",
                        "buildings_killed",
                        "buildings_lost",
                        "assets_value",
                    )
                },
            },
            "enemy": {
                "army_seen_last_90s_by_type": e_by_type,
                "army_value_seen_last_90s": round(
                    sum(cost_of(t) * n for t, n in e_by_type.items())
                ),
                "army_units_visible_now": visible_now,
                "army_avg_hp_pct": round(sum(e_hp) / len(e_hp)) if e_hp else None,
                "army_distance_from_my_base": dist(e_centre, home),
                "army_distance_from_its_base": dist(e_centre, enemy_base),
                "units_at_my_base": near_n,
                "army_value_at_my_base": round(near_v),
                "buildings_known_by_type": e_blds,
                "towers_known": sum(n for t, n in e_blds.items() if t in TOWERS),
                "harvesters_seen": e_harv,
            },
            "map": {"bases_distance": dist(home, enemy_base)},
        }
