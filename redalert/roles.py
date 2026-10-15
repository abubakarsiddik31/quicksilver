"""The chain of command: a General who reads the battle, and soldiers who each hold exactly one role.

Before this module the army was one blob plus three ad-hoc sets in the agent (scouts, guard, flankers),
each sized by its own constant. The lost matches show what that costs: three units walk out in front of an
enemy that keeps its army together, nobody is left at home when the push arrives, and the build never
answers what the enemy actually fields.

* :class:`General` reads the situation (enemy composition, threat, the doctrine, the clock) and issues a
  :class:`Directive`: how many soldiers hold the home line, how many raid, how many scout, what to build
  against what the enemy fields, and which share of the army must be vehicles. It decides; it never moves a unit.
* :func:`assign` gives every combat unit one :class:`Role`, sticky while the unit lives, so a unit is never
  both a scout and a defender and the quotas are met in priority order: scouts, garrison, raiders, strike.

Pure functions over plain data, so each decision is unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


class Role:
    SCOUT = "scout"  # finds the enemy: cheap, fast, expendable, never fights
    GARRISON = "garrison"  # holds the home line: engages in range, never chases
    RAIDER = "raider"  # a small pack that hits from the side while the strike force fixes the front
    STRIKE = "strike"  # the main army: gathers into one wave, moves and fights together


ARMOR = {
    "1tnk",
    "2tnk",
    "3tnk",
    "4tnk",
    "ttnk",
    "ctnk",
    "stnk",
    "mrj",
    "mgg",
    "apc",
    "arty",
    "v2rl",
    "ftrk",
    "jeep",
}
INFANTRY = {"e1", "e2", "e3", "e4", "e7", "dog", "shok"}
AIR = {"mig", "yak", "hind", "heli", "badr", "u2", "mh60", "tran"}
HEAVY = (
    "3tnk",
    "2tnk",
    "1tnk",
    "4tnk",
)  # what to build when the enemy fields infantry: tanks crush it
VEHICLES = ARMOR | {"apc", "harv", "mcv"}

# The counter table: what to put first when the enemy fields mostly X.
VS_ARMOR = {"units": ("e3", "3tnk", "2tnk"), "towers": ("gun", "tsla", "pbox")}
VS_INFANTRY = {"units": ("e1", "3tnk", "2tnk", "1tnk"), "towers": ("ftur", "pbox", "tsla")}
VS_MIXED = {"units": ("e1", "e3", "3tnk", "2tnk"), "towers": ("pbox", "ftur", "gun", "tsla")}
VS_AIR = {"units": ("e3", "e1"), "towers": ("sam", "agun")}


@dataclass(frozen=True)
class Directive:
    garrison: int  # soldiers on the home line
    raiders: int  # size of a flank pack (0: no raids)
    scouts: int  # scouts wanted alive
    units: tuple[str, ...]  # production priority against what the enemy fields, best counter first
    towers: tuple[str, ...]  # tower preference, best counter first
    vehicle_share: float  # target share of army value that is vehicles
    stance: str  # what the enemy looks like: blind | armor | infantry | air | mixed | turtle
    reason: str


def profile(enemy_types: dict[str, int], enemy_towers: int = 0) -> str:
    """Classify the enemy army from the unit types we have seen (type -> count)."""
    n = sum(enemy_types.values())
    if enemy_towers >= 3 and n < 6:
        return "turtle"
    if n < 3:
        return "blind"
    air = sum(c for t, c in enemy_types.items() if t in AIR)
    armor = sum(c for t, c in enemy_types.items() if t in ARMOR)
    inf = sum(c for t, c in enemy_types.items() if t in INFANTRY)
    if air * 4 >= n:
        return "air"
    if armor >= 0.55 * n:
        return "armor"
    if inf >= 0.65 * n:
        return "infantry"
    return "mixed"


HOME_VS_SEEN = 0.3  # home-line soldiers per enemy body seen (a third of what is coming, until the army is half home)


class General:
    """Reads the battle and issues the :class:`Directive` every soldier takes its orders from."""

    ATTACKING = {"pressure", "all_in"}
    HOMEBOUND = {"defend", "rebuild", "expand"}

    def decide(
        self,
        *,
        game_s: float,
        army_n: int,
        doctrine: str,
        enemy_types: dict[str, int],
        enemy_towers: int = 0,
        raided_recently: bool = False,
        enemy_near: int = 0,
        blind_for_s: float = 0.0,
    ) -> Directive:
        stance = profile(enemy_types, enemy_towers)
        table = {"armor": VS_ARMOR, "infantry": VS_INFANTRY, "air": VS_AIR}.get(stance, VS_MIXED)

        # Home line: a quarter of the army whatever the threat (the reflex recalls everyone when the enemy is
        # actually in the base; in the battle sim a standing half-army line lost fights the whole army wins); a thin line even when attacking, because the enemy that keeps its army together
        # walks in as the strike force walks out.
        threatened = raided_recently or enemy_near > 0
        if doctrine in self.HOMEBOUND or threatened:
            share = 0.25
        elif doctrine in self.ATTACKING:
            share = 0.15
        else:
            share = 0.25
        garrison = round(army_n * share)
        if game_s >= 150 and army_n >= 6:
            garrison = max(garrison, 4)  # nobody home was how the early rushes killed us
        # ...and sized to what the enemy has shown: a home line much thinner than the bodies seen coming is a
        # line that dies first. Not for an all-in, and never more than half of what we have.
        if doctrine != "all_in" and game_s >= 150:
            seen = sum(n for n in enemy_types.values() if isinstance(n, int))
            garrison = max(garrison, min(army_n // 2, math.ceil(HOME_VS_SEEN * seen)))
        garrison = min(garrison, army_n)

        # Raids need an army worth splitting and an enemy not already at our door.
        raiders = (
            3
            if (
                army_n >= 12
                and game_s >= 240
                and doctrine not in self.HOMEBOUND
                and enemy_near == 0  # a lone scout's "under attack" must not cancel every flank run
                and doctrine != "all_in"
            )
            else 0
        )

        scouts = 1
        if game_s >= 200 and blind_for_s >= 60.0:
            scouts = 2  # blind this long means the army is deciding on nothing

        vehicle_share = (
            0.0 if game_s < 120 else (0.35 if stance in ("infantry", "mixed", "blind") else 0.45)
        )
        reason = f"enemy={stance} doctrine={doctrine} threatened={threatened}"
        return Directive(
            garrison=garrison,
            raiders=raiders,
            scouts=scouts,
            units=table["units"],
            towers=table["towers"],
            vehicle_share=vehicle_share,
            stance=stance,
            reason=reason,
        )


@dataclass(frozen=True)
class Soldier:
    id: int
    kind: str
    cost: int


def assign(
    soldiers: Iterable[Soldier],
    previous: dict[int, str],
    directive: Directive,
    *,
    keep_scouts: set[int],
    preferred: tuple[str, ...] = (),
) -> dict[int, str]:
    """One role per soldier. Existing roles stick while their quota holds, so the line does not reshuffle
    every tick; the freed-up and the new fill the quotas in priority order: scout, garrison, raider, strike.

    The garrison takes the types that counter what the enemy fields (``preferred``), then the cheapest;
    raiders are the cheapest left; everyone else is the strike force.
    """
    pool = {s.id: s for s in soldiers}
    roles: dict[int, str] = {}
    for uid in keep_scouts & pool.keys():
        roles[uid] = Role.SCOUT

    def rank(s: Soldier) -> tuple[int, int]:
        return (preferred.index(s.kind) if s.kind in preferred else len(preferred), s.cost)

    for role, quota in ((Role.GARRISON, directive.garrison), (Role.RAIDER, directive.raiders)):
        held = [
            uid for uid, r in previous.items() if r == role and uid in pool and uid not in roles
        ]
        held = held[:quota]
        for uid in held:
            roles[uid] = role
        free = sorted(
            (s for uid, s in pool.items() if uid not in roles),
            key=rank if role == Role.GARRISON else lambda s: s.cost,
        )
        for s in free[: max(0, quota - len(held))]:
            roles[s.id] = role
    for uid in pool:
        roles.setdefault(uid, Role.STRIKE)
    return roles
