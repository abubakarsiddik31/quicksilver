"""Battle plans (doctrines): the situational playbook Jev chooses from.

A real war is not one script. Each doctrine is a complete way to fight for the
next stretch of the match: what the money goes into, how big the army must be
before it moves, whether it attacks, holds or goes all in. Jev reads the live
battle numbers (:func:`battle_state`) and picks the doctrine that fits, about
once a second; the executor plays it at 5 Hz.

Switching has hysteresis, so the army does not flip between attack and hold on
every answer: a new doctrine needs a confident pick (or the same pick twice in a
row), and the current one is kept for at least ``MIN_DWELL_S`` unless the new
one is an emergency (``defend``, ``rebuild``).

Without Jev (no key, an outage, low confidence) :func:`fallback` picks with
plain rules over the same numbers, so the play still adapts to the situation.
The hard reflexes in the executor (locked opening, base defence, army floor)
apply under every doctrine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from redalert.plan import Plan

MIN_DWELL_S = 8.0
SWITCH_CONFIDENCE = 0.55  # a pick at least this sure, seen twice in a row, switches
SWITCH_NOW_CONFIDENCE = 0.8  # this sure: switch on the first answer
RULE_CONFIRM_S = 2.0  # a rules pick must hold this long before it switches
EMERGENCY = {"defend", "rebuild"}
ENEMY_MEMORY_S = 90.0  # an enemy unit seen within this long still counts toward its army


@dataclass(frozen=True)
class Doctrine:
    name: str
    when: str  # shown to Jev: when this is the right plan
    army_mode: str  # hold | attack | all_in
    wave_min: int
    army_first: bool
    target: str = "nearest"
    floor_scale: float = 1.0  # multiplies the executor's army floor
    harv_per_proc: int = 2
    extra_towers: int = 0  # towers wanted beyond those already standing
    buildings: tuple[tuple[str, int], ...] = ()  # put in front of the plan's structures
    units: tuple[tuple[str, int], ...] = ()  # put in front of the plan's units (counts are TOTALS)


DOCTRINES: dict[str, Doctrine] = {d.name: d for d in (
    Doctrine(
        "expand",
        "Grow the economy: enemy_army_value_seen_last_90s is small, nothing is near my base and my_army_value "
        "already meets army_needed_for_the_clock; add a refinery, harvesters and power.",
        army_mode="hold", wave_min=10, army_first=False, floor_scale=0.5,
        buildings=(("proc", 2), ("powr", 3)),
    ),
    Doctrine(
        "mass_army",
        "Build up: the armies look even, or my army is below army_needed_for_the_clock with no emergency; put "
        "money into tanks and infantry and attack when a wave of 8 is ready.",
        army_mode="attack", wave_min=8, army_first=True,
    ),
    Doctrine(
        "pressure",
        "Attack now: my_army_vs_enemy_seen is about 1.3 or more (my army is clearly stronger than what I have "
        "seen) and no enemy is at my base; send what is ready and keep reinforcing.",
        army_mode="attack", wave_min=4, army_first=True,
    ),
    Doctrine(
        "defend",
        "Defend: enemy_army_value_near_my_base is large compared with my_army_value (an enemy army is in or at "
        "my base); towers, rocket and rifle infantry, army stays home until the attack is beaten.",
        army_mode="hold", wave_min=12, army_first=True, extra_towers=1,
        units=(("e3", 6), ("e1", 6)),
    ),
    Doctrine(
        "rebuild",
        "Rebuild: my_losses_value is high and my_army_value is now far below army_needed_for_the_clock (my army "
        "was destroyed); cheap units fast, a tower, no tech or expansion until the army is back.",
        army_mode="hold", wave_min=10, army_first=True, harv_per_proc=1, extra_towers=1,
        units=(("e1", 6), ("e3", 6)),
    ),
    Doctrine(
        "all_in",
        "All in: my_army_vs_enemy_seen is 2.5 or more, or remaining_s is under 150 and "
        "kills_value_minus_losses_value is not positive; send every unit at the enemy's buildings. Never on no "
        "information: the enemy army must actually have been seen (or the clock must force it) - on no "
        "information assume the enemy is at least as strong as mine.",
        army_mode="all_in", wave_min=3, army_first=True, target="building",
    ),
)}
DEFAULT = "mass_army"


def apply(doctrine: Doctrine, plan: Plan) -> Plan:
    """The plan to execute under ``doctrine``: its money and army rules on top of the plan's build lists."""
    buildings = (*doctrine.buildings, *(w for w in plan.buildings if w not in doctrine.buildings))
    units = (*doctrine.units, *(w for w in plan.units if w not in doctrine.units))
    return dataclasses.replace(
        plan, buildings=buildings, units=units, army_mode=doctrine.army_mode, wave_min=doctrine.wave_min,
        army_first=doctrine.army_first or plan.army_first,
        target=plan.target if isinstance(plan.target, tuple) else doctrine.target,
    )


def battle_state(
    *, game_s: float, remaining_s: Any, my_army: float, enemy_army_seen: float, enemy_near_base: int,
    enemy_army_near_base: float, my_losses: float, my_kills: float, income: float, harvesters: int,
    refineries: int, towers: int, army_floor: float,
) -> dict[str, Any]:
    """The numbers a battle plan should be chosen on (all engine numbers, no free text)."""
    ratio = my_army / enemy_army_seen if enemy_army_seen > 0 else None
    return {
        "game_s": round(game_s),
        "remaining_s": remaining_s,
        "my_army_value": round(my_army),
        "enemy_army_value_seen_last_90s": round(enemy_army_seen),
        "my_army_vs_enemy_seen": round(ratio, 2) if ratio is not None else "enemy army not seen yet",
        "enemy_units_near_my_base": enemy_near_base,
        "enemy_army_value_near_my_base": round(enemy_army_near_base),
        "kills_value_minus_losses_value": round(my_kills - my_losses),
        "my_losses_value": round(my_losses),
        "income_per_s": round(income, 1),
        "harvesters": harvesters,
        "refineries": refineries,
        "towers": towers,
        "army_needed_for_the_clock": round(army_floor),
    }


def fallback(b: dict[str, Any], current: str = DEFAULT) -> str:
    """Rule-based pick over the same numbers, for when Jev is off, down or unsure.

    ``current`` gives the emergencies a gap between entering and leaving them, so the pick does not flip
    back and forth on a number hovering at the line.
    """
    mine, seen = b["my_army_value"], b["enemy_army_value_seen_last_90s"]
    near = b["enemy_army_value_near_my_base"]
    needed = b["army_needed_for_the_clock"]
    remaining = b["remaining_s"] if isinstance(b["remaining_s"], (int, float)) else 1e9
    if near > max(600, (0.5 if current == "defend" else 0.8) * mine):
        return "defend"
    lost_it = b.get("my_losses_value", 0) >= max(1000, mine)  # the army was beaten, not just not built yet
    if b["game_s"] > 200 and lost_it and mine < (0.6 if current == "rebuild" else 0.35) * needed:
        return "rebuild"
    if mine >= 8000 and (seen == 0 or mine >= 2.5 * seen):
        return "all_in"
    if remaining < 150 and b["kills_value_minus_losses_value"] <= 0 and mine >= 2000:
        return "all_in"
    if mine >= 3000 and ((seen > 0 and mine >= 1.3 * seen) or mine >= 5000):
        return "pressure"  # stronger than what we have seen, or big enough to attack blind
    if b["game_s"] < 300 and seen < 1000 and near == 0 and b["refineries"] < 2 and b["harvesters"] >= 2 \
            and mine >= needed:
        return "expand"  # safe to grow: no enemy army about and ours is where the clock wants it
    return "mass_army"


def all_in_allowed(b: dict[str, Any]) -> bool:
    """True when an all-in is informed: the enemy army was seen, or the clock leaves no choice.

    A blind all-in (nothing seen since the scout died) is how a whole army dies into an unseen
    defence line: on no information the enemy is assumed at least as strong as we are.
    """
    seen = b.get("enemy_army_value_seen_last_90s", 0)
    if isinstance(seen, (int, float)) and seen > 0:
        return True
    remaining = b.get("remaining_s")
    forced = isinstance(remaining, (int, float)) and remaining < 150 \
        and b.get("kills_value_minus_losses_value", 0) <= 0
    return bool(forced)


class DoctrineState:
    """The doctrine in force, switched with hysteresis."""

    def __init__(self) -> None:
        self.name = DEFAULT
        self.since = 0.0
        self.source = "default"
        # The challenger: (doctrine, first seen at, id of the Jev answer that proposed it last).
        self._pending: tuple[str, float, Any] | None = None

    @property
    def doctrine(self) -> Doctrine:
        return DOCTRINES[self.name]

    def update(
        self, now: float, jev_pick: str | None, jev_conf: float | None, rule_pick: str, answer_id: Any = None,
    ) -> bool:
        """Fold in this decision's evidence; True when the doctrine changed.

        Called at 5 Hz while Jev answers about once a second, so "picked twice" means two different Jev
        answers (``answer_id``); a rules pick must instead hold for ``RULE_CONFIRM_S``. Only a sure pick
        (``SWITCH_NOW_CONFIDENCE``) switches on one answer. An emergency (``defend``/``rebuild``) waives the
        dwell clock but still needs that sureness or a second confirming answer: a single 0.6-confidence
        "defend" mid-opening once flipped the whole economy into an over-commit it never recovered from.
        """
        if jev_pick in DOCTRINES and jev_conf is not None and jev_conf >= SWITCH_CONFIDENCE:
            pick, source, sure = jev_pick, "jev", jev_conf >= SWITCH_NOW_CONFIDENCE
        else:
            pick, source, sure, answer_id = rule_pick, "rules", False, None
        if pick == self.name:
            self._pending = None
            self.source = source
            return False
        if pick not in EMERGENCY and now - self.since < MIN_DWELL_S:
            return False  # emergencies waive the dwell, never the confirmation below
        confirmed = False
        if self._pending and self._pending[0] == pick:
            _, first, last_id = self._pending
            confirmed = (answer_id is not None and answer_id != last_id) or \
                (answer_id is None and now - first >= RULE_CONFIRM_S)
            self._pending = (pick, first, answer_id if answer_id is not None else last_id)
        else:
            self._pending = (pick, now, answer_id)
        if not (sure or confirmed):
            return False
        self.name, self.since, self._pending, self.source = pick, now, None, source
        return True
