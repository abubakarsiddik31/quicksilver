"""What the move scorer needs besides our own damage: the field, speed, the
damage coming back at us, and what we did last turn.

Everything is total — an unreadable observation yields a neutral value, never an
exception, because a raised error here kills that match's worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pokemon import dex
from pokemon import stats as stats_mod
from pokemon.damage import DamageResult, damage

AVERAGE_ROLL = 0.925
PROTECT_LIKE = {
    "protect",
    "detect",
    "spikyshield",
    "banefulbunker",
    "kingsshield",
    "silktrap",
    "obstruct",
}


# -- field -------------------------------------------------------------------


def _flat(value: Any) -> str:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return "".join(ch for ch in text.lower() if ch.isalnum())


@dataclass(frozen=True)
class Field:
    weather: str | None = None
    trick_room: bool = False
    tailwind_us: bool = False
    tailwind_them: bool = False
    reflect_us: bool = False
    light_screen_us: bool = False
    reflect_them: bool = False
    light_screen_them: bool = False

    def screen_up(self, name: str) -> bool:
        """Whether *we* already have this screen (Aurora Veil counts for both)."""
        return {"reflect": self.reflect_us, "lightscreen": self.light_screen_us}.get(name, False)


def field_of(obs: dict) -> Field:
    ours, theirs = _flat(obs.get("side_conditions")), _flat(obs.get("opponent_side_conditions"))
    weather = obs.get("weather")
    return Field(
        weather=weather
        if isinstance(weather, str) and weather and weather.lower() != "none"
        else None,
        trick_room="trickroom" in _flat(obs.get("fields")),
        tailwind_us="tailwind" in ours,
        tailwind_them="tailwind" in theirs,
        reflect_us="reflect" in ours or "auroraveil" in ours,
        light_screen_us="lightscreen" in ours or "auroraveil" in ours,
        reflect_them="reflect" in theirs or "auroraveil" in theirs,
        light_screen_them="lightscreen" in theirs or "auroraveil" in theirs,
    )


# -- speed -------------------------------------------------------------------


def raw_speed(active: Any, fld: Field, *, ours: bool) -> float | None:
    """Speed with boosts, Choice Scarf and Tailwind — Trick Room not applied."""
    card = getattr(active, "card", None)
    if card is None:
        return None
    spe = card.stats.get("spe")
    if not isinstance(spe, (int, float)) or isinstance(spe, bool):
        return None
    value = float(spe) * stats_mod.stage_multiplier(active.boosts.get("spe", 0))
    if card.item == "Choice Scarf":
        value *= 1.5
    if fld.tailwind_us if ours else fld.tailwind_them:
        value *= 2.0
    return value


def moves_first(a: float | None, b: float | None, fld: Field) -> bool | None:
    """Whether speed ``a`` acts before ``b`` (``None`` if either is unknown)."""
    if a is None or b is None:
        return None
    return a < b if fld.trick_room else a > b


# -- damage with the field applied ---------------------------------------------


def calc(
    attacker: Any,
    defender: Any,
    move_id: str,
    fld: Field,
    *,
    defender_is_ours: bool,
    spread: bool = False,
) -> DamageResult | None:
    """:func:`damage` with weather, screens, burn and the spread penalty applied."""
    if getattr(attacker, "card", None) is None or getattr(defender, "card", None) is None:
        return None
    status = str(attacker.summary.get("status") or "").lower()
    try:
        return damage(
            attacker.card,
            defender.card,
            move_id,
            attacker_boosts=attacker.boosts,
            defender_boosts=defender.boosts,
            weather=fld.weather,
            spread=spread,
            reflect=fld.reflect_us if defender_is_ours else fld.reflect_them,
            light_screen=fld.light_screen_us if defender_is_ours else fld.light_screen_them,
            defender_full_hp=defender.hp_fraction >= 1.0,
            burned=status in ("brn", "burn", "burned"),
        )
    except Exception:  # noqa: BLE001 - scoring must never break a decision
        return None


def best_hit(
    attacker: Any, defender: Any, fld: Field, *, defender_is_ours: bool, spread_ok: bool = True
) -> float:
    """The most this attacker's known moves can take off the defender, as an
    expected percent of its max HP (average roll, times accuracy)."""
    card = getattr(attacker, "card", None)
    if card is None or not getattr(attacker, "usable", False):
        return 0.0
    best = 0.0
    known = [m for m in card.moves if dex.is_damaging(m)]
    candidates = known or _guessed_moves(card)
    for move_id in candidates:
        target = (dex.move(move_id) or {}).get("target")
        spread = spread_ok and target in ("allAdjacentFoes", "allAdjacent")
        result = calc(
            attacker, defender, move_id, fld, defender_is_ours=defender_is_ours, spread=spread
        )
        if result is None:
            continue
        best = max(best, result.max_percent * AVERAGE_ROLL * dex.move_accuracy(move_id))
    return best


def _guessed_moves(card: Any) -> list[str]:
    """Stand-in attacks for a foe whose set we cannot see: one STAB move of its
    stronger attacking category per type."""
    base = dex.base_stats(card.species)
    category = "Physical" if base.get("atk", 0) >= base.get("spa", 0) else "Special"
    guesses = (dex.generic_move(t, category) for t in card.types)
    return [g for g in guesses if g]


# -- the danger to each of our slots ------------------------------------------


@dataclass(frozen=True)
class Danger:
    """Per our slot: the share of its *remaining* HP the foes are expected to
    take this turn (1.0 = dead), and whether a faster foe can KO it before it moves."""

    risk: tuple[float, ...] = (0.0, 0.0)
    doomed: tuple[bool, ...] = (False, False)

    def risk_of(self, slot: int | None) -> float:
        return self.risk[slot] if slot is not None and 0 <= slot < len(self.risk) else 0.0

    def doomed_of(self, slot: int | None) -> bool:
        return self.doomed[slot] if slot is not None and 0 <= slot < len(self.doomed) else False


def danger_of(ours: list, theirs: list, fld: Field) -> Danger:
    risks: list[float] = []
    doomed: list[bool] = []
    for me in ours:
        if not getattr(me, "usable", False):
            risks.append(0.0)
            doomed.append(False)
            continue
        hp_percent = max(1.0, me.hp_fraction * 100.0)
        hits = sorted(
            (
                (best_hit(foe, me, fld, defender_is_ours=True), foe)
                for foe in theirs
                if getattr(foe, "usable", False)
            ),
            key=lambda pair: -pair[0],
        )
        # Foes may split their attacks: the best hit lands in full, the rest on average half.
        total = (hits[0][0] + 0.5 * sum(h for h, _ in hits[1:])) if hits else 0.0
        risks.append(min(1.0, total / hp_percent))

        mine = raw_speed(me, fld, ours=True)
        is_doomed = False
        for hit, foe in hits:
            if hit / AVERAGE_ROLL >= hp_percent and moves_first(
                raw_speed(foe, fld, ours=False), mine, fld
            ):
                is_doomed = True
                break
        doomed.append(is_doomed)
    return Danger(tuple(risks), tuple(doomed))


# -- memory across turns --------------------------------------------------------


@dataclass
class Memory:
    """What the observation does not publish: how long each of our Pokémon has
    been out (Fake Out) and what it used last turn (Protect fails twice in a row)."""

    last_moves: dict[str, str] = field(default_factory=dict)
    entered: dict[str, int] = field(default_factory=dict)
    _active: set[str] = field(default_factory=set)

    @staticmethod
    def _species(summaries: list) -> list[str | None]:
        out: list[str | None] = []
        for s in summaries:
            sp = s.get("species") if isinstance(s, dict) else None
            out.append(sp if isinstance(sp, str) and sp else None)
        return out

    def observe(self, ours_summaries: list, turn: int | None) -> None:
        current = {s for s in self._species(ours_summaries) if s}
        for species in current - self._active:
            self.entered[species] = turn if isinstance(turn, int) else 1
        self._active = current

    def record(self, ours_summaries: list, plan: Any) -> None:
        """Remember each slot's move (``None`` for a switch or pass)."""
        self.last_moves = {}
        if not isinstance(plan, dict):
            return
        for slot, species in enumerate(self._species(ours_summaries)):
            choice = plan.get(f"slot_{slot}")
            if species and isinstance(choice, dict) and choice.get("type") == "move":
                self.last_moves[species] = dex.to_id(choice.get("move_id"))

    def protected_last(self, species: str | None) -> bool:
        return bool(species) and self.last_moves.get(species) in PROTECT_LIKE

    def fake_out_ok(self, species: str | None, turn: int | None) -> bool:
        """Fake Out only works on the turn a Pokémon comes in."""
        if species and species in self.entered:
            return self.entered[species] == turn
        return turn == 1
