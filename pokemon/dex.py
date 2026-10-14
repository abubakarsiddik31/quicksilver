"""Read-only access to the vendored Showdown data in ``pokemon/data``.

Four files drive everything: ``pokedex`` (types and base stats), ``moves``
(type, category, base power, priority, target), ``typechart`` (the Gen 9
effectiveness table) and ``natures`` (stat modifiers).

Two conventions are worth stating, because both are easy to get subtly wrong:

- **Keys are Showdown IDs**, not display names: lowercase, punctuation and
  spaces removed. ``"Urshifu-Rapid-Strike"`` and ``"urshifurapidstrike"`` are
  the same card, and ``to_id`` is the one way to fold them.
- **Type chart codes are not multipliers.** Showdown stores ``0`` normal,
  ``1`` super effective, ``2`` not very effective and ``3`` immune, so the
  numbers must be mapped rather than used directly.

Everything is lazy and cached: importing this module reads no JSON.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")

# Showdown's damageTaken code → actual multiplier.
_TAKEN_MULTIPLIER = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

# The 18 types, in the game's own display order.
ALL_TYPES: tuple[str, ...] = (
    "Normal",
    "Fighting",
    "Flying",
    "Poison",
    "Ground",
    "Rock",
    "Bug",
    "Ghost",
    "Steel",
    "Fire",
    "Water",
    "Grass",
    "Electric",
    "Psychic",
    "Ice",
    "Dragon",
    "Dark",
    "Fairy",
)


@cache
def _load(name: str) -> dict:
    try:
        text = (DATA_DIR / f"{name}.json").read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def pokedex() -> dict:
    return _load("pokedex")


def moves() -> dict:
    return _load("moves")


def natures() -> dict:
    return _load("natures")


def typechart() -> dict:
    return _load("typechart")


def to_id(name: Any) -> str:
    """Fold a name to its Showdown ID: lowercase, punctuation stripped."""
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# -- species ----------------------------------------------------------------


# Species and moves the vendored Showdown data lacks (a new release, a new form),
# learned from the live observation. The vendored data always wins.
_LEARNED_SPECIES: dict[str, dict] = {}
_LEARNED_MOVES: dict[str, dict] = {}


def learn_species(name: Any, *, types: Any, base_stats: Any) -> None:
    key = to_id(name)
    if not key or key in pokedex() or key in _LEARNED_SPECIES:
        return
    if (
        isinstance(types, list)
        and types
        and all(isinstance(t, str) for t in types)
        and isinstance(base_stats, dict)
        and all(
            isinstance(base_stats.get(k), int) for k in ("hp", "atk", "def", "spa", "spd", "spe")
        )
    ):
        _LEARNED_SPECIES[key] = {"name": name, "types": list(types), "baseStats": dict(base_stats)}


def learn_move(entry: dict) -> None:
    key = to_id(entry.get("id") or entry.get("move_id") or entry.get("name"))
    if not key or key in moves() or key in _LEARNED_MOVES:
        return
    category, mtype = entry.get("category"), entry.get("type")
    power = entry.get("base_power", entry.get("basePower"))
    if category not in ("Physical", "Special", "Status") or not isinstance(mtype, str):
        return
    accuracy = entry.get("accuracy")
    _LEARNED_MOVES[key] = {
        "name": entry.get("name") or key,
        "type": mtype,
        "category": category,
        "basePower": power if isinstance(power, int) and not isinstance(power, bool) else 0,
        "accuracy": accuracy
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and accuracy > 0
        else True,
        "priority": entry.get("priority") if isinstance(entry.get("priority"), int) else 0,
        "target": entry.get("target") if isinstance(entry.get("target"), str) else "normal",
    }


def species(name: Any) -> dict | None:
    key = to_id(name)
    entry = pokedex().get(key) or _LEARNED_SPECIES.get(key)
    return entry if isinstance(entry, dict) else None


def types_of(name: Any) -> tuple[str, ...]:
    entry = species(name) or {}
    value = entry.get("types")
    return tuple(t for t in value if isinstance(t, str)) if isinstance(value, list) else ()


def base_stats(name: Any) -> dict:
    entry = species(name) or {}
    value = entry.get("baseStats")
    return value if isinstance(value, dict) else {}


def base_stat_total(name: Any) -> int:
    return sum(v for v in base_stats(name).values() if isinstance(v, int))


def abilities_of(name: Any) -> tuple[str, ...]:
    entry = species(name) or {}
    value = entry.get("abilities")
    if not isinstance(value, dict):
        return ()
    return tuple(a for a in value.values() if isinstance(a, str))


# -- moves ------------------------------------------------------------------


def move(name: Any) -> dict | None:
    key = to_id(name)
    entry = moves().get(key) or _LEARNED_MOVES.get(key)
    return entry if isinstance(entry, dict) else None


def move_type(name: Any) -> str | None:
    value = (move(name) or {}).get("type")
    return value if isinstance(value, str) else None


def move_category(name: Any) -> str | None:
    """``"Physical"``, ``"Special"`` or ``"Status"``."""
    value = (move(name) or {}).get("category")
    return value if isinstance(value, str) else None


def move_power(name: Any) -> int:
    value = (move(name) or {}).get("basePower")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def move_priority(name: Any) -> int:
    value = (move(name) or {}).get("priority")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def is_damaging(name: Any) -> bool:
    return move_category(name) in ("Physical", "Special") and move_power(name) > 0


# -- type effectiveness -----------------------------------------------------


def effectiveness(attacking_type: Any, defending_types: Any) -> float:
    """Damage multiplier when ``attacking_type`` hits a Pokémon with
    ``defending_types``. ``0.0`` for an immunity.

    Note the indexing direction, which is the opposite of how it reads:
    Showdown stores ``typechart[defending_type]["damageTaken"][attacking_type]``
    — the *defender's* row records what each attacker type does to it. Reading
    it the other way silently produces plausible-looking but wrong numbers
    (Electric into Water comes out neutral instead of doubled).
    """
    if not isinstance(attacking_type, str):
        return 1.0
    if isinstance(defending_types, str):
        defending_types = (defending_types,)
    chart = typechart()
    multiplier = 1.0
    for defending in defending_types or ():
        if not isinstance(defending, str):
            continue
        row = chart.get(defending.lower())
        taken = row.get("damageTaken") if isinstance(row, dict) else None
        if not isinstance(taken, dict):
            continue
        code = taken.get(attacking_type)
        if isinstance(code, int) and not isinstance(code, bool):
            multiplier *= _TAKEN_MULTIPLIER.get(code, 1.0)
    return multiplier


# -- natures ----------------------------------------------------------------


def nature(name: Any) -> dict | None:
    entry = natures().get(to_id(name))
    return entry if isinstance(entry, dict) else None


def nature_multiplier(nature_name: Any, stat: str) -> float:
    """1.1, 0.9 or 1.0 for a stat under a nature."""
    entry = nature(nature_name)
    if not entry:
        return 1.0
    if entry.get("plus") == stat:
        return 1.1
    if entry.get("minus") == stat:
        return 0.9
    return 1.0


def move_accuracy(name: Any) -> float:
    """Hit chance in ``[0, 1]``; moves that never miss report ``True`` upstream."""
    value = (move(name) or {}).get("accuracy")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    return max(0.0, min(1.0, value / 100.0))


_GENERIC: dict[tuple[str, str], str | None] = {}


def generic_move(move_type: str, category: str) -> str | None:
    """A representative ~90-power move of this type and category.

    Used to estimate the threat of a foe whose real moves we cannot see: a
    plain, accurate, single-target attack with no recoil or recharge.
    """
    key = (move_type, category)
    if key not in _GENERIC:
        best: tuple[int, str] | None = None
        for move_id, entry in moves().items():
            if (
                not isinstance(entry, dict)
                or entry.get("isNonstandard")
                or entry.get("type") != move_type
            ):
                continue
            power = entry.get("basePower")
            if (
                entry.get("category") != category
                or not isinstance(power, int)
                or not 80 <= power <= 95
                or entry.get("accuracy") is not True
                and entry.get("accuracy") != 100
                or entry.get("target") != "normal"
                or entry.get("priority", 0) != 0
                or any(
                    entry.get(f)
                    for f in (
                        "recoil",
                        "selfdestruct",
                        "self",
                        "multihit",
                        "drain",
                        "willCrit",
                        "ohko",
                    )
                )
                or any(f in (entry.get("flags") or {}) for f in ("recharge", "charge"))
            ):
                continue
            if best is None or (abs(power - 90), move_id) < best:
                best = (abs(power - 90), move_id)
        _GENERIC[key] = best[1] if best else None
    return _GENERIC[key]
