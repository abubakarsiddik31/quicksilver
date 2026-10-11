"""Level-50 stat computation.

The modern (Gen 3+) formula, which is what ``gen9vgc2025regi`` uses::

    HP      = floor((2·base + iv + floor(ev/4)) · level / 100) + level + 10
    other   = floor((floor((2·base + iv + floor(ev/4)) · level / 100) + 5) · nature)

Two details are easy to get wrong and both change the numbers:

- **An unspecified IV is 31**, not 0. Showdown omits IVs equal to the default,
  and draft cards publish ``"ivs": {}`` — reading that as zero would silently
  understate every stat.
- **Nature applies after the +5**, and HP is never nature-modified.
"""

from __future__ import annotations

from typing import Any

from pokemon import dex

DEFAULT_IV = 31
MAX_EV_PER_STAT = 252
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def compute_stats(
    species: Any,
    evs: Any = None,
    ivs: Any = None,
    nature: Any = None,
    level: int = 50,
) -> dict[str, int]:
    """The six stats of a Pokémon at ``level``, or ``{}`` for an unknown species."""
    base_stats = dex.base_stats(species)
    if not base_stats:
        return {}
    evs = evs if isinstance(evs, dict) else {}
    ivs = ivs if isinstance(ivs, dict) else {}
    level = _clamp(_int(level, 50), 1, 100)

    out: dict[str, int] = {}
    for key in STAT_KEYS:
        base = _int(base_stats.get(key, 0), 0)
        if base <= 0:
            continue
        ev = _clamp(_int(evs.get(key, 0), 0), 0, MAX_EV_PER_STAT)
        iv = _clamp(_int(ivs.get(key, DEFAULT_IV), DEFAULT_IV), 0, 31)
        core = ((2 * base + iv + ev // 4) * level) // 100
        if key == "hp":
            # Shedinja is the one species with a fixed 1 HP.
            out[key] = 1 if base == 1 else core + level + 10
        else:
            out[key] = int((core + 5) * dex.nature_multiplier(nature, key))
    return out


def stat_of(stats: dict, key: str) -> int:
    return _int(stats.get(key, 0), 0)


def stage_multiplier(stage: Any) -> float:
    """The multiplier for a stat stage (-6..+6)."""
    stage = _clamp(_int(stage, 0), -6, 6)
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
