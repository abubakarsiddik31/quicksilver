"""Teach the dex what the live observation knows and the vendored data doesn't.

A new species, form or move that the vendored Showdown tables lack would
otherwise read as "no stats, no types, deals no damage". The observation
publishes ``types`` and ``base_stats`` for every Pokémon it shows and the
type, category, power and accuracy of every move we can use, so those are
learned the first time they are seen. Total: junk is ignored.
"""

from __future__ import annotations

from typing import Any

from pokemon import dex, observation


def _summaries(obs: dict) -> list[dict]:
    out: list[dict] = []
    for key in ("active_pokemon", "opponent_active_pokemon", "your_roster", "opponent_roster"):
        out += [s for s in observation._slots(obs.get(key)) if s]
    for key in ("team", "opponent_team"):
        value = obs.get(key)
        if isinstance(value, dict):
            out += [s for s in value.values() if isinstance(s, dict)]
        elif isinstance(value, list):
            out += [s for s in value if isinstance(s, dict)]
    return out


def _moves(obs: dict, template: Any) -> list[dict]:
    out: list[dict] = []
    for per_slot in observation.available_moves(obs):
        out += per_slot
    slots = template.get("slots") if isinstance(template, dict) else None
    for slot in slots if isinstance(slots, list) else []:
        for option in (slot.get("options") if isinstance(slot, dict) else None) or []:
            if isinstance(option, dict) and option.get("type") == "move":
                out.append(option)
    return out


def from_observation(obs: dict, template: Any = None) -> None:
    try:
        for summary in _summaries(obs):
            dex.learn_species(
                summary.get("species"),
                types=summary.get("types"),
                base_stats=summary.get("base_stats"),
            )
        for entry in _moves(obs, template):
            if isinstance(entry, dict):
                dex.learn_move({**entry, "id": entry.get("id") or entry.get("move_id")})
    except Exception:  # noqa: BLE001 - bookkeeping must never cost a match
        return
