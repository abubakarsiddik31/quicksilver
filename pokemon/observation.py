"""Typed readers for Pokémon's observation blocks.

Pokémon's ``observation`` is a JSON object whose shape changes with the phase
(``draft`` → ``team_preview`` → ``moving``), so the policy needs one place that
knows those shapes. Everything here is *total*: an unexpected shape yields an
empty value rather than an exception, for the same reason as
``pokemon/actions.py`` — a raised decision kills that match's worker process.

.. warning::

    ``GameState.observation`` is **not** the JSON. ``altruagent/models.py``
    coerces it with ``str(...)``, so what you get is a Python ``repr`` of the
    dict, and ``json.loads`` on it fails. The authoritative, unmodified
    response is ``GameState.raw`` (``raw=state``). ``observation_of()`` below
    reads that first and only falls back to the repr, which is why policy code
    should always go through it rather than touching ``state.observation``.
"""

from __future__ import annotations

import ast
import json
from typing import Any


def observation_of(state: Any) -> dict:
    """The Pokémon observation object for a state, or ``{}``.

    Prefers ``state.raw['observation']`` (the unmodified response); falls back
    to the ``str()`` repr on ``state.observation``.
    """
    raw = getattr(state, "raw", None)
    if isinstance(raw, dict):
        candidate = raw.get("observation")
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str) and candidate:
            parsed = _parse(candidate)
            if parsed:
                return parsed

    fallback = getattr(state, "observation", None)
    if isinstance(fallback, dict):
        return fallback
    if isinstance(fallback, str) and fallback:
        return _parse(fallback)
    return {}


def _parse(text: str) -> dict:
    """A dict from either JSON or a Python ``repr`` (the latter is what
    ``GameState.observation`` actually holds)."""
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def phase_of(state: Any, observation: dict | None = None) -> str:
    """The observation's own ``phase``, falling back to the state's.

    The two are worth distinguishing: ``state.phase`` is what the runner
    branches on, while ``observation.phase`` is the adapter's finer-grained
    view.
    """
    obs = observation_of(state) if observation is None else observation
    value = obs.get("phase")
    if isinstance(value, str) and value:
        return value
    state_phase = getattr(state, "phase", None)
    return state_phase if isinstance(state_phase, str) else ""


def is_doubles(observation: dict) -> bool:
    return bool(observation.get("is_doubles"))


def normalize_species(name: Any) -> str:
    """Fold a species name for comparison.

    The server ignores case and punctuation when matching a lineup
    ("Landorus-Therian" matches "landorustherian"), so we do too.
    """
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _slots(value: Any) -> list[dict | None]:
    """A per-slot field: doubles give a 2-element list, singles a bare object.
    Positions are preserved, with ``None`` for a missing slot, so slot indices
    never shift."""
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, dict) else None for item in value]


# -- draft ------------------------------------------------------------------


def pool_of(observation: dict) -> list[dict]:
    """Every card still on offer, each a complete pre-built set.

    Each entry carries ``card_id``, ``species``, ``item``, ``ability``,
    ``nature``, ``level``, ``evs``, ``ivs`` and ``moves`` — everything a
    damage calculation needs. The pool is fixed across games, so capturing it
    once (see ``pokemon.pool``) makes the rest of the work offline.
    """
    cards = _dicts(observation.get("available_cards"))
    if cards:
        return cards
    # Some states carry only the ids.
    ids = observation.get("available_card_ids")
    if isinstance(ids, list):
        return [{"card_id": cid} for cid in ids if isinstance(cid, str)]
    return []


def available_card_ids(observation: dict) -> list[str]:
    ids = observation.get("available_card_ids")
    if isinstance(ids, list):
        return [cid for cid in ids if isinstance(cid, str)]
    return [
        card["card_id"]
        for card in pool_of(observation)
        if isinstance(card.get("card_id"), str)
    ]


def current_seat(observation: dict) -> str | None:
    value = observation.get("current_seat")
    return value if isinstance(value, str) else None


def first_drafter(observation: dict) -> str | None:
    value = observation.get("first_drafter")
    return value if isinstance(value, str) else None


def pick_number(observation: dict) -> int | None:
    return _int_or_none(observation.get("pick_number"))


def picks_remaining(observation: dict) -> int | None:
    return _int_or_none(observation.get("picks_remaining"))


def draft_complete(observation: dict) -> bool:
    return bool(observation.get("draft_complete"))


def rosters(observation: dict) -> dict[str, list[dict]]:
    """``{agent_id: [card, ...]}`` — both players' drafted cards, which is
    public: denying the opponent a card needs their roster, not just yours."""
    value = observation.get("rosters")
    if not isinstance(value, dict):
        return {}
    return {key: _dicts(cards) for key, cards in value.items() if isinstance(key, str)}


def picks(observation: dict) -> list[dict]:
    return _dicts(observation.get("picks"))


def my_roster(observation: dict, agent_id: str | None) -> list[dict]:
    if not agent_id:
        return []
    return rosters(observation).get(agent_id, [])


# -- team preview -----------------------------------------------------------


def your_roster(observation: dict) -> list[dict]:
    return _dicts(observation.get("your_roster"))


def opponent_roster(observation: dict) -> list[dict]:
    return _dicts(observation.get("opponent_roster"))


# -- moving -----------------------------------------------------------------


def active_pokemon(observation: dict) -> list[dict | None]:
    """The board, one entry per active slot (``[None]``-padded to 2 in
    doubles)."""
    return _slots(observation.get("active_pokemon"))


def opponent_active_pokemon(observation: dict) -> list[dict | None]:
    return _slots(observation.get("opponent_active_pokemon"))


def force_switch(observation: dict) -> list[Any]:
    value = observation.get("force_switch")
    if isinstance(value, list):
        return value
    if isinstance(value, bool):
        return [value]
    return []


def available_moves(observation: dict) -> list[list[dict]]:
    """Per-slot move lists: ``[[...], [...]]`` in doubles."""
    value = observation.get("available_moves")
    if not isinstance(value, list):
        return []
    if value and all(isinstance(entry, dict) for entry in value):
        return [_dicts(value)]  # singles: one flat list
    return [_dicts(entry) for entry in value]


def available_switches(observation: dict) -> list[list[dict]]:
    value = observation.get("available_switches")
    if not isinstance(value, list):
        return []
    if value and all(isinstance(entry, dict) for entry in value):
        return [_dicts(value)]
    return [_dicts(entry) for entry in value]


def weather(observation: dict) -> str | None:
    value = observation.get("weather")
    return value if isinstance(value, str) else None


def turn(observation: dict) -> int | None:
    return _int_or_none(observation.get("turn"))


# -- clock ------------------------------------------------------------------

# The platform takes the decision out of your hands a little before Showdown's
# own timer does (see docs/altruagent-skill/pokemon.md, section 7).
PLATFORM_DECISION_SECONDS = 50
PLATFORM_PREVIEW_SECONDS = 85


def clock(observation: dict) -> dict:
    value = observation.get("clock")
    return value if isinstance(value, dict) else {}


def decision_seconds_left(observation: dict) -> int | None:
    return _int_or_none(clock(observation).get("decision_seconds_left"))


def bank_seconds_left(observation: dict) -> int | None:
    return _int_or_none(clock(observation).get("bank_seconds_left"))


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
