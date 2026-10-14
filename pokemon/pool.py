"""Capture and persist the fixed 18-card draft pool.

The pool is not published anywhere. It is not in the platform documentation,
and ``get_game_config("pokemon_vgc_doubles_draft")`` returns only ``phases``,
``battle_format`` and ``team_mode``. The cards arrive in the **draft
observation**, as ``available_cards``, each a complete pre-built set.

That matters for planning: it means the matchup precomputation is *not* pure
offline work — it cannot start until the agent has seen one draft state. But
the pool is **fixed** (both players draft from the same 18 pre-built cards in
every game), so a single sighting is enough to make everything after it
offline work that needs no API key and no game server.

This module banks that sighting to ``pokemon/pool.json``.

``available_cards`` shrinks as cards are taken (and an Item Clause clash
removes more), so a single observation may not show all 18. ``merge`` unions
observations by ``card_id``, which reconstructs the full pool over the first
draft without needing to catch it at pick 1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pokemon import observation as obs

POOL_PATH = Path(__file__).with_name("pool.json")

# The documented size of the pool. Used only to report how complete a capture
# is — never to pad or truncate one.
EXPECTED_POOL_SIZE = 18


def default_pool_path() -> Path:
    """Where the pool is banked, unless a caller passes its own path.

    Overridable with ``QUICKSILVER_POOL_PATH`` so a test (or a second run)
    never writes the repository's real capture.
    """
    override = os.environ.get("QUICKSILVER_POOL_PATH")
    return Path(override) if override else POOL_PATH


def load_pool(path: Path | None = None) -> list[dict] | None:
    """The banked pool, or ``None`` if nothing has been captured yet."""
    target = default_pool_path() if path is None else path
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    cards = data.get("cards") if isinstance(data, dict) else data
    if not isinstance(cards, list):
        return None
    return [card for card in cards if isinstance(card, dict)]


def save_pool(cards: list[dict], path: Path | None = None) -> Path:
    """Write the pool, deterministically ordered by ``card_id``."""
    target = default_pool_path() if path is None else path
    ordered = sorted(cards, key=lambda card: str(card.get("card_id", "")))
    payload = {"expected_size": EXPECTED_POOL_SIZE, "count": len(ordered), "cards": ordered}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def merge(existing: list[dict] | None, cards: list[dict]) -> list[dict]:
    """Union by ``card_id``. A newer sighting of a card replaces the older one:
    the pool is *not* fixed — new cards turn up between matches, and a card id can
    come back with a different set — so the latest set is the one to trust."""
    merged: dict[str, dict] = {}
    for card in list(existing or []) + list(cards or []):
        if not isinstance(card, dict):
            continue
        card_id = card.get("card_id")
        if isinstance(card_id, str) and card_id:
            merged[card_id] = {k: v for k, v in card.items() if not k.startswith("_")}
    return sorted(merged.values(), key=lambda card: str(card.get("card_id", "")))


def is_complete(cards: list[dict] | None) -> bool:
    """Whether a capture looks like the whole pool. Informational: the policy
    must still work from a partial capture."""
    return bool(cards) and len(cards) >= EXPECTED_POOL_SIZE


def is_full_card(card: Any) -> bool:
    """Whether a card carries an actual pre-built set, rather than just an id.

    Some states publish ``available_card_ids`` without ``available_cards``.
    Those are useless for matchup work, and worse than useless to *bank*:
    ``merge`` keeps the first sighting, so a bare id recorded now would block
    the real definition seen later.
    """
    return (
        isinstance(card, dict)
        and isinstance(card.get("card_id"), str)
        and bool(card.get("card_id"))
        and isinstance(card.get("species"), str)
        and bool(card.get("species"))
    )


class MatchCards:
    """Every full card seen *in this match*, with who holds it.

    The banked file is only what earlier matches showed. A new match can bring
    cards the file has never seen, or a known card with a different set, so the
    match's own draft — available cards and both rosters — is the source of
    truth and the file is the fallback for what this match never showed.
    """

    def __init__(self) -> None:
        self._cards: dict[str, dict] = {}

    def observe(self, observation: dict, agent_id: str | None = None) -> None:
        for card in obs.pool_of(observation):
            self._add(card, None)
        for owner_id, cards in obs.rosters(observation).items():
            side = None if agent_id is None else ("ours" if owner_id == agent_id else "theirs")
            for card in cards:
                self._add(card, side)

    def _add(self, card: Any, owner: str | None) -> None:
        if not is_full_card(card):
            return
        known = self._cards.get(card["card_id"])
        owner = owner or (known or {}).get("_owner")
        self._cards[card["card_id"]] = {**card, **({"_owner": owner} if owner else {})}

    def cards(self) -> list[dict]:
        return list(self._cards.values())

    def with_fallback(self, banked: list[dict] | None) -> list[dict] | None:
        """This match's cards first, then banked cards for anything it never showed."""
        mine = self.cards()
        if not mine:
            return banked
        seen = {c["card_id"] for c in mine}
        return mine + [c for c in (banked or []) if c.get("card_id") not in seen]


def capture(state, *, path: Path | None = None) -> list[dict] | None:
    """Merge this state's draft cards into the banked pool.

    Returns the pool as it now stands, or ``None`` when the state shows nothing
    worth banking (a non-draft phase, only ids, or nothing new).
    """
    observation = obs.observation_of(state)
    cards = [card for card in obs.pool_of(observation) if is_full_card(card)]
    if not cards:
        return None
    merged = merge(load_pool(path), cards)
    save_pool(merged, path)
    return merged
