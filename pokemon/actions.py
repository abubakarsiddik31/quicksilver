"""Reading the Pokémon action templates GameAPI sends.

``pokemon_vgc_doubles_draft`` has three decision shapes:

``draft pick``
    Enumerated in ``state.legal_actions``, each entry carrying ready-made
    ``input`` arguments. There is nothing to build — choose one (and the
    server's action list is already filtered to what is legal).
``select_lineup``
    Team Preview. One legal action whose ``input['action']`` is a *template*:
    submit 4 of your 6, of which the first 2 lead the battle.
``doubles_turn``
    One legal action whose ``input['action']`` is a template with one entry
    per active slot; each slot's ``options`` lists that slot's moves,
    available switches and ``pass``.

Everything here reads from plain dicts, so it can be exercised offline against
recorded states without a game server. **Nothing here raises.** A raised
decision stops that match's worker process and hands the rest of the battle to
the server's auto-play, so every reader returns empty on an unexpected shape
and lets the caller degrade to something legal.
"""

from __future__ import annotations

from typing import Any

PASS: dict[str, Any] = {"type": "pass"}


def _dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def template_of(action: Any) -> dict:
    """The action template inside a ``LegalAction``, or ``{}`` if this action
    isn't a structured one (an ordinary enumerated move, or a draft pick)."""
    payload = getattr(action, "input", None)
    if not isinstance(payload, dict):
        return {}
    template = payload.get("action")
    return template if isinstance(template, dict) else {}


def template_type(action: Any) -> str | None:
    """``'select_lineup'``, ``'doubles_turn'``, or ``None`` for an ordinary
    enumerated action whose ``input`` is already a finished ``play_action``
    argument set."""
    kind = template_of(action).get("type")
    return kind if isinstance(kind, str) else None


def roster_of(template: dict) -> list[str]:
    """The 6 species available to bring, in template order."""
    roster = template.get("roster")
    if not isinstance(roster, list):
        return []
    return [s for s in roster if isinstance(s, str) and s]


def slots_of(template: dict) -> list[dict]:
    """The active slots of a ``doubles_turn`` template, in slot order."""
    return _dicts(template.get("slots"))


def options_of(slot: dict) -> list[dict]:
    return _dicts(slot.get("options"))


def moves_of(slot: dict) -> list[dict]:
    return [o for o in options_of(slot) if o.get("type") == "move"]


def switches_of(slot: dict) -> list[dict]:
    return [o for o in options_of(slot) if o.get("type") == "switch"]


def has_pass(slot: dict) -> bool:
    return any(o.get("type") == "pass" for o in options_of(slot))


def target_for(move: dict) -> int | None:
    """The target to send with a move, or ``None`` to omit the field.

    Targets are seat-relative: positive numbers are opponents, so they are
    preferred; a move with no targets at all takes none.
    """
    targets = move.get("targets")
    if not isinstance(targets, list):
        return None
    seats = [t for t in targets if isinstance(t, int) and not isinstance(t, bool)]
    if not seats:
        return None
    opponents = [t for t in seats if t > 0]
    return opponents[0] if opponents else seats[0]


def move_choice(move: dict) -> dict | None:
    move_id = move.get("move_id")
    if not isinstance(move_id, str) or not move_id:
        return None
    choice: dict[str, Any] = {"type": "move", "move_id": move_id}
    target = target_for(move)
    if target is not None:
        choice["target"] = target
    return choice


def switch_choice(switch: dict) -> dict | None:
    species = switch.get("species")
    if not isinstance(species, str) or not species:
        return None
    return {"type": "switch", "species": species}
