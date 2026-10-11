"""Example: a deterministic infrastructure smoke-test agent — valid moves, no strategy.

Plays every phase of `pokemon_vgc_doubles_draft` (draft -> Team Preview ->
doubles battle) and all of Werewolf by building the simplest valid action
from what GameAPI offers, so a live self-hosted test match can run end to
end. No LLM, no API keys, no randomness, no memory between turns. It does
not play Red Alert (no `legal_actions` there; see `llm_agent.py`).

- Anything enumerable (Pokémon draft picks, every Werewolf action):
  `state.legal_actions[0]`, exactly like `basic_agent.py`.
- Pokémon Team Preview (`select_lineup`): the first 4 species of the roster
  in the action template, leading with the first 2.
- Pokémon doubles turn (`doubles_turn`): per slot, pass if pass is the only
  option; else the first move (first positive target — an opponent — if the
  move takes one, otherwise its first target, omitted when it has none);
  else the first switch the other slot didn't already pick; else pass if
  offered.

Both Pokémon templates arrive as a single legal action whose
`input["action"]` describes what to submit (Agent_ACP gameapi
`pokemon_adapter/teampreview_mapper.py` `teampreview_action_schema`,
`pokemon_adapter/doubles_action_mapper.py` `build_doubles_legal_actions`).
A template that doesn't have the expected shape raises `SmokeAgentError`
(the runner reports it as a DecisionError) rather than sending a guess.

Run it for your test matches (add --tournament for your tournament games):

    python -m agent --match --agent examples.smoke_agent
"""

from altruagent import DecisionContext, GameState, LegalAction

_PASS = {"type": "pass"}


class SmokeAgentError(ValueError):
    """The server offered an action template this agent can't read."""


def choose_action(state: GameState, context: DecisionContext) -> LegalAction | dict:
    if not state.legal_actions:
        raise SmokeAgentError("choose_action was called with no legal actions.")
    first = state.legal_actions[0]
    if first.action_id == "select_lineup":
        return _select_lineup(_template(first, "select_lineup"))
    if first.action_id == "doubles_turn":
        return _doubles_turn(_template(first, "doubles_turn"))
    if first.action_id == "submit_team":
        raise SmokeAgentError(
            "submit_team (build-your-own-team modes) needs a full 6-Pokémon team; "
            "the smoke agent only plays pokemon_vgc_doubles_draft."
        )
    return first


def create_agent():
    return choose_action


def _template(action: LegalAction, expected_type: str) -> dict:
    template = action.input.get("action")
    if not isinstance(template, dict) or template.get("type") != expected_type:
        raise SmokeAgentError(
            f"{expected_type} legal action has no {expected_type!r} template in "
            f"input['action']: {template!r}"
        )
    return template


def _select_lineup(template: dict) -> dict:
    roster = template.get("roster")
    if (
        not isinstance(roster, list)
        or len(roster) < 4
        or not all(isinstance(species, str) and species for species in roster)
        or len(set(roster[:4])) != 4
    ):
        raise SmokeAgentError(
            f"select_lineup template needs a roster of at least 4 distinct species, got {roster!r}"
        )
    return {"type": "select_lineup", "bring": roster[:4], "leads": roster[:2]}


def _doubles_turn(template: dict) -> dict:
    slots = template.get("slots")
    if not isinstance(slots, list) or len(slots) != 2:
        raise SmokeAgentError(f"doubles_turn template needs exactly 2 slots, got {slots!r}")

    by_slot: dict[int, list[dict]] = {}
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise SmokeAgentError(f"doubles_turn slot {index} is not an object: {slot!r}")
        number = slot.get("slot", index)
        options = slot.get("options")
        if number not in (0, 1) or number in by_slot:
            raise SmokeAgentError(f"doubles_turn slots must be numbered 0 and 1, got {slots!r}")
        if not isinstance(options, list) or not options or not all(
            isinstance(option, dict) and option.get("type") in ("move", "switch", "pass")
            for option in options
        ):
            raise SmokeAgentError(f"doubles_turn slot {number} has unreadable options: {options!r}")
        by_slot[number] = options

    slot_0 = _slot_choice(0, by_slot[0], taken_species=None)
    taken = slot_0["species"] if slot_0["type"] == "switch" else None
    slot_1 = _slot_choice(1, by_slot[1], taken_species=taken)
    return {"type": "doubles_turn", "slot_0": slot_0, "slot_1": slot_1}


def _slot_choice(number: int, options: list[dict], *, taken_species: str | None) -> dict:
    kinds = {option["type"] for option in options}
    if kinds == {"pass"}:
        return dict(_PASS)

    for option in options:
        if option["type"] == "move":
            return _move_choice(number, option)

    for option in options:
        if option["type"] == "switch":
            species = option.get("species")
            if not isinstance(species, str) or not species:
                raise SmokeAgentError(f"doubles_turn slot {number} switch has no species: {option!r}")
            if species != taken_species:
                return {"type": "switch", "species": species}

    if "pass" in kinds:
        return dict(_PASS)
    raise SmokeAgentError(
        f"doubles_turn slot {number} has no move, no distinct switch, and no pass: {options!r}"
    )


def _move_choice(number: int, option: dict) -> dict:
    move_id = option.get("move_id")
    targets = option.get("targets", [])
    if not isinstance(move_id, str) or not move_id:
        raise SmokeAgentError(f"doubles_turn slot {number} move has no move_id: {option!r}")
    if not isinstance(targets, list) or not all(
        isinstance(target, int) and not isinstance(target, bool) for target in targets
    ):
        raise SmokeAgentError(f"doubles_turn slot {number} move has unreadable targets: {option!r}")

    choice = {"type": "move", "move_id": move_id}
    if targets:
        choice["target"] = next((target for target in targets if target > 0), targets[0])
    return choice
