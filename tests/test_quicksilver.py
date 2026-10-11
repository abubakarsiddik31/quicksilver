"""Offline tests for Quicksilver's own decision layer.

These cover the code we wrote — dispatch, the Pokémon template readers and the
Werewolf baseline — with no game server and no API key, so they keep working
after Testing closes (see ``SUMMARY.md``). The vendored starter's own suite is
in the other ``test_*.py`` files.
"""

from __future__ import annotations

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, LegalAction
from common.dispatch import QuicksilverAgent, create_agent
from pokemon import actions
from pokemon.agent import PokemonAgent
from werewolf.agent import WerewolfAgent

DOUBLES = "pokemon_vgc_doubles_draft"


def make_state(*legal: LegalAction, game_name: str = DOUBLES) -> GameState:
    return GameState(
        session_id="s1",
        game_name=game_name,
        status="in_progress",
        observation="",
        current_player=None,
        legal_actions=list(legal),
        is_terminal=False,
        is_current_actor=True,
        state_version=3,
        returns=None,
        move_count=0,
        termination_reason=None,
        messaging_enabled=False,
        phase="moving",
        next_actions=[],
        new_messages=[],
        terminated_messaging=[],
        messaging_mode="",
    )


def make_context(game_type: str | None) -> DecisionContext:
    return DecisionContext(
        session_id="s1", tournament_id=None, game_type=game_type, agent_id="a1"
    )


def enumerated(action_id: str = "0") -> LegalAction:
    """A draft pick / Werewolf move: finished ``play_action`` arguments."""
    return LegalAction(action_id=action_id, label=None, input={"action_id": action_id})


def structured(kind: str, **template) -> LegalAction:
    return LegalAction(action_id=kind, label=None, input={"action": {"type": kind, **template}})


# -- dispatch ---------------------------------------------------------------


def test_create_agent_returns_the_dispatcher():
    assert isinstance(create_agent(), QuicksilverAgent)


def test_routes_each_game_type_to_its_own_policy():
    for game_type, expected in (
        (DOUBLES, PokemonAgent),
        ("werewolf", WerewolfAgent),
    ):
        agent = QuicksilverAgent()
        agent.choose_action(make_state(enumerated()), make_context(game_type))
        assert isinstance(agent._delegate, expected)
        assert agent.game_type == game_type


def test_unknown_game_type_gets_the_safe_fallback():
    """An unentered game (Red Alert) must still produce a legal move rather
    than crashing the match process."""
    agent = QuicksilverAgent()
    result = agent.choose_action(make_state(enumerated("0")), make_context("red_alert"))
    assert result is not None


def test_choose_message_defaults_to_terminating_the_discussion():
    """The Pokémon delegate defines no ``choose_message``, so the dispatcher
    must supply the runtime's own default (vote the window closed)."""
    agent = QuicksilverAgent()
    result = agent.choose_message(make_state(enumerated()), make_context(DOUBLES))
    assert result is TERMINATE_MESSAGING


def test_choose_message_forwards_to_a_delegate_that_has_one():
    class Talker:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return "spoken"

    agent = QuicksilverAgent()
    agent._delegate = Talker()
    assert agent.choose_message(make_state(enumerated()), make_context("werewolf")) == "spoken"


# -- Pokémon: Team Preview --------------------------------------------------


def test_team_preview_brings_four_and_leads_two():
    action = structured("select_lineup", roster=["a", "b", "c", "d", "e", "f"])
    result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
    assert result == {"type": "select_lineup", "bring": ["a", "b", "c", "d"], "leads": ["a", "b"]}


def test_team_preview_with_a_short_roster_does_not_raise():
    action = structured("select_lineup", roster=["a", "b"])
    result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
    assert isinstance(result, dict)


# -- Pokémon: doubles turns -------------------------------------------------


def test_doubles_turn_prefers_moves_and_targets_an_opponent():
    action = structured(
        "doubles_turn",
        slots=[
            {"slot": 0, "options": [{"type": "move", "move_id": "protect", "targets": [0]}]},
            {"slot": 1, "options": [{"type": "move", "move_id": "moonblast", "targets": [1, 2]}]},
        ],
    )
    result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
    # A self-targeting move keeps its own (non-positive) target; an opponent
    # target is preferred when the move offers one.
    assert result["slot_0"] == {"type": "move", "move_id": "protect", "target": 0}
    assert result["slot_1"] == {"type": "move", "move_id": "moonblast", "target": 1}


def test_two_slots_never_switch_to_the_same_species():
    action = structured(
        "doubles_turn",
        slots=[
            {"slot": 0, "options": [{"type": "switch", "species": "Incineroar"}]},
            {
                "slot": 1,
                "options": [
                    {"type": "switch", "species": "Incineroar"},
                    {"type": "pass"},
                ],
            },
        ],
    )
    result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
    assert result["slot_0"] == {"type": "switch", "species": "Incineroar"}
    assert result["slot_1"] == {"type": "pass"}


def test_a_pass_only_slot_passes():
    action = structured(
        "doubles_turn",
        slots=[
            {"slot": 0, "options": [{"type": "pass"}]},
            {"slot": 1, "options": [{"type": "move", "move_id": "tackle"}]},
        ],
    )
    result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
    assert result["slot_0"] == {"type": "pass"}


def test_a_malformed_doubles_template_does_not_raise():
    """A DecisionError would kill the match worker, so an unreadable template
    must degrade instead."""
    for slots in ([], [{"slot": 0}], "nonsense", [{"slot": 9, "options": []}]):
        action = structured("doubles_turn", slots=slots)
        result = PokemonAgent().choose_action(make_state(action), make_context(DOUBLES))
        assert isinstance(result, dict)


# -- Pokémon: draft ---------------------------------------------------------


def test_a_draft_pick_returns_the_enumerated_action():
    picks = [enumerated("draft_pick:card-1"), enumerated("draft_pick:card-2")]
    result = PokemonAgent().choose_action(make_state(*picks), make_context(DOUBLES))
    assert result is picks[0]


def test_no_legal_actions_does_not_raise():
    assert isinstance(PokemonAgent().choose_action(make_state(), make_context(DOUBLES)), dict)


# -- readers are total ------------------------------------------------------


def test_readers_tolerate_garbage_without_raising():
    for value in (None, {}, [], "text", 7, {"slots": "x"}, {"roster": 3}):
        assert actions.template_of(value) == {}
        assert actions.template_type(value) is None
        assert actions.roster_of(value if isinstance(value, dict) else {}) == []
        assert actions.slots_of(value if isinstance(value, dict) else {}) == []
    assert actions.move_choice({}) is None
    assert actions.switch_choice({"species": ""}) is None
    assert actions.target_for({"targets": ["a"]}) is None


# -- pool capture is automatic ----------------------------------------------


def test_a_draft_state_banks_the_card_pool(tmp_path):
    """The pool is published nowhere but the draft observation, so the agent
    must capture it the first time it sees one — that single sighting is what
    makes the matchup precomputation offline work."""
    from pokemon import pool
    from pokemon.agent import PokemonAgent

    target = tmp_path / "pool.json"
    card = {"card_id": "vgc-gyarados", "species": "Gyarados", "item": "Mystic Water"}
    state = make_state(enumerated("draft_pick:vgc-gyarados"))
    state.raw = {"observation": {"phase": "draft", "available_cards": [card]}}

    PokemonAgent(pool_path=target).choose_action(state, make_context(DOUBLES))

    banked = pool.load_pool(target)
    assert banked is not None and banked[0]["card_id"] == "vgc-gyarados"


def test_a_broken_pool_write_never_costs_the_match(tmp_path):
    """Bookkeeping must never raise: a DecisionError here would kill the
    worker and hand the battle to auto-play."""
    from pokemon.agent import PokemonAgent

    unwritable = tmp_path / "nope" / "deeper" / "pool.json"  # parent doesn't exist
    state = make_state(enumerated("draft_pick:x"))
    state.raw = {"observation": {"phase": "draft", "available_cards": [{"card_id": "x"}]}}

    result = PokemonAgent(pool_path=unwritable).choose_action(state, make_context(DOUBLES))
    assert result is not None
