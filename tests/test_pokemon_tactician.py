"""Tests for the Jev tactician: the full game goes out, only a typed choice
comes back, and every failure mode plays the deterministic #1."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from common.jev import Answer, JevClient
from pokemon import battle, draft, lineup, tactician
from pokemon.agent import PokemonAgent


class FakeJev:
    def __init__(self, answers=None):
        self.answers, self.calls, self.last = answers, 0, None
        self.last_error = None

    def ask(self, state, questions, **kw):
        self.calls += 1
        self.last = (state, questions)
        return self.answers


# -- client wiring -----------------------------------------------------------


def test_from_env_flag_is_per_game():
    env = {"TYPESAFE_API_KEY": "k"}
    assert JevClient.from_env(env, flag="POKEMON_JEV") is not None
    assert JevClient.from_env({**env, "POKEMON_JEV": "0"}, flag="POKEMON_JEV") is None
    # Red Alert's kill switch does not take Pokémon's tactician down with it.
    assert JevClient.from_env({**env, "RA_JEV": "0"}, flag="POKEMON_JEV") is not None
    assert JevClient.from_env({"TYPESAFE_API_KEY": " "}, flag="POKEMON_JEV") is None


# -- the choice plumbing -----------------------------------------------------


def test_a_confident_pick_is_taken_and_sent_typed():
    fake = FakeJev({"pick": Answer("choice", choice="PLAN_B", confidence=0.9)})
    t = tactician.Tactician(fake, log=lambda m: None)
    got = t.choose(
        state={"phase": "moving"},
        instructions="Which plan?",
        criteria={"PLAN_A": "attack", "PLAN_B": "protect"},
    )
    assert got == "PLAN_B"
    state, questions = fake.last
    assert state == {"phase": "moving"}
    question = questions["pick"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"PLAN_A", "PLAN_B"}


def test_every_shrug_falls_back_to_none():
    criteria = {"PLAN_A": "a", "PLAN_B": "b"}
    low = tactician.Tactician(
        FakeJev({"pick": Answer("choice", choice="PLAN_B", confidence=0.3)}), log=lambda m: None
    )
    assert low.choose(state={}, instructions="?", criteria=criteria) is None

    off_menu = tactician.Tactician(
        FakeJev({"pick": Answer("choice", choice="PLAN_Z", confidence=0.99)}), log=lambda m: None
    )
    assert off_menu.choose(state={}, instructions="?", criteria=criteria) is None

    dead = tactician.Tactician(FakeJev(None), log=lambda m: None)
    assert dead.choose(state={}, instructions="?", criteria=criteria) is None
    assert dead.errors == 1

    disabled = tactician.Tactician(None)
    assert disabled.choose(state={}, instructions="?", criteria=criteria) is None


# -- the state builders ------------------------------------------------------

# Real cards, as a draft state publishes them.
FLUTTER = {
    "card_id": "c-flutter",
    "species": "Flutter Mane",
    "item": "Booster Energy",
    "ability": "Protosynthesis",
    "nature": "Timid",
    "level": 50,
    "evs": {},
    "ivs": {},
    "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
}
URS = {
    "card_id": "c-urshifu",
    "species": "Urshifu-Rapid-Strike",
    "item": "Mystic Water",
    "ability": "Unseen Fist",
    "nature": "Adamant",
    "level": 50,
    "evs": {},
    "ivs": {},
    "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
}
GRA = {
    "card_id": "c-gyarados",
    "species": "Gyarados",
    "item": "Sitrus Berry",
    "ability": "Intimidate",
    "nature": "Adamant",
    "level": 50,
    "evs": {},
    "ivs": {},
    "moves": ["Waterfall", "Crunch", "Icy Wind", "Protect"],
}


def draft_obs():
    return {
        "phase": "draft",
        "pick_number": 1,
        "first_drafter": "us",
        "available_cards": [FLUTTER, URS, GRA],
        "available_card_ids": [c["card_id"] for c in (FLUTTER, URS, GRA)],
        "rosters": {"us": [], "them": []},
        "clock": {"decision_seconds_left": 12, "bank_seconds_left": 400},
    }


def test_draft_request_carries_the_full_picture():
    board = draft_obs()
    top = draft.top_cards(board, agent_id="us")
    assert len(top) == 3

    request = tactician.draft_request(board, "us", top)
    assert request.state["phase"] == "draft"
    assert request.state["rules"]["denial"] and request.state["rules"]["snake"]
    assert request.state["our_roster"] == [] and request.state["their_roster"] == []
    assert [entry["species"] for entry in request.state["candidates_detail"]] == [
        card["species"] for _, _, card in top
    ]
    # The criteria name the card, its set, and both halves of the swing math.
    for i, (card_id, score, card) in enumerate(top):
        text = request.criteria[tactician.LABELS[i]]
        assert card["species"] in text and card["item"] in text
        assert f"swing {score:.1f}" in text
    assert "denial" in request.criteria[tactician.LABELS[0]]


def mon(species, *, hp=100, max_hp=100, **extra):
    summary = {
        "species": species,
        "current_hp": hp,
        "max_hp": max_hp,
        "current_hp_fraction": hp / max_hp,
        "boosts": {},
        "fainted": hp <= 0,
    }
    summary.update(extra)
    return summary


def battle_board():
    return {
        "phase": "moving",
        "is_doubles": True,
        "turn": 2,
        "active_pokemon": [
            mon("Urshifu-Rapid-Strike", hp=176, max_hp=176, item="Mystic Water"),
            mon("Incineroar", hp=202, max_hp=202, ability="Intimidate", boosts={"atk": 1}),
        ],
        "opponent_active_pokemon": [
            mon("Flutter Mane", hp=131, max_hp=131),
            mon("Incineroar", hp=60, max_hp=202),
        ],
        "clock": {"decision_seconds_left": 40, "bank_seconds_left": 400},
    }


def test_battle_request_shows_the_board_damage_and_speed():
    board = battle_board()
    ours, theirs = battle.board(board)
    plans = [
        (
            {
                "type": "doubles_turn",
                "slot_0": {"type": "move", "move_id": "Close Combat", "target": 2},
                "slot_1": {"type": "move", "move_id": "Aqua Jet", "target": 1},
            },
            90.0,
            "focus fire",
        ),
        (
            {
                "type": "doubles_turn",
                "slot_0": {"type": "move", "move_id": "Surging Strikes", "target": 1},
                "slot_1": {"type": "pass"},
            },
            60.0,
            "elsewhere",
        ),
    ]
    request = tactician.battle_request(board, plans, ours, theirs)

    assert request.state["turn"] == 2
    assert request.state["our_active"][1]["ability"] == "Intimidate"
    assert request.state["our_active"][1]["boosts"] == {"atk": 1}
    assert request.state["their_active"][1]["hp_percent"] == 30
    # Flutter Mane (135 base speed) outpaces everyone here.
    assert request.state["speed_order"][0].startswith("Flutter Mane (them")

    # Damage facts are exact ranges, with the KO verdict on the wounded foe.
    plan_a = request.criteria["PLAN_A"]
    assert "Close Combat" in plan_a and "Incineroar" in plan_a
    assert "guaranteed KO" in plan_a  # 60/202 HP left into a 2x Close Combat
    assert "%" in plan_a and "engine score 90.0" in plan_a


def test_describe_plan_reports_ally_damage_for_spread_moves():
    board = {
        "phase": "moving",
        "active_pokemon": [
            mon("Gyarados", hp=200, max_hp=200),
            mon("Incineroar", hp=200, max_hp=200),
        ],
        "opponent_active_pokemon": [
            mon("Flutter Mane", hp=100, max_hp=100),
            mon("Chien-Pao", hp=100, max_hp=100),
        ],
    }
    ours, theirs = battle.board(board)
    text = tactician.describe_plan(
        {
            "type": "doubles_turn",
            "slot_0": {"type": "move", "move_id": "Surf", "target": 0},
            "slot_1": {"type": "pass"},
        },
        ours,
        theirs,
    )
    assert "both foes" in text and "also hits our partner" in text


# -- the fallback turn -------------------------------------------------------


def move_template(*slots):
    return {
        "type": "doubles_turn",
        "slots": [{"slot": i, "options": list(options)} for i, options in enumerate(slots)],
    }


def move(move_id, targets):
    return {"type": "move", "move_id": move_id, "targets": list(targets)}


SWITCH = {"type": "switch", "species": "gholdengo"}


def test_slots_fallback_picks_one_action_per_slot():
    t = tactician.Tactician(
        FakeJev(
            {
                "slot_0": Answer("choice", choice="move:Close Combat:2", confidence=0.9),
                "slot_1": Answer("choice", choice="switch:gholdengo", confidence=0.9),
            }
        ),
        log=lambda m: None,
    )
    turn = tactician.slots_fallback(
        move_template([move("Close Combat", [1, 2])], [SWITCH, {"type": "pass"}]),
        {},
        t,
    )
    assert turn == {
        "type": "doubles_turn",
        "slot_0": {"type": "move", "move_id": "Close Combat", "target": 2},
        "slot_1": {"type": "switch", "species": "gholdengo"},
    }
    # Both questions ride one request.
    _, questions = t.client.last
    assert sorted(questions) == ["slot_0", "slot_1"]


def test_slots_fallback_refuses_illegal_and_unconfident_pairs():
    duplicate = tactician.Tactician(
        FakeJev(
            {
                "slot_0": Answer("choice", choice="switch:Rillaboom", confidence=0.9),
                "slot_1": Answer("choice", choice="switch:Rillaboom", confidence=0.9),
            }
        ),
        log=lambda m: None,
    )
    both_switch = move_template(
        [{"type": "switch", "species": "Rillaboom"}], [{"type": "switch", "species": "Rillaboom"}]
    )
    assert tactician.slots_fallback(both_switch, {}, duplicate) is None

    shrug = tactician.Tactician(
        FakeJev(
            {
                "slot_0": Answer("choice", choice="move:Close Combat:2", confidence=0.2),
                "slot_1": Answer("choice", choice="switch:gholdengo", confidence=0.9),
            }
        ),
        log=lambda m: None,
    )
    assert (
        tactician.slots_fallback(move_template([move("Close Combat", [1, 2])], [SWITCH]), {}, shrug)
        is None
    )

    assert (
        tactician.slots_fallback(
            move_template([move("Close Combat", [1, 2])], [SWITCH]), {}, tactician.Tactician(None)
        )
        is None
    )


# -- the agent wiring --------------------------------------------------------


class FakeState:
    def __init__(self, body, legal):
        self.raw = {"observation": body}
        self.observation = ""
        self.legal_actions = legal


def test_the_agent_obeys_a_confident_jev_draft_pick(tmp_path):
    board = draft_obs()
    top = draft.top_cards(board, agent_id="us")
    assert len(top) >= 2

    fake = FakeJev({"pick": Answer("choice", choice=tactician.LABELS[1], confidence=0.95)})
    agent = PokemonAgent(pool_path=tmp_path / "pool.json", jev_client=fake)
    legal = [SimpleNamespace(action_id=f"draft_pick:{cid}") for cid, _, _ in top]
    got = agent.choose_action(FakeState(board, legal), SimpleNamespace(agent_id="us"))
    assert got.action_id == f"draft_pick:{top[1][0]}"


def test_the_agent_plays_the_math_when_jev_fails(tmp_path):
    board = draft_obs()
    top = draft.top_cards(board, agent_id="us")
    fake = FakeJev(None)
    agent = PokemonAgent(pool_path=tmp_path / "pool.json", jev_client=fake)
    legal = [SimpleNamespace(action_id=f"draft_pick:{cid}") for cid, _, _ in top]
    got = agent.choose_action(FakeState(board, legal), SimpleNamespace(agent_id="us"))
    assert got.action_id == f"draft_pick:{top[0][0]}"


def test_the_agent_obeys_a_confident_jev_lineup():
    species = [
        "Incineroar",
        "Flutter Mane",
        "Urshifu-Rapid-Strike",
        "Gyarados",
        "Rillaboom",
        "Whimsicott",
    ]
    board = {
        "phase": "team_preview",
        "your_roster": [
            {"species": s, "item": "Sitrus Berry", "ability": "Intimidate"} for s in species
        ],
        "opponent_roster": [
            {"species": "Flutter Mane", "item": "Choice Specs"},
            {"species": "Chien-Pao", "item": "Focus Sash"},
        ],
        "clock": {"decision_seconds_left": 60, "bank_seconds_left": 400},
    }
    template = {"type": "select_lineup", "roster": list(species)}
    expected = lineup.generate_candidate_lineups(template, board)
    assert len(expected) >= 2

    fake = FakeJev({"pick": Answer("choice", choice="PLAN_B", confidence=0.9)})
    agent = PokemonAgent(pool_path=Path("/nonexistent/pool.json"), jev_client=fake)
    legal = [SimpleNamespace(action_id="select_lineup", input={"action": template})]
    got = agent.choose_action(FakeState(board, legal), SimpleNamespace(agent_id="us"))
    assert got == expected[1][0]


def test_the_agent_obeys_a_confident_jev_battle_plan(monkeypatch, tmp_path):
    board = battle_board()
    template = move_template(
        [move("Close Combat", [1, 2]), move("Aqua Jet", [1, 2]), move("Surging Strikes", [1, 2])],
        [move("Aqua Jet", [1, 2])],
    )
    monkeypatch.setattr(battle, "is_tactical_dilemma", lambda *a, **k: True)

    fake = FakeJev({"pick": Answer("choice", choice="PLAN_B", confidence=0.9)})
    agent = PokemonAgent(pool_path=tmp_path / "pool.json", jev_client=fake)
    legal = [SimpleNamespace(action_id="doubles_turn", input={"action": template})]
    got = agent.choose_action(FakeState(board, legal), SimpleNamespace(agent_id="us"))
    expected = battle.generate_candidate_plans(template, board)
    assert len(expected) >= 2
    assert got == expected[1][0]
