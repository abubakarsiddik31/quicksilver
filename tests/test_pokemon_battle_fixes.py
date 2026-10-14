"""Regression tests for the battle-policy weaknesses found reading the code:
no threat model, memoryless status/setup moves, spread and accuracy handling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from pokemon import battle, dex, lineup, threat
from pokemon.damage import parse_card


def mon(species, *, hp=100, max_hp=100, moves=None, **extra):
    s = {
        "species": species,
        "current_hp": hp,
        "max_hp": max_hp,
        "current_hp_fraction": hp / max_hp,
        "boosts": {},
        "fainted": hp <= 0,
    }
    if moves is not None:
        s["moves"] = moves
    s.update(extra)
    return s


def obs(ours, theirs, turn=3, **extra):
    return {
        "phase": "moving",
        "is_doubles": True,
        "turn": turn,
        "active_pokemon": list(ours),
        "opponent_active_pokemon": list(theirs),
        **extra,
    }


def ctx_for(ours, theirs, memory=None, **extra):
    return battle.make_ctx(obs(ours, theirs, **extra), memory=memory)


def score(move_id, ctx, slot=0, targets=(0,)):
    s, t = battle.move_score(
        move_id,
        attacker=ctx.ours[slot],
        defenders=ctx.theirs,
        targets=list(targets),
        turn=ctx.turn,
        partner=battle._partner(ctx.ours, slot),
        ctx=ctx,
        slot_index=slot,
    )
    return s, t


GARCHOMP = mon("Garchomp", moves=["Earthquake", "Dragon Claw"])
FLUTTER = mon("Flutter Mane", moves=["Moonblast", "Shadow Ball"])
INCI = mon("Incineroar", moves=["Flare Blitz", "Knock Off"])


# -- setup moves ---------------------------------------------------------------


def test_setup_moves_stop_once_boosted():
    bulky = mon("Corviknight", moves=["Brave Bird"])  # Steel / Flying
    foes = [mon("Amoonguss"), mon("Amoonguss")]
    fresh = ctx_for([bulky, INCI], foes)
    boosted = ctx_for([{**bulky, "boosts": {"atk": 2}}, INCI], foes)
    assert score("Swords Dance", fresh)[0] == 35.0
    assert score("Swords Dance", boosted)[0] < 5.0
    threatened = ctx_for([GARCHOMP, INCI], [FLUTTER, INCI])
    assert score("Swords Dance", threatened)[0] < 35.0  # no setting up while about to be hit hard


# -- field moves ---------------------------------------------------------------


def test_tailwind_is_worthless_while_ours_is_up():
    up = ctx_for([GARCHOMP, INCI], [FLUTTER, INCI], side_conditions={"tailwind": 3})
    assert score("Tailwind", up)[0] == 0.0
    assert score("Tailwind", ctx_for([GARCHOMP, INCI], [FLUTTER, INCI], turn=1))[0] == 42.0


def test_trick_room_is_not_reused_to_undo_itself():
    slow = [mon("Torkoal", moves=["Eruption"]), mon("Amoonguss")]
    fast = [mon("Flutter Mane"), mon("Dragapult")]
    assert score("Trick Room", ctx_for(slow, fast, turn=1))[0] > 30
    assert (
        score("Trick Room", ctx_for(slow, fast, fields=["Trick Room"]))[0] == 0.0
    )  # we want it up
    assert (
        score("Trick Room", ctx_for(fast, slow, fields=["Trick Room"]))[0] > 0
    )  # ending it helps the fast side


def test_a_screen_already_up_is_not_set_again():
    up = ctx_for([GARCHOMP, INCI], [FLUTTER, INCI], side_conditions=["Reflect"])
    assert score("Reflect", up)[0] == 0.0
    assert score("Light Screen", up)[0] > 0.0


def test_two_slots_never_set_the_same_field_move():
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "move", "move_id": "Tailwind", "targets": [0]},
                    {"type": "move", "move_id": "Protect", "targets": [0]},
                ],
            },
            {
                "slot": 1,
                "options": [
                    {"type": "move", "move_id": "Tailwind", "targets": [0]},
                    {"type": "move", "move_id": "Protect", "targets": [0]},
                ],
            },
        ],
    }
    plans = battle.generate_candidate_plans(
        template, obs([mon("Talonflame"), mon("Tornadus")], [FLUTTER, INCI], turn=1)
    )
    top = plans[0][0]
    assert not (top["slot_0"]["move_id"] == top["slot_1"]["move_id"] == "Tailwind")


# -- Protect and Fake Out memory -----------------------------------------------


def test_protect_is_not_repeated_back_to_back():
    memory = threat.Memory()
    state = [mon("Urshifu-Rapid-Strike", hp=20, max_hp=175), INCI]
    memory.observe(state, 2)
    memory.record(state, {"slot_0": {"type": "move", "move_id": "Protect"}})
    ctx = ctx_for(state, [FLUTTER, INCI], memory=memory)
    assert score("Protect", ctx)[0] < 0
    assert score("Protect", ctx_for(state, [FLUTTER, INCI]))[0] >= 45


def test_fake_out_only_counts_the_turn_a_pokemon_comes_in():
    memory = threat.Memory()
    memory.observe([INCI, GARCHOMP], 1)
    memory.observe([INCI, GARCHOMP], 4)  # still out on turn 4
    stale = ctx_for([INCI, GARCHOMP], [FLUTTER, INCI], memory=memory, turn=4)
    assert score("Fake Out", stale, targets=(2,))[0] < battle.FAKE_OUT_TURN_ONE_BONUS
    memory.observe([GARCHOMP, mon("Rillaboom")], 5)
    memory.observe([INCI, GARCHOMP], 6)  # Incineroar switched back in
    fresh = ctx_for([INCI, GARCHOMP], [FLUTTER, INCI], memory=memory, turn=6)
    assert score("Fake Out", fresh, targets=(2,))[0] >= 50


def test_fake_out_is_not_valued_against_a_ghost():
    ctx = ctx_for([INCI, GARCHOMP], [mon("Gengar"), mon("Gholdengo")], turn=1)
    assert score("Fake Out", ctx, targets=(1, 2))[0] < 10


# -- crippling status ------------------------------------------------------------


def test_status_moves_respect_immunities_and_existing_status():
    ours = [mon("Grimmsnarl", moves=["Thunder Wave"]), INCI]
    ground = ctx_for(ours, [mon("Garchomp"), mon("Landorus-Therian")])
    assert score("Thunder Wave", ground, targets=(1, 2))[0] == 0.0
    statused = ctx_for(ours, [mon("Flutter Mane", status="par"), mon("Landorus-Therian")])
    assert (
        score("Thunder Wave", statused, targets=(1, 2))[1] == 2
        or score("Thunder Wave", statused, targets=(1, 2))[0] == 0.0
    )
    fire = ctx_for(ours, [mon("Incineroar"), mon("Gholdengo")])
    assert score("Will-O-Wisp", fire, targets=(1, 2))[1] != 1


# -- spread moves and accuracy ---------------------------------------------------


def test_a_spread_move_is_credited_for_every_foe_it_hits():
    two = ctx_for(
        [mon("Gardevoir", moves=["Dazzling Gleam"]), INCI], [mon("Garchomp"), mon("Incineroar")]
    )
    one = ctx_for(
        [mon("Gardevoir", moves=["Dazzling Gleam"]), INCI],
        [mon("Garchomp"), mon("Incineroar", hp=0)],
    )
    total = score("Dazzling Gleam", two, targets=(0,))[0]
    garchomp_only = threat.calc(
        two.ours[0], two.theirs[0], "Dazzling Gleam", two.field, defender_is_ours=False, spread=True
    )
    assert total > garchomp_only.max_percent  # both foes count, not just the best one
    # With one foe left there is no spread penalty on the survivor.
    lone = threat.calc(
        one.ours[0],
        one.theirs[0],
        "Dazzling Gleam",
        one.field,
        defender_is_ours=False,
        spread=False,
    )
    assert lone.max_percent > garchomp_only.max_percent


def test_inaccurate_moves_are_discounted():
    assert dex.move_accuracy("Thunder") == 0.7
    assert dex.move_accuracy("Moonblast") == 1.0
    assert dex.move_accuracy("Aura Sphere") == 1.0  # never misses
    foe = mon("Incineroar", hp=170, max_hp=170)
    ctx = ctx_for(
        [mon("Flutter Mane", moves=["Thunder", "Thunderbolt"]), INCI], [foe, mon("Skarmory")]
    )
    thunder = threat.calc(ctx.ours[0], ctx.theirs[0], "Thunder", ctx.field, defender_is_ours=False)
    bolt = threat.calc(ctx.ours[0], ctx.theirs[0], "Thunderbolt", ctx.field, defender_is_ours=False)
    assert thunder.max_percent > bolt.max_percent  # raw power says Thunder
    assert (
        score("Thunder", ctx, targets=(1,))[0] < score("Thunderbolt", ctx, targets=(1,))[0]
    )  # accuracy says no


def test_weather_and_screens_reach_the_damage_calculation():
    plain = ctx_for(
        [mon("Pelipper", moves=["Hydro Pump"]), INCI], [mon("Incineroar"), mon("Garchomp")]
    )
    rain = ctx_for(
        [mon("Pelipper", moves=["Hydro Pump"]), INCI],
        [mon("Incineroar"), mon("Garchomp")],
        weather="Rain",
    )
    walled = ctx_for(
        [mon("Pelipper", moves=["Hydro Pump"]), INCI],
        [mon("Incineroar"), mon("Garchomp")],
        opponent_side_conditions=["Light Screen"],
    )
    base = score("Hydro Pump", plain, targets=(1,))[0]
    assert score("Hydro Pump", rain, targets=(1,))[0] > base
    assert score("Hydro Pump", walled, targets=(1,))[0] < base


# -- threat ----------------------------------------------------------------------


def test_a_slot_the_foes_can_kill_is_flagged_and_its_attack_is_discounted():
    # Fragile, slower Garchomp at 20% facing a Flutter Mane with Moonblast (Fairy-neutral).
    weak = mon("Dragonite", hp=20, max_hp=100, moves=["Dragon Claw"])
    ctx = ctx_for([weak, INCI], [FLUTTER, mon("Iron Hands", moves=["Close Combat"])])
    assert ctx.danger.risk_of(0) >= 0.9
    assert ctx.danger.doomed_of(0)
    calm = dataclasses.replace(ctx, danger=threat.Danger())
    assert (
        score("Dragon Claw", ctx, targets=(2,))[0]
        == score("Dragon Claw", calm, targets=(2,))[0] * battle.DOOMED_FACTOR
    )
    # A priority move still gets to go first, so it is not discounted.
    assert (
        score("Extreme Speed", ctx, targets=(2,))[0]
        == score("Extreme Speed", calm, targets=(2,))[0]
    )


def test_switching_needs_a_reason_and_a_resilient_target():
    weak = mon("Dragonite", hp=20, max_hp=100, moves=["Dragon Claw"])
    moonblast = mon("Flutter Mane", moves=["Moonblast"])
    threatened = ctx_for([weak, INCI], [moonblast, mon("Amoonguss", moves=["Giga Drain"])])
    calm = ctx_for(
        [mon("Dragonite", moves=["Dragon Claw"]), INCI],
        [mon("Amoonguss", moves=["Giga Drain"]), mon("Amoonguss", moves=["Giga Drain"])],
    )
    opt = {"type": "switch", "species": "Gholdengo"}  # Steel resists Fairy
    kw = dict(slot_index=0, taken_switches=set(), turn=3)
    hot = battle._option_score(
        opt, ours=threatened.ours, theirs=threatened.theirs, ctx=threatened, **kw
    )
    cold = battle._option_score(opt, ours=calm.ours, theirs=calm.theirs, ctx=calm, **kw)
    assert hot > cold
    assert cold < hot / 2


def test_jev_is_not_skipped_on_a_lopsided_board_when_we_are_threatened():
    plans = [({}, 90.0, ""), ({}, 20.0, "")]
    calm = threat.Danger((0.1, 0.0), (False, False))
    hot = threat.Danger((0.9, 0.0), (True, False))
    assert not battle.is_tactical_dilemma(plans, {}, danger=calm)
    assert battle.is_tactical_dilemma(plans, {}, danger=hot)
    assert battle.is_tactical_dilemma([({}, 50.0, ""), ({}, 40.0, "")], {}, danger=calm)


def test_choose_turn_uses_the_joint_overkill_logic():
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [{"type": "move", "move_id": "Surging Strikes", "targets": [1]}],
            },
            {
                "slot": 1,
                "options": [
                    {"type": "move", "move_id": "Flare Blitz", "targets": [1]},
                    {"type": "move", "move_id": "Flare Blitz", "targets": [2]},
                ],
            },
        ],
    }
    board = obs(
        [mon("Urshifu-Rapid-Strike", hp=175, max_hp=175), mon("Incineroar", hp=200, max_hp=200)],
        [mon("Incineroar", hp=30, max_hp=200), mon("Gholdengo", hp=160, max_hp=160)],
    )
    assert battle.choose_turn(template, board)["slot_1"]["target"] == 2


def test_diverse_candidates_include_a_protect_plan():
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": i,
                "options": [
                    {"type": "move", "move_id": "Surging Strikes", "targets": [1, 2]},
                    {"type": "move", "move_id": "Close Combat", "targets": [1, 2]},
                    {"type": "move", "move_id": "Aqua Jet", "targets": [1, 2]},
                    {"type": "move", "move_id": "Protect", "targets": [0]},
                ],
            }
            for i in (0, 1)
        ],
    }
    board = obs([mon("Urshifu-Rapid-Strike"), mon("Urshifu-Rapid-Strike")], [FLUTTER, INCI])
    plans = battle.generate_candidate_plans(template, board, max_candidates=3, diverse=True)
    assert any(battle._uses_protect(p) for p, _, _ in plans)
    assert len(battle.generate_candidate_plans(template, board, max_candidates=3)) == 3


# -- unknown foes and Team Preview -------------------------------------------------


def test_an_unseen_foe_still_counts_as_a_threat():
    ctx = ctx_for(
        [mon("Urshifu-Rapid-Strike", hp=30, max_hp=175), INCI], [mon("Flutter Mane"), INCI]
    )
    assert ctx.danger.risk_of(0) > 0.5


def test_team_preview_reads_the_foes_real_moves():
    stab_only = parse_card({"species": "Flutter Mane"})  # types Ghost/Fairy, set unknown
    ground_set = parse_card({"species": "Flutter Mane", "moves": ["Earthquake"]})
    skarmory = parse_card({"species": "Skarmory"})  # Steel / Flying: immune to Ground
    assert (
        lineup.defensive_risk(skarmory, stab_only) == 1.0
    )  # unknown set: guessed from Ghost/Fairy typing
    assert (
        lineup.defensive_risk(skarmory, ground_set) == 0.0
    )  # known set: Earthquake cannot touch it


def test_the_agent_remembers_its_protect_between_turns():
    from pokemon.agent import PokemonAgent

    agent = PokemonAgent(pool_path=Path("/nonexistent/pool.json"), jev_client=None)
    ours = [mon("Urshifu-Rapid-Strike", hp=15, max_hp=175), mon("Incineroar", hp=170, max_hp=170)]
    theirs = [FLUTTER, mon("Iron Hands", moves=["Close Combat"])]
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "move", "move_id": "Protect", "targets": [0]},
                    {"type": "move", "move_id": "Close Combat", "targets": [1, 2]},
                ],
            },
            {"slot": 1, "options": [{"type": "move", "move_id": "Knock Off", "targets": [1, 2]}]},
        ],
    }

    class State:
        raw = {"observation": obs(ours, theirs, turn=2)}
        observation = ""

    first = agent._doubles_turn(template, State())
    assert first["slot_0"]["move_id"] == "Protect"
    State.raw = {"observation": obs(ours, theirs, turn=3)}
    second = agent._doubles_turn(template, State())
    assert second["slot_0"]["move_id"] != "Protect"


# -- new cards ---------------------------------------------------------------------

from pokemon import learn, pool  # noqa: E402


def _card(card_id, species, moves, **extra):
    return {
        "card_id": card_id,
        "species": species,
        "item": None,
        "ability": None,
        "nature": "Hardy",
        "level": 50,
        "evs": {},
        "ivs": {},
        "moves": moves,
        **extra,
    }


def test_this_matches_cards_beat_a_stale_banked_set():
    banked = [_card("vgc-incineroar", "Incineroar", ["Fake Out"])]
    match = pool.MatchCards()
    match.observe(
        {"available_cards": [_card("vgc-incineroar", "Incineroar", ["Flare Blitz", "Knock Off"])]}
    )
    cards = match.with_fallback(banked)
    assert battle._find_in_pool("Incineroar", cards)["moves"] == ["Flare Blitz", "Knock Off"]
    # A species this match never showed still comes from the bank.
    banked.append(_card("vgc-rillaboom", "Rillaboom", ["Grassy Glide"]))
    assert battle._find_in_pool("Rillaboom", match.with_fallback(banked))["moves"] == [
        "Grassy Glide"
    ]


def test_a_brand_new_card_is_read_from_the_drafts_rosters_not_the_file():
    match = pool.MatchCards()
    match.observe(
        {
            "rosters": {
                "me": [_card("new-1", "Gholdengo", ["Make It Rain"])],
                "them": [_card("new-2", "Gholdengo", ["Shadow Ball"])],
            }
        },
        agent_id="me",
    )
    cards = match.with_fallback(None)
    # Two Gholdengo cards: each side gets its own.
    assert battle._find_in_pool("Gholdengo", cards, "ours")["moves"] == ["Make It Rain"]
    assert battle._find_in_pool("Gholdengo", cards, "theirs")["moves"] == ["Shadow Ball"]
    ctx = battle.make_ctx(obs([mon("Gholdengo")], [mon("Gholdengo")]), pool_cards=cards)
    assert ctx.ours[0].card.moves == ("Make It Rain",)
    assert ctx.theirs[0].card.moves == ("Shadow Ball",)


def test_the_dex_learns_a_species_and_move_it_has_never_heard_of():
    assert dex.types_of("Zorblax") == ()
    observed = obs(
        [
            mon(
                "Zorblax",
                types=["Fire", "Ghost"],
                base_stats={"hp": 80, "atk": 100, "def": 80, "spa": 120, "spd": 80, "spe": 110},
            )
        ],
        [INCI],
        available_moves=[
            [
                {
                    "id": "zorbblast",
                    "type": "Ghost",
                    "category": "Special",
                    "base_power": 90,
                    "accuracy": 100,
                    "priority": 0,
                }
            ],
            [],
        ],
    )
    learn.from_observation(observed)
    assert dex.types_of("Zorblax") == ("Fire", "Ghost")
    assert dex.base_stats("Zorblax")["spa"] == 120
    assert dex.is_damaging("Zorbblast") and dex.move_type("Zorbblast") == "Ghost"
    card = parse_card({"species": "Zorblax", "moves": ["Zorbblast"]})
    assert card.stats["spa"] > 0
    # The vendored data is never overridden.
    learn.from_observation(
        obs(
            [
                mon(
                    "Incineroar",
                    types=["Water"],
                    base_stats={"hp": 1, "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1},
                )
            ],
            [],
        )
    )
    assert dex.types_of("Incineroar") == ("Fire", "Dark")


def test_the_agent_uses_a_card_that_only_the_draft_showed():
    from pokemon.agent import PokemonAgent

    agent = PokemonAgent(pool_path=Path("/nonexistent/pool.json"), jev_client=None)

    class Draft:
        raw = {
            "observation": {
                "phase": "draft",
                "available_cards": [],
                "rosters": {
                    "me": [_card("x", "Incineroar", ["Fake Out", "Flare Blitz"])],
                    "them": [],
                },
            }
        }
        observation = ""

    agent._agent_id = "me"
    agent._learn(Draft())
    assert agent._cards()[0]["moves"] == ["Fake Out", "Flare Blitz"]
