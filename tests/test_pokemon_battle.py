"""Tests for doubles turn selection.

These assert the policy's *decisions*, not just that it runs: it should pick the
hardest-hitting move, target the Pokémon it actually threatens, and respect the
server's two structural rules (no duplicate switch, no double pass).
"""

from __future__ import annotations

from pokemon import battle

URSHIFU = {
    "card_id": "u",
    "species": "Urshifu-Rapid-Strike",
    "item": "Mystic Water",
    "ability": "Unseen Fist",
    "nature": "Adamant",
    "level": 50,
    "evs": {"hp": 4, "atk": 252, "spe": 252},
    "ivs": {},
    "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
}
INCINEROAR = {
    "card_id": "i",
    "species": "Incineroar",
    "item": "Sitrus Berry",
    "ability": "Intimidate",
    "nature": "Careful",
    "level": 50,
    "evs": {"hp": 252, "def": 4, "spd": 252},
    "ivs": {},
    "moves": ["Flare Blitz", "Knock Off", "Fake Out", "Parting Shot"],
}


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


def obs(*, ours, theirs, turn=2):
    return {
        "phase": "moving",
        "is_doubles": True,
        "turn": turn,
        "active_pokemon": list(ours),
        "opponent_active_pokemon": list(theirs),
    }


def template(*slots):
    return {
        "type": "doubles_turn",
        "slots": [{"slot": i, "options": list(options)} for i, options in enumerate(slots)],
    }


def move(move_id, targets):
    return {"type": "move", "move_id": move_id, "targets": list(targets)}


SWITCH = {"type": "switch", "species": "gholdengo"}
PASS = {"type": "pass"}


# -- move choice ------------------------------------------------------------


def test_it_picks_the_hardest_hitting_move():
    """Urshifu's Close Combat (120 BP, 2x into Incineroar) must beat Aqua Jet."""
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Incineroar", hp=202, max_hp=202)],
    )
    turn = battle.choose_turn(
        template(
            [
                move("Close Combat", [1, 2]),
                move("Aqua Jet", [1, 2]),
                move("Surging Strikes", [1, 2]),
            ],
            [move("Aqua Jet", [1, 2])],
        ),
        board,
    )
    assert turn["slot_0"]["move_id"] == "Close Combat"


def test_it_targets_the_opponent_it_threatens():
    """Close Combat is 2x into Incineroar and 0x into a Ghost, so the target
    must be the Incineroar slot even though it is listed second."""
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Flutter Mane", hp=131, max_hp=131), mon("Incineroar", hp=202, max_hp=202)],
    )
    turn = battle.choose_turn(
        template([move("Close Combat", [1, 2])], [move("Aqua Jet", [1, 2])]),
        board,
    )
    assert turn["slot_0"] == {"type": "move", "move_id": "Close Combat", "target": 2}


def test_a_move_with_no_legal_opposing_target_gets_no_target_field():
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Incineroar", hp=202, max_hp=202)],
    )
    turn = battle.choose_turn(
        template([move("Surging Strikes", [0])], [move("Aqua Jet", [1])]), board
    )
    assert "target" not in turn["slot_0"]


def test_fake_out_is_worth_more_on_the_first_turn():
    """Fake Out only flinches on the turn the user comes out, and the turn
    number is the only signal available for that."""
    options = [move("Fake Out", [1, 2]), move("Knock Off", [1, 2])]
    first = battle.choose_turn(
        template(options, [move("Protect", [0])]),
        obs(
            ours=[mon("Incineroar", hp=202, max_hp=202)],
            theirs=[mon("Garchomp", hp=183, max_hp=183)],
            turn=1,
        ),
    )
    later = battle.choose_turn(
        template(options, [move("Protect", [0])]),
        obs(
            ours=[mon("Incineroar", hp=202, max_hp=202)],
            theirs=[mon("Garchomp", hp=183, max_hp=183)],
            turn=5,
        ),
    )
    assert first["slot_0"]["move_id"] == "Fake Out"
    # On a later turn the stronger attack wins instead.
    assert later["slot_0"]["move_id"] != "Fake Out"


def test_protect_is_preferred_when_the_slot_is_about_to_faint():
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=30, max_hp=176)],  # 17% left
        theirs=[mon("Flutter Mane", hp=131, max_hp=131)],
    )
    turn = battle.choose_turn(
        template(
            [move("Aqua Jet", [1, 2]), move("Protect", [0])],  # Water into Fairy is weak
            [move("Protect", [0])],
        ),
        board,
    )
    assert turn["slot_0"]["move_id"] == "Protect"


def test_a_damaging_move_still_wins_at_full_health():
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Incineroar", hp=202, max_hp=202)],
    )
    turn = battle.choose_turn(
        template([move("Close Combat", [1, 2]), move("Protect", [0])], [move("Protect", [0])]),
        board,
    )
    assert turn["slot_0"]["move_id"] == "Close Combat"


# -- structural rules -------------------------------------------------------


def test_two_slots_never_switch_into_the_same_pokemon():
    """The documented case: both slots are offered one reserve. One switches,
    the other passes."""
    board = obs(
        ours=[
            mon("Urshifu-Rapid-Strike", hp=176, max_hp=176),
            mon("Incineroar", hp=202, max_hp=202),
        ],
        theirs=[mon("Garchomp", hp=183, max_hp=183)],
    )
    turn = battle.choose_turn(template([SWITCH, PASS], [SWITCH, PASS]), board)
    picks = [turn["slot_0"], turn["slot_1"]]
    switches = [p for p in picks if p.get("type") == "switch"]
    assert len(switches) <= 1, picks


def test_both_slots_never_pass():
    """A pass is only legal when that slot genuinely has nothing to do."""
    board = obs(
        ours=[
            mon("Urshifu-Rapid-Strike", hp=176, max_hp=176),
            mon("Incineroar", hp=202, max_hp=202),
        ],
        theirs=[mon("Garchomp", hp=183, max_hp=183)],
    )
    turn = battle.choose_turn(
        template([PASS, move("Close Combat", [1])], [PASS, move("Knock Off", [1])]),
        board,
    )
    assert not (turn["slot_0"].get("type") == "pass" and turn["slot_1"].get("type") == "pass")
    assert turn["slot_0"]["type"] == "move" and turn["slot_1"]["type"] == "move"


# -- declining rather than guessing -----------------------------------------


def test_it_declines_when_the_board_is_unreadable():
    """Without a board every score is zero and a move loses its required
    target, which the server refuses — the caller's baseline is better."""
    assert battle.choose_turn(template([move("Close Combat", [1])]), {}) is None
    assert battle.choose_turn(template([move("Close Combat", [1])]), {"active_pokemon": []}) is None


def test_it_declines_when_the_opponent_side_is_empty():
    board = obs(ours=[mon("Urshifu-Rapid-Strike")], theirs=[])
    assert battle.choose_turn(template([move("Close Combat", [1])]), board) is None


def test_it_declines_on_an_unreadable_template():
    board = obs(ours=[mon("Urshifu-Rapid-Strike")], theirs=[mon("Garchomp")])
    for bad in (None, {}, {"slots": []}, {"slots": "nope"}, []):
        assert battle.choose_turn(bad, board) is None


def test_it_is_total_on_junk_slots():
    board = obs(ours=[mon("Urshifu-Rapid-Strike")], theirs=[mon("Garchomp")])
    turn = battle.choose_turn({"slots": [None, "junk", {"slot": 9}]}, board)
    assert turn is None


# -- determinism ------------------------------------------------------------


def test_the_same_board_always_yields_the_same_turn():
    board = obs(
        ours=[
            mon("Urshifu-Rapid-Strike", hp=176, max_hp=176),
            mon("Incineroar", hp=202, max_hp=202),
        ],
        theirs=[mon("Garchomp", hp=183, max_hp=183), mon("Flutter Mane", hp=131, max_hp=131)],
    )
    tmpl = template(
        [move("Close Combat", [1, 2]), move("Surging Strikes", [1, 2])],
        [move("Knock Off", [1, 2]), move("Flare Blitz", [1, 2])],
    )
    turns = {str(battle.choose_turn(tmpl, board)) for _ in range(5)}
    assert len(turns) == 1


# -- helpers ----------------------------------------------------------------


def test_spread_moves_are_recognised():
    assert battle.is_spread("Earthquake") is True  # allAdjacent
    assert battle.is_spread("Rock Slide") is True  # allAdjacentFoes
    assert battle.is_spread("Icy Wind") is True
    assert battle.is_spread("Close Combat") is False
    assert battle.is_spread(None) is False


def test_only_some_spread_moves_hit_our_own_partner():
    """Earthquake hits everything adjacent, including our ally; Rock Slide
    only hits foes. The two need different treatment."""
    assert battle.hits_own_ally("Earthquake") is True
    assert battle.hits_own_ally("Surf") is True
    assert battle.hits_own_ally("Rock Slide") is False
    assert battle.hits_own_ally("Close Combat") is False


def test_damaging_our_own_partner_counts_against_the_move():
    """A Ground move is scored lower when the partner would eat it than when
    the partner is immune to it."""
    theirs = [mon("Gholdengo", hp=100, max_hp=100), mon("Incineroar", hp=100, max_hp=100)]

    def score_with(partner_species):
        ours, their_actives = battle.board(
            obs(ours=[mon("Garchomp", hp=183, max_hp=183), mon(partner_species)], theirs=theirs)
        )
        score, _ = battle.move_score(
            "Earthquake",
            attacker=ours[0],
            defenders=their_actives,
            targets=[1, 2],
            turn=2,
            partner=ours[1],
        )
        return score

    immune_partner = score_with("Corviknight")  # Steel/Flying: Ground is 0x
    weak_partner = score_with("Incineroar")  # Fire/Dark: Ground is 2x
    assert weak_partner < immune_partner


def test_target_indices_map_onto_opposing_slots():
    assert battle.target_index(1) == 0
    assert battle.target_index(2) == 1
    assert battle.target_index(0) is None  # self / field / spread
    assert battle.target_index(-1) is None  # our own side
    assert battle.target_index(None) is None
    assert battle.target_index(True) is None


def test_a_pool_card_gives_the_real_set():
    """When the captured pool has our Pokémon, its real item and nature are
    used instead of the assumed spread."""
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Incineroar", hp=202, max_hp=202)],
    )
    ours, theirs = battle.board(board, pool_cards=[URSHIFU, INCINEROAR])
    assert ours[0].card.item == "Mystic Water"
    assert ours[0].card.stats["atk"] == 200
    assert theirs[0].card.item == "Sitrus Berry"


def test_the_assumed_spread_uses_the_observed_hp():
    """EVs are unpublished, so HP is taken from the board and the rest assumed."""
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176)],
        theirs=[mon("Incineroar", hp=202, max_hp=202)],
    )
    ours, theirs = battle.board(board)
    assert ours[0].card.max_hp == 176
    assert theirs[0].card.max_hp == 202
    assert theirs[0].card.item is None


def test_board_slots_stay_aligned_when_a_slot_is_empty():
    board = obs(
        ours=[mon("Urshifu-Rapid-Strike", hp=176, max_hp=176), None],
        theirs=[None, mon("Garchomp", hp=183, max_hp=183)],
    )
    ours, theirs = battle.board(board)
    assert ours[0].card is not None
    assert ours[1].card is None
    assert theirs[1].card is not None


def test_fainted_pokemon_are_not_usable():
    active = battle.Active(card=None, summary={"species": "X", "fainted": True})
    assert active.usable is False
    assert active.fainted is True
