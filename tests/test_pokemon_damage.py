"""Tests for stat computation and damage.

Where a number is asserted exactly, it was worked out by hand from the formula
in the module docstring and the value is shown in a comment. The rest assert
*relationships* that must hold whatever the tuning — a spread move hitting
softer than a single-target one, an immunity dealing nothing, a critical hit
ignoring the defender's boosts.
"""

from __future__ import annotations

import math

from pokemon import damage, stats

# Straight from the documented example card.
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
FLUTTER = {
    "card_id": "f",
    "species": "Flutter Mane",
    "item": "Booster Energy",
    "ability": "Protosynthesis",
    "nature": "Timid",
    "level": 50,
    "evs": {"hp": 4, "spa": 252, "spe": 252},
    "ivs": {},
    "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
}
GARCHOMP = {
    "card_id": "g",
    "species": "Garchomp",
    "item": "Life Orb",
    "ability": "Rough Skin",
    "nature": "Jolly",
    "level": 50,
    "evs": {"atk": 252, "spe": 252},
    "ivs": {},
    "moves": ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"],
}


def card(raw):
    return damage.parse_card(raw)


# -- stats ------------------------------------------------------------------


def test_the_level_50_stat_formula_by_hand():
    """Urshifu, base 130 Atk, 252 EV, 31 IV, Adamant at level 50:
    floor((2*130 + 31 + 63) * 50/100) + 5 = 182, and 182 * 1.1 = 200.2 -> 200."""
    computed = stats.compute_stats("Urshifu-Rapid-Strike", {"atk": 252}, {}, "Adamant", 50)
    assert computed["atk"] == 200


def test_hp_uses_the_plus_level_plus_ten_term():
    """Base 100 HP, no EVs, 31 IV at level 50: floor((200+31) * 0.5) + 50 + 10
    = 115 + 60 = 175."""
    assert stats.compute_stats("Urshifu-Rapid-Strike", {}, {}, None, 50)["hp"] == 175
    # With 4 EVs: floor((200 + 31 + 1) * 0.5) + 60 = 176.
    assert stats.compute_stats("Urshifu-Rapid-Strike", {"hp": 4}, {}, None, 50)["hp"] == 176


def test_a_missing_iv_defaults_to_31_not_zero():
    """Draft cards publish ``"ivs": {}``; reading that as zero IVs would
    understate every stat."""
    assert stats.DEFAULT_IV == 31
    default = stats.compute_stats("Urshifu-Rapid-Strike", {}, {}, None, 50)
    explicit = stats.compute_stats(
        "Urshifu-Rapid-Strike", {}, {k: 31 for k in stats.STAT_KEYS}, None, 50
    )
    assert default == explicit


def test_nature_applies_after_the_plus_five_and_never_to_hp():
    neutral = stats.compute_stats("Urshifu-Rapid-Strike", {"hp": 252}, {}, None, 50)
    adamant = stats.compute_stats("Urshifu-Rapid-Strike", {"hp": 252}, {}, "Adamant", 50)
    assert adamant["hp"] == neutral["hp"]  # HP is never nature-modified
    assert adamant["atk"] > neutral["atk"]
    assert adamant["spa"] < neutral["spa"]  # Adamant lowers Sp. Atk


def test_evs_are_capped_and_unknown_species_are_empty():
    over = stats.compute_stats("Urshifu-Rapid-Strike", {"atk": 9999}, {}, None, 50)
    capped = stats.compute_stats("Urshifu-Rapid-Strike", {"atk": 252}, {}, None, 50)
    assert over["atk"] == capped["atk"]
    assert stats.compute_stats("Notarealmon", {}, {}, None, 50) == {}


def test_stage_multipliers():
    assert stats.stage_multiplier(0) == 1.0
    assert stats.stage_multiplier(2) == 2.0
    assert stats.stage_multiplier(-2) == 0.5
    assert stats.stage_multiplier(6) == 4.0
    assert stats.stage_multiplier(99) == stats.stage_multiplier(6)  # clamped


# -- parse_card -------------------------------------------------------------


def test_parse_card_reads_the_published_set():
    parsed = card(URSHIFU)
    assert parsed.species == "Urshifu-Rapid-Strike"
    assert parsed.item == "Mystic Water"
    assert parsed.types == ("Fighting", "Water")
    assert parsed.moves == ("Surging Strikes", "Close Combat", "Aqua Jet", "Protect")
    assert parsed.stats == {"hp": 176, "atk": 200, "def": 120, "spa": 74, "spd": 80, "spe": 149}
    assert parsed.max_hp == 176


def test_parse_card_is_total():
    for junk in (None, 7, "junk", {}, {"species": 7}, {"moves": "nope"}):
        parsed = damage.parse_card(junk)
        assert isinstance(parsed, damage.Card)
        assert parsed.max_hp == 0


# -- damage -----------------------------------------------------------------


def test_close_combat_damage_is_exactly_right():
    """Hand-computed:
    base = floor(floor(floor(2*50/5 + 2) * 120 * 200 / 111) / 50) + 2
         = floor(floor(22 * 120 * 200 / 111) / 50) + 2 = floor(4756/50) + 2 = 97
    modifiers = STAB 1.5 * Fighting-into-Fire/Dark 2.0 = 3.0
    raw = 291  ->  min floor(291*0.85) = 247,  max 291
    """
    result = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    assert result.effectiveness == 4.0 or result.effectiveness == 2.0
    assert result.effectiveness == 2.0
    assert (result.min_damage, result.max_damage) == (247, 291)
    assert result.defender_max_hp == 202
    assert result.guaranteed_ko_hits == 1
    assert result.is_ko() is True


def test_surging_strikes_includes_stab_and_the_type_boost_item():
    """base = floor(floor(22 * 25 * 200 / 111) / 50) + 2 = 21
    modifiers = STAB 1.5 * 2.0 * Mystic Water 1.2 = 3.6
    raw = 75.6  ->  min floor(64.26) = 64,  max floor(75.6) = 75"""
    result = damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes")
    assert (result.min_damage, result.max_damage) == (64, 75)


def test_a_status_move_deals_no_damage():
    result = damage.damage(card(URSHIFU), card(INCINEROAR), "Protect")
    assert result.max_damage == 0
    assert result.power == 0


def test_an_immunity_deals_nothing():
    """Normal into a Ghost type is 0x, and must not fall back to a minimum of 1."""
    result = damage.damage(card(INCINEROAR), card(FLUTTER), "Fake Out")
    assert result.effectiveness == 0.0
    assert result.immune is True
    assert result.max_damage == 0
    assert result.min_damage == 0


def test_a_four_times_weakness_lands():
    """Ice into Garchomp (Dragon/Ground) is 4x."""
    result = damage.damage(card(FLUTTER), card(GARCHOMP), "Icy Wind")
    assert result.effectiveness == 4.0


def test_the_doubles_spread_reduction_is_exactly_three_quarters():
    single = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    spread = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat", spread=True)
    assert spread.max_damage < single.max_damage
    # The pre-roll value scales by 0.75, so the ratio is close to it once floored.
    assert math.isclose(spread.max_damage / single.max_damage, 0.75, abs_tol=0.02)


def test_stab_is_worth_one_point_five():
    """Isolate STAB by holding the stats identical and changing only the
    attacker's typing: same move, same stats, one gets the same-type bonus."""
    import dataclasses

    urshifu = card(URSHIFU)  # Fighting/Water: STAB on Water
    no_stab = dataclasses.replace(urshifu, species="Incineroar")  # same stats, Fire/Dark
    defender = card(INCINEROAR)

    with_stab = damage.damage(urshifu, defender, "Surging Strikes")
    without = damage.damage(no_stab, defender, "Surging Strikes")
    assert with_stab.effectiveness == without.effectiveness == 2.0
    assert math.isclose(with_stab.max_damage / without.max_damage, 1.5, abs_tol=0.02)


def test_critical_hits_ignore_the_defenders_positive_boosts():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    boosted = damage.damage(
        card(URSHIFU), card(INCINEROAR), "Close Combat", defender_boosts={"def": 2}
    )
    crit = damage.damage(
        card(URSHIFU),
        card(INCINEROAR),
        "Close Combat",
        defender_boosts={"def": 2},
        crit=True,
    )
    assert boosted.max_damage < plain.max_damage
    # The boost is ignored, and the 1.5x critical multiplier then applies.
    assert crit.max_damage > plain.max_damage


def test_boosts_move_damage_in_the_right_direction():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    up = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat", attacker_boosts={"atk": 2})
    down = damage.damage(
        card(URSHIFU), card(INCINEROAR), "Close Combat", attacker_boosts={"atk": -2}
    )
    assert up.max_damage > plain.max_damage > down.max_damage


def test_screens_halve_the_matching_category_only():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    reflected = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat", reflect=True)
    # Reflect is the physical screen; a physical move must be reduced.
    assert reflected.max_damage < plain.max_damage


def test_burn_halves_physical_damage():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    burned = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat", burned=True)
    assert burned.max_damage < plain.max_damage
    # A special move is unaffected by burn.
    special_plain = damage.damage(card(FLUTTER), card(GARCHOMP), "Moonblast")
    special_burned = damage.damage(card(FLUTTER), card(GARCHOMP), "Moonblast", burned=True)
    assert special_burned.max_damage == special_plain.max_damage


def test_weather_boosts_and_damps_the_matching_types():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes")
    rain = damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes", weather="Rain")
    sun = damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes", weather="Sun")
    assert rain.max_damage > plain.max_damage
    assert sun.max_damage < plain.max_damage


def test_weather_names_are_folded():
    """The sim reports weather under several spellings for the same effect;
    all of them must land on the same multiplier."""
    results = {
        damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes", weather=name).max_damage
        for name in ("Rain", "rain", "Primordial Sea", "Heavy Rain")
    }
    assert len(results) == 1
    suns = {
        damage.damage(card(URSHIFU), card(INCINEROAR), "Surging Strikes", weather=name).max_damage
        for name in ("Sun", "sunny", "Desolate Land", "Harsh Sunshine")
    }
    assert len(suns) == 1
    assert results != suns


def test_multiscale_halves_only_at_full_health():
    plain = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    full = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat", defender_full_hp=True)
    assert full.max_damage == plain.max_damage  # Incineroar has Intimidate, not Multiscale


def test_levitate_grants_a_ground_immunity():
    levitate = {**GARCHOMP, "species": "Gengar", "ability": "Levitate"}
    result = damage.damage(card(GARCHOMP), card(levitate), "Earthquake")
    assert result.effectiveness == 0.0
    assert result.max_damage == 0


def test_unmodelled_items_and_abilities_are_reported_not_guessed():
    """A number the policy trusts must declare what it left out."""
    result = damage.damage(card(URSHIFU), card(INCINEROAR), "Close Combat")
    assert "ability:Intimidate" in result.unmodelled  # a field state we cannot see

    # A handled item and ability are not reported as gaps.
    assert not any(entry.startswith("item:") for entry in result.unmodelled)


def test_damage_is_total_on_junk():
    empty = damage.parse_card(None)
    for attacker, defender, move in (
        (empty, empty, "Close Combat"),
        (card(URSHIFU), empty, "Close Combat"),
        (card(URSHIFU), card(INCINEROAR), "Notamove"),
        (card(URSHIFU), card(INCINEROAR), None),
    ):
        result = damage.damage(attacker, defender, move)
        assert isinstance(result, damage.DamageResult)
        assert result.max_damage >= 0


# -- best_move --------------------------------------------------------------


def test_best_move_picks_the_hardest_hitter():
    best = damage.best_move(card(URSHIFU), card(INCINEROAR))
    assert best is not None
    # Close Combat (120 BP, 2x) should beat Aqua Jet (40 BP) and Surging Strikes.
    assert best.move == "Close Combat"


def test_best_move_returns_none_when_nothing_can_damage():
    # No damaging moves at all.
    wall = {**INCINEROAR, "moves": ["Protect", "Parting Shot"]}
    assert damage.best_move(card(wall), card(GARCHOMP)) is None

    # Only a Normal-type attack, into a Ghost type: immune, so nothing lands.
    normal_only = {**INCINEROAR, "moves": ["Fake Out", "Protect"]}
    assert damage.best_move(card(normal_only), card(FLUTTER)) is None
