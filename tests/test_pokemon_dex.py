"""Tests for the vendored-data readers.

The type chart gets the most attention here, because getting its indexing
direction wrong produces numbers that look entirely plausible — Electric into
Water reads as neutral instead of doubled — and every downstream damage and
coverage judgement would quietly be wrong.
"""

from __future__ import annotations

from pokemon import dex

# -- id folding -------------------------------------------------------------


def test_ids_fold_case_and_punctuation():
    """Cards are published as display names in the draft but as ids elsewhere."""
    assert dex.to_id("Urshifu-Rapid-Strike") == "urshifurapidstrike"
    assert dex.to_id("Landorus-Therian") == "landorustherian"
    assert dex.to_id("Mr. Mime") == "mrmime"
    assert dex.to_id(None) == ""
    assert dex.to_id(7) == ""


def test_a_display_name_and_an_id_resolve_to_the_same_species():
    by_name = dex.species("Urshifu-Rapid-Strike")
    by_id = dex.species("urshifurapidstrike")
    assert by_name is not None and by_name is by_id
    assert dex.types_of("Urshifu-Rapid-Strike") == ("Fighting", "Water")


def test_base_stats_and_totals():
    assert dex.base_stat_total("Flutter Mane") == 570
    assert dex.base_stats("Incineroar")["atk"] == 115
    assert dex.abilities_of("Incineroar") == ("Blaze", "Intimidate")


def test_unknown_species_is_empty_not_an_error():
    for name in ("Notarealmon", None, 7, ""):
        assert dex.species(name) is None
        assert dex.types_of(name) == ()
        assert dex.base_stats(name) == {}
        assert dex.base_stat_total(name) == 0


# -- type chart -------------------------------------------------------------


def test_type_chart_matches_hand_checked_values():
    """Every one of these is checkable from the games directly."""
    cases = [
        # (attacking type, defending types, expected multiplier)
        ("Electric", ("Water",), 2.0),
        ("Water", ("Fire",), 2.0),
        ("Fire", ("Water",), 0.5),
        ("Grass", ("Water",), 2.0),
        ("Fighting", ("Normal",), 2.0),
        ("Ice", ("Dragon", "Flying"), 4.0),  # doubly weak
        ("Rock", ("Fire", "Flying"), 4.0),
        ("Bug", ("Grass", "Psychic"), 4.0),  # 2x on both
        ("Fairy", ("Dragon",), 2.0),
        ("Steel", ("Fairy",), 2.0),
        ("Bug", ("Fire",), 0.5),
        ("Ghost", ("Normal",), 0.0),  # immunity
        ("Normal", ("Ghost",), 0.0),
        ("Ground", ("Fire", "Flying"), 0.0),  # Flying is immune to Ground
        ("Electric", ("Ground",), 0.0),
        ("Fighting", ("Ghost",), 0.0),
        ("Psychic", ("Dark",), 0.0),
        ("Poison", ("Steel",), 0.0),
        ("Dragon", ("Fairy",), 0.0),
    ]
    for attacking, defending, expected in cases:
        got = dex.effectiveness(attacking, defending)
        assert got == expected, f"{attacking} -> {defending}: got {got}, want {expected}"


def test_the_indexing_direction_is_the_way_round_it_looks_wrong():
    """Showdown stores typechart[defender]['damageTaken'][attacker]. Asserting
    an asymmetric pair pins the direction down: if the lookup were reversed
    these two would swap."""
    assert dex.effectiveness("Electric", ("Water",)) == 2.0
    assert dex.effectiveness("Water", ("Electric",)) == 1.0
    assert dex.effectiveness("Ground", ("Flying",)) == 0.0
    assert dex.effectiveness("Flying", ("Ground",)) == 1.0


def test_a_single_type_accepts_a_bare_string():
    assert dex.effectiveness("Water", "Fire") == 2.0


def test_effectiveness_is_total():
    assert dex.effectiveness(None, ("Water",)) == 1.0  # nothing to hit with
    assert dex.effectiveness("Water", None) == 1.0  # nothing to hit
    assert dex.effectiveness("Water", ("Notatype",)) == 1.0
    assert dex.effectiveness("Water", ("Water", "Flying")) == 0.5  # multiplied through
    assert dex.effectiveness("Notatype", ("Water",)) == 1.0


# -- moves ------------------------------------------------------------------


def test_move_lookup_and_fields():
    assert dex.move_type("Surging Strikes") == "Water"
    assert dex.move_category("Surging Strikes") == "Physical"
    assert dex.move_power("Surging Strikes") == 25
    assert dex.move_priority("Fake Out") == 3
    assert dex.move_priority("Protect") == 4
    assert dex.move_power("Protect") == 0
    assert dex.move_category("Protect") == "Status"


def test_is_damaging_excludes_status_and_zero_power_moves():
    assert dex.is_damaging("Close Combat") is True
    assert dex.is_damaging("Protect") is False
    assert dex.is_damaging("Notamove") is False


def test_unknown_move_is_total():
    for name in ("Notamove", None, 7):
        assert dex.move(name) is None
        assert dex.move_type(name) is None
        assert dex.move_power(name) == 0
        assert dex.move_priority(name) == 0
        assert dex.is_damaging(name) is False


# -- natures ----------------------------------------------------------------


def test_nature_multipliers():
    assert dex.nature_multiplier("Adamant", "atk") == 1.1
    assert dex.nature_multiplier("Adamant", "spa") == 0.9
    assert dex.nature_multiplier("Adamant", "spe") == 1.0
    assert dex.nature_multiplier("Timid", "spe") == 1.1
    assert dex.nature_multiplier("Notanature", "atk") == 1.0
    assert dex.nature_multiplier(None, "atk") == 1.0
