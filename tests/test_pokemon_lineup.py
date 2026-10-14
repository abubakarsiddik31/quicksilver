"""Tests for the Team Preview policy.

Team Preview is the one decision made with full information about the opposing
team, so the tests check that it actually uses it: a Pokémon that cannot hurt
their six should not be brought, and a lead should be chosen for how it opens
rather than for where it sits in the roster.
"""

from __future__ import annotations

from pokemon import dex, lineup

# Our six, as the template lists them.
OURS = [
    "gyarados",
    "urshifurapidstrike",
    "barraskewda",
    "rillaboom",
    "amoonguss",
    "fluttermane",
]
# Their six: Fire-heavy, which Water punishes and Grass cannot touch.
THEIRS = ["charizard", "talonflame", "incineroar", "armarouge"]


def summary(species, moves=None):
    entry = {
        "species": species,
        "name": species,
        "types": list(dex.types_of(species)),
        "base_stats": dex.base_stats(species),
    }
    if moves:
        entry["moves"] = moves
    return entry


def template(roster=None):
    return {"type": "select_lineup", "roster": list(roster or OURS)}


def obs(*, ours=None, theirs=None, our_moves=None):
    our_moves = our_moves or {}
    return {
        "phase": "team_preview",
        "your_roster": [summary(s, our_moves.get(s)) for s in (ours or OURS)],
        "opponent_roster": [summary(s) for s in (theirs or THEIRS)],
    }


MOVE_SETS = {
    "gyarados": ["Waterfall", "Crunch", "Icy Wind", "Protect"],
    "urshifurapidstrike": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
    "barraskewda": ["Liquidation", "Close Combat", "Aqua Jet", "Protect"],
    "rillaboom": ["Grassy Glide", "Knock Off", "Fake Out", "Protect"],
    "amoonguss": ["Spore", "Sludge Bomb", "Pollen Puff", "Rage Powder"],
    "fluttermane": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
}


# -- matchups ---------------------------------------------------------------


def test_offensive_pressure_reads_type_effectiveness():
    """Water into a Fire type is 2x. Grass into Fire/Flying is 0.25x — doubly
    resisted, so it is not merely weak, it is nearly useless."""
    gyarados = lineup.card_for_summary(summary("gyarados", ["Waterfall"]), None)
    grass_only = lineup.card_for_summary(summary("rillaboom", ["Grassy Glide"]), None)
    charizard = lineup.card_for_summary(summary("charizard"), None)

    assert lineup.offensive_pressure(gyarados, charizard) == 2.0
    assert lineup.offensive_pressure(grass_only, charizard) == 0.25


def test_a_neutral_move_can_be_the_best_option():
    """Rillaboom's real set is not Grass-only: Dark and Normal are neutral into
    Fire/Flying, so the best it can manage is neutral."""
    rillaboom = lineup.card_for_summary(summary("rillaboom", MOVE_SETS["rillaboom"]), None)
    charizard = lineup.card_for_summary(summary("charizard"), None)
    assert lineup.offensive_pressure(rillaboom, charizard) == 1.0


def test_defensive_risk_uses_the_opponents_own_types():
    """Fire threatens a Grass type for 2x. Against Water/Flying it can only
    manage neutral, because Flying is not resisted."""
    rillaboom = lineup.card_for_summary(summary("rillaboom", MOVE_SETS["rillaboom"]), None)
    gyarados = lineup.card_for_summary(summary("gyarados", MOVE_SETS["gyarados"]), None)
    charizard = lineup.card_for_summary(summary("charizard"), None)

    assert lineup.defensive_risk(rillaboom, charizard) == 2.0
    assert lineup.defensive_risk(gyarados, charizard) == 1.0
    assert lineup.defensive_risk(rillaboom, charizard) > lineup.defensive_risk(gyarados, charizard)


def test_matchup_averages_over_the_whole_six():
    gyarados = lineup.card_for_summary(summary("gyarados", MOVE_SETS["gyarados"]), None)
    theirs = [lineup.card_for_summary(summary(s), None) for s in THEIRS]
    score = lineup.matchup(gyarados, theirs)
    assert isinstance(score, float)
    # A Water type against a Fire-heavy team should be clearly positive.
    assert score > 0


def test_matchup_is_total_on_junk():
    assert lineup.matchup(None, []) == 0.0
    assert lineup.matchup(None, [None]) == 0.0
    assert isinstance(lineup.offensive_pressure(None, None), float)
    assert isinstance(lineup.defensive_risk(None, None), float)


# -- leads ------------------------------------------------------------------


def test_a_fake_out_user_makes_a_better_lead():
    """Fake Out on a lead is close to a free turn."""
    with_fake_out = lineup.card_for_summary(summary("rillaboom", MOVE_SETS["rillaboom"]), None)
    without = lineup.card_for_summary(summary("rillaboom", ["Grassy Glide", "Knock Off"]), None)
    theirs = [lineup.card_for_summary(summary(s), None) for s in THEIRS]
    assert lineup.lead_score(with_fake_out, theirs) > lineup.lead_score(without, theirs)


def test_speed_makes_a_better_lead():
    fast = lineup.card_for_summary(
        {"species": "barraskewda", "moves": ["Liquidation", "Protect"]}, None
    )
    slow = lineup.card_for_summary(
        {"species": "amoonguss", "moves": ["Liquidation", "Protect"]}, None
    )
    assert fast.stats["spe"] > slow.stats["spe"]
    assert lineup.lead_score(fast, []) > lineup.lead_score(slow, [])


# -- the choice -------------------------------------------------------------


def test_it_brings_four_and_leads_two_from_the_template_roster():
    chosen = lineup.choose_lineup(template(), obs(our_moves=MOVE_SETS))
    assert chosen is not None
    assert chosen["type"] == "select_lineup"
    assert len(chosen["bring"]) == lineup.BRING == 4
    assert len(chosen["leads"]) == lineup.LEADS == 2
    # The server validates against its own vocabulary, so ids must come from it.
    assert set(chosen["bring"]) <= set(OURS)
    assert set(chosen["leads"]) <= set(chosen["bring"])
    assert len(set(chosen["bring"])) == 4
    assert len(set(chosen["leads"])) == 2


def test_it_leaves_out_the_pokemon_that_cannot_hurt_them():
    """Against a Fire-heavy team, Grass is dead weight and Water is not."""
    chosen = lineup.choose_lineup(template(), obs(our_moves=MOVE_SETS))
    assert "rillaboom" not in chosen["bring"]
    assert "amoonguss" not in chosen["bring"]
    assert "gyarados" in chosen["bring"]


def test_the_brought_four_track_the_opposing_team():
    """The same six, against a team Water cannot touch, must bring differently."""
    water_proof = ["gyarados", "vaporeon", "milotic", "lapras"]
    chosen = lineup.choose_lineup(
        template(), obs(ours=OURS, theirs=water_proof, our_moves=MOVE_SETS)
    )
    # Grass and Electric answers should now survive the cut; Water walls should
    # not monopolise it.
    assert len(chosen["bring"]) == 4
    assert not set(chosen["bring"]) == {
        "gyarados",
        "urshifurapidstrike",
        "barraskewda",
        "fluttermane",
    }


def test_the_choice_is_deterministic():
    picks = {
        tuple(lineup.choose_lineup(template(), obs(our_moves=MOVE_SETS))["bring"]) for _ in range(5)
    }
    assert len(picks) == 1


def test_it_uses_the_pool_when_one_is_available():
    """The pool carries the real moves and EVs, which sharpens the matchup."""
    pool_cards = [
        {
            "card_id": "a",
            "species": "Gyarados",
            "item": "Mystic Water",
            "ability": "Intimidate",
            "nature": "Adamant",
            "level": 50,
            "evs": {"atk": 252},
            "ivs": {},
            "moves": ["Waterfall", "Crunch", "Icy Wind", "Protect"],
        },
    ]
    chosen = lineup.choose_lineup(template(), obs(our_moves={}), pool_cards=pool_cards)
    assert chosen is not None
    assert "gyarados" in chosen["bring"]


# -- declining --------------------------------------------------------------


def test_it_declines_rather_than_guessing():
    assert lineup.choose_lineup(None, obs()) is None
    assert lineup.choose_lineup({}, obs()) is None
    assert lineup.choose_lineup({"roster": []}, obs()) is None
    # Fewer than four readable roster entries.
    assert lineup.choose_lineup(template(["a", "b"]), obs()) is None


def test_unrecognised_species_still_yield_a_legal_lineup():
    """An unknown species cannot be scored, but the ids come from the server's
    own roster, so the action is still legal — which beats declining."""
    chosen = lineup.choose_lineup(template(["x", "y", "z", "w", "v", "u"]), obs())
    assert chosen is not None
    assert set(chosen["bring"]) <= {"x", "y", "z", "w", "v", "u"}
    assert len(chosen["bring"]) == 4
    assert set(chosen["leads"]) <= set(chosen["bring"])


def test_summaries_are_keyed_by_normalized_species():
    mapped = lineup._summaries([summary("Landorus-Therian")])
    assert dex.to_id("Landorus-Therian") in mapped
    assert lineup._summaries([None, "junk", {}]) == {}
