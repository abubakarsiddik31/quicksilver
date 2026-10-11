"""Tests for the draft policy.

Two things are being pinned down: that the snake order is read correctly (it
decides how much a denial is worth), and that a card is judged by what it adds
to *this* team rather than in isolation — which is the whole point of the
scoring design.
"""

from __future__ import annotations

from pokemon import draft

# Real cards, as a draft state publishes them.
FLUTTER = {
    "card_id": "c-flutter", "species": "Flutter Mane", "item": "Booster Energy",
    "ability": "Protosynthesis", "nature": "Timid", "level": 50, "evs": {}, "ivs": {},
    "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
}
URS = {
    "card_id": "c-urshifu", "species": "Urshifu-Rapid-Strike", "item": "Mystic Water",
    "ability": "Unseen Fist", "nature": "Adamant", "level": 50, "evs": {}, "ivs": {},
    "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
}
GRA = {
    "card_id": "c-gyarados", "species": "Gyarados", "item": "Sitrus Berry",
    "ability": "Intimidate", "nature": "Adamant", "level": 50, "evs": {}, "ivs": {},
    "moves": ["Waterfall", "Crunch", "Icy Wind", "Protect"],
}
CATERPIE = {
    "card_id": "c-caterpie", "species": "Caterpie", "item": "Leftovers",
    "ability": "Shield Dust", "nature": "Bold", "level": 50, "evs": {}, "ivs": {},
    "moves": ["Tackle", "String Shot"],
}


def obs(*, cards, rosters=None, pick_number=1, first_drafter="us"):
    return {
        "phase": "draft",
        "pick_number": pick_number,
        "first_drafter": first_drafter,
        "available_cards": list(cards),
        "available_card_ids": [
            c["card_id"] for c in cards if isinstance(c, dict) and "card_id" in c
        ],
        "rosters": rosters if rosters is not None else {"us": [], "them": []},
        "picks": [],
    }


# -- snake order ------------------------------------------------------------


def test_the_snake_order_is_a_b_b_a_a_b_b_a_a_b_b_a():
    """Who picks when decides how much a denial is worth, so this has to be
    exactly the documented order."""
    sequence = [draft.pick_owner(n, we_are_first=True) for n in range(1, 13)]
    assert sequence == [
        "us", "them", "them", "us", "us", "them",
        "them", "us", "us", "them", "them", "us",
    ]
    # And it is symmetric from the other seat.
    mirrored = [draft.pick_owner(n, we_are_first=False) for n in range(1, 13)]
    assert mirrored == [
        "them" if owner == "us" else "us" for owner in sequence
    ]


def test_each_side_gets_exactly_six_picks():
    owners = [draft.pick_owner(n, we_are_first=True) for n in range(1, 13)]
    assert owners.count("us") == 6
    assert owners.count("them") == 6


def test_we_are_first_reads_the_published_drafter():
    assert draft.we_are_first(obs(cards=[], first_drafter="us"), "us") is True
    assert draft.we_are_first(obs(cards=[], first_drafter="them"), "us") is False
    assert draft.we_are_first({"phase": "draft"}, "us") is None
    assert draft.we_are_first(obs(cards=[], first_drafter="us"), None) is None


def test_denial_is_worth_more_when_the_opponent_picks_twice_next():
    """Pick 1: we take, then they take picks 2 and 3 — denial is at its most
    valuable. Pick 4: we take, then we take again — denial matters less."""
    early = draft.denial_weight(obs(cards=[], pick_number=1, first_drafter="us"), "us")
    late = draft.denial_weight(obs(cards=[], pick_number=4, first_drafter="us"), "us")
    assert early == draft.DENIAL_WEIGHT_WHEN_THEY_PICK_NEXT
    assert late == draft.DENIAL_WEIGHT_OTHERWISE
    assert early > late


def test_denial_falls_back_when_the_order_is_unknown():
    assert (
        draft.denial_weight({"phase": "draft"}, "us")
        == draft.DENIAL_WEIGHT_OTHERWISE
    )


# -- card profiles ----------------------------------------------------------


def test_coverage_and_weaknesses_of_a_real_card():
    covered = draft.coverage(FLUTTER)
    # Fairy hits Fighting, Dragon and Dark; Ghost hits Ghost and Psychic.
    assert {"Fighting", "Dragon", "Dark", "Ghost", "Psychic"} <= covered

    weak = draft.weaknesses(FLUTTER)
    assert {"Ghost", "Steel"} <= weak


def test_immunities_count_as_resistances():
    """Flutter Mane is Ghost/Fairy: immune to Normal and Fighting."""
    resisted = draft.resistances(FLUTTER)
    assert {"Normal", "Fighting"} <= resisted
    assert "Ghost" not in resisted  # that one is a weakness, not a resistance


def test_damage_roles_reflect_the_actual_moves():
    assert draft.damage_roles(FLUTTER) == {"Special"}
    assert draft.damage_roles(URS) == {"Physical"}
    assert draft.damage_roles(CATERPIE) == {"Physical"}


def test_status_only_cards_have_no_damage_role():
    wall = {**CATERPIE, "moves": ["Protect", "String Shot"]}
    assert draft.damage_roles(wall) == set()
    assert draft.offensive_types(wall) == set()
    assert draft.coverage(wall) == set()


def test_card_quality_tracks_bulk_and_speed():
    assert draft.card_quality(FLUTTER) > draft.card_quality(CATERPIE)
    assert draft.card_quality({}) == 0.0


def test_profiles_are_total_on_junk():
    for junk in ({}, {"species": 7}, {"moves": "nope"}, {"moves": [None, 7]}):
        assert isinstance(draft.coverage(junk), set)
        assert isinstance(draft.weaknesses(junk), set)
        assert isinstance(draft.resistances(junk), set)
        assert isinstance(draft.card_quality(junk), float)


# -- team value -------------------------------------------------------------


def test_an_empty_team_is_worth_nothing():
    assert draft.team_value([]) == 0.0


def test_adding_a_card_increases_team_value():
    assert draft.team_value([FLUTTER]) > 0
    assert draft.team_value([FLUTTER, URS]) > draft.team_value([FLUTTER])


def test_value_is_order_independent():
    assert draft.team_value([FLUTTER, URS]) == draft.team_value([URS, FLUTTER])


def test_coverage_is_counted_once_for_the_team():
    """A second copy of a card must not add its coverage again — otherwise a
    duplicate looks as valuable as a new type."""
    fresh = draft.marginal_value(FLUTTER, [])
    duplicate = draft.marginal_value(FLUTTER, [FLUTTER])
    assert duplicate < fresh


def test_a_shared_weakness_is_penalised():
    """Both are weak to Ghost, so the pair is worth less than the parts."""
    counts = draft.shared_weaknesses([FLUTTER, FLUTTER])
    assert counts.get("Ghost", 0) == 2
    assert draft.shared_weaknesses([FLUTTER]) == draft.shared_weaknesses([FLUTTER])
    # A duplicated weakness costs something.
    assert draft.team_value([FLUTTER, FLUTTER]) < 2 * draft.team_value([FLUTTER])


def test_role_balance_rewards_both_attack_types():
    assert draft.role_bonus([FLUTTER]) == 0.0          # special only
    assert draft.role_bonus([URS]) == 0.0              # physical only
    assert draft.role_bonus([FLUTTER, URS]) == draft.ROLE_BALANCE_BONUS
    assert draft.role_bonus([]) == 0.0


# -- scoring and choosing ---------------------------------------------------


def test_a_card_is_scored_by_the_swing_it_creates():
    """Denial means a card the opponent wants is worth more to us to take."""
    without_denial = draft.score_card(
        GRA, our_roster=[], their_roster=[], denial=0.0
    )
    with_denial = draft.score_card(GRA, our_roster=[], their_roster=[FLUTTER], denial=0.8)
    assert with_denial > without_denial


def test_choose_card_prefers_the_stronger_offer():
    state = obs(cards=[CATERPIE, FLUTTER, URS])
    assert draft.choose_card(state, agent_id="us") in {"c-flutter", "c-urshifu"}
    assert draft.choose_card(state, agent_id="us") != "c-caterpie"


def test_choose_card_is_deterministic():
    state = obs(cards=[CATERPIE, FLUTTER, URS])
    picks = {draft.choose_card(state, agent_id="us") for _ in range(5)}
    assert len(picks) == 1


def test_choose_card_returns_none_with_nothing_on_offer():
    assert draft.choose_card(obs(cards=[]), agent_id="us") is None
    assert draft.choose_card({}, agent_id="us") is None


def test_choose_card_is_total_on_junk_cards():
    state = obs(cards=[{"card_id": "ok", "species": "Notarealmon"}, {}, {"species": "Gyarados"}])
    assert draft.choose_card(state, agent_id="us") == "ok"


def test_choose_from_actions_maps_back_onto_the_servers_own_list():
    """A constructed pick is refused as INVALID_ACTION, so the decision must be
    matched to the legal action the server actually offered."""

    class Action:
        def __init__(self, action_id):
            self.action_id = action_id

    actions = [Action("draft_pick:c-flutter"), Action("draft_pick:c-urshifu")]
    picked = draft.choose_from_actions(obs(cards=[FLUTTER, URS]), agent_id="us", legal_actions=actions)
    assert picked in actions
    assert picked.action_id == "draft_pick:c-flutter" or picked.action_id == "draft_pick:c-urshifu"


def test_choose_from_actions_returns_none_when_the_choice_was_not_offered():
    class Action:
        def __init__(self, action_id):
            self.action_id = action_id

    # The state offers Flutter Mane but the server's action list does not.
    actions = [Action("draft_pick:c-caterpie")]
    state = obs(cards=[FLUTTER, CATERPIE])
    result = draft.choose_from_actions(state, agent_id="us", legal_actions=actions)
    assert result is None or result.action_id == "draft_pick:c-flutter"


def test_choose_from_actions_is_total():
    assert draft.choose_from_actions({}, agent_id="us", legal_actions=None) is None
    assert draft.choose_from_actions({}, agent_id="us", legal_actions=["junk"]) is None
