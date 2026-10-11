"""Tests for the Pokémon observation readers and the pool capture.

No network, no game server, no API key: every case is a recorded-shape dict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from pokemon import observation as obs
from pokemon import pool

CARD = {
    "card_id": "vgc-urshifu-rapid-strike",
    "species": "Urshifu-Rapid-Strike",
    "item": "Mystic Water",
    "ability": "Unseen Fist",
    "nature": "Adamant",
    "level": 50,
    "evs": {"hp": 4, "atk": 252, "spe": 252},
    "ivs": {},
    "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
}

DRAFT = {
    "phase": "draft",
    "draft_complete": False,
    "battle_format": "gen9vgc2025regi",
    "current_seat": "agent-a",
    "pick_number": 1,
    "picks_remaining": 12,
    "first_drafter": "agent-a",
    "available_card_ids": ["vgc-urshifu-rapid-strike", "vgc-gyarados"],
    "available_cards": [CARD, {**CARD, "card_id": "vgc-gyarados", "species": "Gyarados"}],
    "rosters": {"agent-a": [], "agent-b": []},
    "picks": [],
    "decision_timeout_seconds": 15,
}


def state(observation, *, phase="draft"):
    """A state stand-in: ``raw`` is authoritative, ``observation`` is the
    ``str()`` repr the SDK actually produces."""
    return SimpleNamespace(
        raw={"observation": observation},
        observation=str(observation),
        phase=phase,
    )


# -- observation_of ---------------------------------------------------------


def test_reads_the_authoritative_dict_from_raw():
    assert obs.observation_of(state(DRAFT))["phase"] == "draft"


def test_falls_back_to_the_str_repr_when_raw_is_missing():
    """``GameState.observation`` is a Python ``repr``, not JSON — the SDK
    coerces it with ``str()``. Reading it must still work."""
    s = SimpleNamespace(raw={}, observation=str(DRAFT), phase="draft")
    assert obs.observation_of(s)["pick_number"] == 1


def test_reads_a_json_string_too():
    s = SimpleNamespace(raw={}, observation=json.dumps(DRAFT), phase="draft")
    assert obs.observation_of(s)["phase"] == "draft"


def test_tolerates_garbage():
    for observation in ("", "not a dict", 7, None, "[1,2,3]"):
        s = SimpleNamespace(raw={"observation": observation}, observation=observation)
        assert obs.observation_of(s) == {}
    assert obs.observation_of(SimpleNamespace()) == {}


def test_phase_prefers_the_observation_then_the_state():
    assert obs.phase_of(state(DRAFT)) == "draft"
    assert obs.phase_of(state({}), ) == "draft"
    assert obs.phase_of(SimpleNamespace(raw={"observation": {}}, phase="moving")) == "moving"


# -- species folding --------------------------------------------------------


def test_species_names_fold_case_and_punctuation():
    """The server ignores case and punctuation when matching a lineup."""
    assert obs.normalize_species("Landorus-Therian") == obs.normalize_species("landorustherian")
    assert obs.normalize_species("Urshifu-Rapid-Strike") == "urshifurapidstrike"
    assert obs.normalize_species(None) == ""
    assert obs.normalize_species(7) == ""


# -- draft ------------------------------------------------------------------


def test_pool_carries_everything_a_damage_calc_needs():
    cards = obs.pool_of(DRAFT)
    assert len(cards) == 2
    first = cards[0]
    for field in ("card_id", "species", "item", "ability", "nature", "level", "evs", "ivs", "moves"):
        assert field in first, field
    assert first["moves"] == CARD["moves"]


def test_pool_falls_back_to_ids_when_only_ids_are_present():
    cards = obs.pool_of({"available_card_ids": ["a", "b"]})
    assert cards == [{"card_id": "a"}, {"card_id": "b"}]


def test_draft_scalar_readers():
    assert obs.current_seat(DRAFT) == "agent-a"
    assert obs.first_drafter(DRAFT) == "agent-a"
    assert obs.pick_number(DRAFT) == 1
    assert obs.picks_remaining(DRAFT) == 12
    assert obs.draft_complete(DRAFT) is False
    assert obs.available_card_ids(DRAFT) == ["vgc-urshifu-rapid-strike", "vgc-gyarados"]


def test_rosters_and_picks_are_total():
    assert obs.rosters(DRAFT) == {"agent-a": [], "agent-b": []}
    assert obs.picks(DRAFT) == []
    assert obs.rosters({"rosters": "nonsense"}) == {}
    assert obs.my_roster(DRAFT, "agent-a") == []
    assert obs.my_roster(DRAFT, None) == []


# -- team preview -----------------------------------------------------------


def test_team_preview_rosters():
    preview = {
        "phase": "team_preview",
        "your_roster": [{"species": "incineroar", "name": "Incineroar"}],
        "opponent_roster": [{"species": "urshifurapidstrike"}],
    }
    assert len(obs.your_roster(preview)) == 1
    assert obs.opponent_roster(preview)[0]["species"] == "urshifurapidstrike"


# -- board and slots --------------------------------------------------------


def test_doubles_slots_preserve_position_when_a_slot_is_missing():
    """A fainted/absent slot must not shift the other slot's index."""
    board = {"active_pokemon": [None, {"species": "gholdengo"}]}
    slots = obs.active_pokemon(board)
    assert slots[0] is None
    assert slots[1]["species"] == "gholdengo"


def test_singles_board_becomes_a_one_entry_list():
    board = {"active_pokemon": {"species": "garchomp"}, "is_doubles": False}
    assert obs.active_pokemon(board) == [{"species": "garchomp"}]
    assert obs.is_doubles(board) is False


def test_doubles_move_lists_stay_per_slot():
    board = {
        "is_doubles": True,
        "available_moves": [[{"id": "fakeout"}], [{"id": "protect"}]],
        "available_switches": [[{"species": "gholdengo"}], []],
    }
    assert obs.is_doubles(board) is True
    assert obs.available_moves(board)[0][0]["id"] == "fakeout"
    assert obs.available_switches(board)[1] == []


def test_a_flat_move_list_is_wrapped_as_one_slot():
    """Singles sends one flat list; it must not be mistaken for two slots."""
    board = {"available_moves": [{"id": "earthquake"}, {"id": "protect"}]}
    assert obs.available_moves(board) == [[{"id": "earthquake"}, {"id": "protect"}]]


def test_force_switch_handles_bool_and_list():
    assert obs.force_switch({"force_switch": True}) == [True]
    assert obs.force_switch({"force_switch": [True, False]}) == [True, False]
    assert obs.force_switch({}) == []


# -- clock ------------------------------------------------------------------


def test_clock_readers():
    board = {"clock": {"decision_seconds_left": 41, "bank_seconds_left": 312}}
    assert obs.decision_seconds_left(board) == 41
    assert obs.bank_seconds_left(board) == 312
    assert obs.decision_seconds_left({}) is None
    # A bool is not a number here, whatever Python's isinstance says.
    assert obs.decision_seconds_left({"clock": {"decision_seconds_left": True}}) is None


def test_platform_acts_before_showdown_does():
    """The platform plays a random legal choice at 50 s (85 s at preview),
    before Showdown's own 55 s / 90 s timers — the policy must budget for the
    earlier deadline."""
    assert obs.PLATFORM_DECISION_SECONDS == 50
    assert obs.PLATFORM_PREVIEW_SECONDS == 85


# -- pool capture -----------------------------------------------------------


def test_merge_unions_by_card_id_and_keeps_the_first_sighting():
    existing = [{"card_id": "a", "species": "First"}]
    incoming = [{"card_id": "a", "species": "Second"}, {"card_id": "b", "species": "New"}]
    merged = pool.merge(existing, incoming)
    assert [c["card_id"] for c in merged] == ["a", "b"]
    assert merged[0]["species"] == "First"


def test_merge_ignores_junk():
    assert pool.merge(None, [{"card_id": "a"}, "nope", None, {"no_id": 1}]) == [{"card_id": "a"}]
    assert pool.merge(None, None) == []


def test_save_and_load_round_trip(tmp_path):
    target = tmp_path / "pool.json"
    assert pool.load_pool(target) is None
    pool.save_pool([{"card_id": "b"}, {"card_id": "a"}], target)
    assert [c["card_id"] for c in pool.load_pool(target)] == ["a", "b"]


def test_load_pool_tolerates_a_corrupt_file(tmp_path):
    target = tmp_path / "pool.json"
    target.write_text("{not json", encoding="utf-8")
    assert pool.load_pool(target) is None


def test_capture_banks_the_pool_and_grows_across_observations(tmp_path):
    target = tmp_path / "pool.json"
    first = pool.capture(state(DRAFT), path=target)
    assert len(first) == 2

    # The pool shrinks as cards are taken, so a later observation adds nothing
    # new but must not lose what we already banked.
    later = {**DRAFT, "available_cards": [CARD], "pick_number": 5}
    after = pool.capture(state(later), path=target)
    assert len(after) == 2
    assert pool.load_pool(target) is not None


def test_capture_returns_none_when_there_are_no_cards(tmp_path):
    target = tmp_path / "pool.json"
    assert pool.capture(state({"phase": "moving"}), path=target) is None


def test_is_complete_is_informational():
    assert pool.is_complete([]) is False
    assert pool.is_complete(None) is False
    assert pool.is_complete([{"card_id": "a"}]) is False
    assert pool.is_complete([{"card_id": str(i)} for i in range(18)]) is True


def test_capture_ignores_id_only_cards_and_writes_nothing(tmp_path):
    """A state may publish ``available_card_ids`` without the definitions.
    Banking a bare id would be actively harmful: ``merge`` keeps the first
    sighting, so it would block the real set seen later."""
    target = tmp_path / "pool.json"
    ids_only = {"phase": "draft", "available_card_ids": ["vgc-gyarados"]}

    assert pool.capture(state(ids_only), path=target) is None
    assert not target.exists(), "capture must not write a file it has nothing to put in"


def test_capture_banks_a_full_card_after_seeing_only_ids_first(tmp_path):
    """The id-only sighting must not poison a later real definition."""
    target = tmp_path / "pool.json"
    pool.capture(state({"phase": "draft", "available_card_ids": ["vgc-gyarados"]}), path=target)

    banked = pool.capture(state(DRAFT), path=target)
    assert banked is not None
    species = {card["card_id"]: card.get("species") for card in banked}
    assert species["vgc-urshifu-rapid-strike"] == "Urshifu-Rapid-Strike"


def test_is_full_card_requires_a_species():
    assert pool.is_full_card({"card_id": "a", "species": "Gyarados"}) is True
    assert pool.is_full_card({"card_id": "a"}) is False
    assert pool.is_full_card({"species": "Gyarados"}) is False
    assert pool.is_full_card({"card_id": "", "species": "Gyarados"}) is False
    assert pool.is_full_card(None) is False
    assert pool.is_full_card("junk") is False
