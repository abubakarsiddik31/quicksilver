"""Tests verifying strategic LLM advisors for draft and team preview."""

from __future__ import annotations

from pokemon import draft, lineup


def test_advise_lineup_selects_chosen_plan():
    plan_a = {"type": "select_lineup", "bring": ["A", "B", "C", "D"], "leads": ["A", "B"]}
    plan_b = {"type": "select_lineup", "bring": ["A", "B", "C", "E"], "leads": ["A", "C"]}
    candidates = [
        (plan_a, 50.0, "Balanced lead"),
        (plan_b, 48.0, "Counter lead"),
    ]

    class MockClient:
        def generate(self, prompt, **kwargs):
            return "CHOICE: PLAN_B - Hard counters opponent's predicted Flutter Mane lead."

    obs = {"opponent_roster": [{"species": "Flutter Mane"}]}
    chosen = lineup.advise_lineup(MockClient(), obs, candidates)
    assert chosen == plan_b


def test_advise_draft_pick_selects_synergy_card():
    card_pelipper = {"card_id": "vgc-pelipper", "species": "Pelipper", "item": "Damp Rock", "moves": ["Hurricane"]}
    card_urshifu = {"card_id": "vgc-urshifu-rapid-strike", "species": "Urshifu-Rapid-Strike", "item": "Mystic Water", "moves": ["Surging Strikes"]}

    obs = {
        "available_cards": [card_pelipper, card_urshifu],
        "rosters": {"us": [{"species": "Archaludon"}], "them": []},
    }

    # Urshifu has higher score -> OPTION_A; Pelipper is OPTION_B
    class MockClient:
        def generate(self, prompt, **kwargs):
            return "CHOICE: OPTION_A - Top offensive threat with rain synergy."

    pick = draft.advise_draft_pick(MockClient(), obs, agent_id="us")
    assert pick == "vgc-urshifu-rapid-strike"


def test_generate_candidate_lineups_creates_valid_lineups():
    template = {
        "type": "select_lineup",
        "roster": ["Incineroar", "Urshifu-Rapid-Strike", "Flutter Mane", "Rillaboom", "Archaludon", "Ting-Lu"],
    }
    obs = {
        "your_roster": [{"species": sp} for sp in template["roster"]],
        "opponent_roster": [{"species": "Chien-Pao"}, {"species": "Dragonite"}],
    }

    candidates = lineup.generate_candidate_lineups(template, obs)
    assert len(candidates) > 0
    for ldict, score, desc in candidates:
        assert len(ldict["bring"]) == 4
        assert len(ldict["leads"]) == 2
        for lead in ldict["leads"]:
            assert lead in ldict["bring"]
