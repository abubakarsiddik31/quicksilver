"""Tests for the Pokemon VGC tactical advisor and candidate plans generator."""

from __future__ import annotations

from pokemon import advisor, battle


def test_format_board_summary():
    obs = {
        "turn": 2,
        "active_pokemon": [
            {"species": "Urshifu-Rapid-Strike", "hp": 150, "max_hp": 175, "status": "healthy"},
            {"species": "Incineroar", "hp": 200, "max_hp": 200, "status": "healthy"},
        ],
        "opponent_active_pokemon": [
            {"species": "Chien-Pao", "hp": 140, "max_hp": 155, "status": "healthy"},
            {"species": "Gholdengo", "hp": 160, "max_hp": 162, "status": "healthy"},
        ],
        "weather": "rain",
        "terrain": "grassy",
    }
    summary = advisor.format_board_summary(obs)
    assert "Urshifu-Rapid-Strike" in summary
    assert "Chien-Pao" in summary
    assert "Weather: rain" in summary
    assert "Terrain: grassy" in summary


def test_parse_advisor_choice():
    labels = ["PLAN_A", "PLAN_B", "PLAN_C"]
    
    # Exact format
    reply1 = "CHOICE: PLAN_B - Opponent is likely to Protect Chien-Pao."
    assert advisor.parse_advisor_choice(reply1, labels) == "PLAN_B"

    # Lowercase / variant
    reply2 = "choice: plan_a is the safest option here."
    assert advisor.parse_advisor_choice(reply2, labels) == "PLAN_A"

    # Embedded in prose
    reply3 = "I strongly recommend selecting PLAN_C to secure the lead."
    assert advisor.parse_advisor_choice(reply3, labels) == "PLAN_C"

    # Invalid / hallucinated label
    reply4 = "You should use PLAN_D or something else."
    assert advisor.parse_advisor_choice(reply4, labels) is None

    # Empty or junk
    assert advisor.parse_advisor_choice(None, labels) is None
    assert advisor.parse_advisor_choice("", labels) is None


def test_advisor_selects_chosen_plan():
    plan_a = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "surgingstrikes", "target": 1}}
    plan_b = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "aquajet", "target": 1}}

    candidates = [
        (plan_a, 80.0, "Heavy attack on slot 1"),
        (plan_b, 75.0, "Priority attack on slot 1"),
    ]

    class MockClient:
        def generate(self, prompt, **kwargs):
            return "CHOICE: PLAN_B - Anticipating focus-sash priority cleanup."

    obs = {"turn": 1, "active_pokemon": [], "opponent_active_pokemon": []}
    result = advisor.advise(MockClient(), obs, candidates)
    assert result == plan_b


def test_advisor_fails_gracefully_to_none_on_timeout():
    plan_a = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "surgingstrikes", "target": 1}}
    candidates = [(plan_a, 80.0, "attack")]

    class CrashingClient:
        def generate(self, prompt, **kwargs):
            raise TimeoutError("Model timed out")

    obs = {"turn": 1}
    # When only 1 candidate, returns candidate directly
    assert advisor.advise(CrashingClient(), obs, candidates) == plan_a

    # When multiple candidates and client crashes, returns None (caller uses Plan A)
    candidates2 = [(plan_a, 80.0, "attack"), ({"type": "doubles_turn"}, 70.0, "alt")]
    assert advisor.advise(CrashingClient(), obs, candidates2) is None


def test_generate_candidate_plans_prevents_duplicate_switches():
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "switch", "species": "Rillaboom"},
                    {"type": "move", "move_id": "protect"},
                ],
            },
            {
                "slot": 1,
                "options": [
                    {"type": "switch", "species": "Rillaboom"},
                    {"type": "move", "move_id": "protect"},
                ],
            },
        ],
    }

    obs = {
        "phase": "moving",
        "turn": 1,
        "active_pokemon": [
            {"species": "Urshifu-Rapid-Strike", "hp": 100, "max_hp": 175},
            {"species": "Incineroar", "hp": 100, "max_hp": 200},
        ],
        "opponent_active_pokemon": [
            {"species": "Flutter Mane", "hp": 100, "max_hp": 130},
        ],
    }

    candidates = battle.generate_candidate_plans(template, obs)
    assert len(candidates) > 0

    for plan, score, desc in candidates:
        s0 = plan.get("slot_0", {})
        s1 = plan.get("slot_1", {})
        if s0.get("type") == "switch" and s1.get("type") == "switch":
            assert s0.get("species") != s1.get("species")


def test_is_tactical_dilemma_logic():
    obs_healthy = {"clock": {"bank_seconds_left": 300.0}}
    obs_low_bank = {"clock": {"bank_seconds_left": 45.0}}

    plan1 = ({"type": "doubles_turn"}, 70.0, "plan 1")
    plan2 = ({"type": "doubles_turn"}, 65.0, "plan 2")  # delta = 5.0 -> dilemma
    plan_blowout = ({"type": "doubles_turn"}, 30.0, "plan blowout")  # delta = 40.0 -> no dilemma

    # Close candidates and healthy clock -> True
    assert battle.is_tactical_dilemma([plan1, plan2], obs_healthy) is True

    # Obvious winner -> False
    assert battle.is_tactical_dilemma([plan1, plan_blowout], obs_healthy) is False

    # Low bank timer (< 90s) -> False (speed conservation mode)
    assert battle.is_tactical_dilemma([plan1, plan2], obs_low_bank) is False
