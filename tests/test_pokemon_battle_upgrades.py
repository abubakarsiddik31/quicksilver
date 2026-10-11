"""Tests verifying the competitive upgrades to battle and draft policies."""

from __future__ import annotations

from pokemon import battle, draft


def mon(species: str, hp: int = 100, max_hp: int = 100):
    return {
        "species": species,
        "current_hp": hp,
        "max_hp": max_hp,
        "current_hp_fraction": hp / max_hp,
        "fainted": False,
        "boosts": {},
    }


def test_finishing_off_wounded_foe_receives_ko_bonus():
    """A move dealing 50 damage into an opponent with 30 remaining HP
    (out of 200 max HP) must be recognized as a guaranteed KO (+60 KO bonus)."""
    attacker_card = battle.Card(
        card_id="vgc-urshifu",
        species="Urshifu-Rapid-Strike",
        stats={"hp": 175, "atk": 200, "def": 120, "spa": 70, "spd": 80, "spe": 149},
        nature="Jolly",
        item="Mystic Water",
        ability="Unseen Fist",
        moves=("Surging Strikes", "Aqua Jet"),
    )
    # Wounded Incineroar: 30 HP left out of 202
    defender_summary = mon("Incineroar", hp=30, max_hp=202)
    defender_card = battle.Card(
        card_id="vgc-incineroar",
        species="Incineroar",
        stats={"hp": 202, "atk": 135, "def": 110, "spa": 90, "spd": 110, "spe": 80},
        nature="Careful",
        item="Safety Goggles",
        ability="Intimidate",
        moves=(),
    )

    attacker = battle.Active(card=attacker_card, summary=mon("Urshifu-Rapid-Strike", 175, 175))
    defender = battle.Active(card=defender_card, summary=defender_summary)

    # Aqua Jet is a weak 40-power move, but deals ~40-50 damage into wounded Incineroar
    score, target = battle.move_score(
        "Aqua Jet",
        attacker=attacker,
        defenders=[defender],
        targets=[1],
        turn=2,
    )

    # Score must include KO_BONUS (60.0) because 40+ damage >= 30 remaining HP
    assert score >= 60.0


def test_status_moves_receive_competitive_scores():
    """Tailwind on turn 1 scores high (+42.0) instead of being ignored (0.0)."""
    card = battle.Card(
        card_id="vgc-talonflame",
        species="Talonflame",
        stats={"hp": 150, "atk": 100, "def": 90, "spa": 90, "spd": 85, "spe": 178},
        nature="Jolly",
        item="Sharp Beak",
        ability="Gale Wings",
        moves=("Tailwind",),
    )
    active = battle.Active(card=card, summary=mon("Talonflame", 150, 150))

    score_t1, _ = battle.move_score("Tailwind", attacker=active, defenders=[], targets=[0], turn=1)
    assert score_t1 == 42.0

    score_reflect, _ = battle.move_score("Reflect", attacker=active, defenders=[], targets=[0], turn=1)
    assert score_reflect == 32.0


def test_overkill_avoids_redundant_double_targeting():
    """If Slot 0 guarantees a KO on Defender 0, Slot 1 should be penalized for
    attacking Defender 0 when another living opponent exists."""
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "move", "move_id": "Surging Strikes", "targets": [1]},
                ],
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

    obs = {
        "phase": "moving",
        "turn": 2,
        "active_pokemon": [
            mon("Urshifu-Rapid-Strike", hp=175, max_hp=175),
            mon("Incineroar", hp=200, max_hp=200),
        ],
        "opponent_active_pokemon": [
            mon("Incineroar", hp=30, max_hp=200),  # Slot 0 easily KOs with Surging Strikes
            mon("Gholdengo", hp=160, max_hp=160),  # Second opponent
        ],
    }

    candidates = battle.generate_candidate_plans(template, obs)
    assert len(candidates) > 0
    best_plan = candidates[0][0]

    # Slot 1 should target defender 2 (Gholdengo) instead of redundant defender 1
    assert best_plan["slot_1"]["target"] == 2


def test_draft_archetype_bonus():
    """Incineroar and speed control (Talonflame) receive archetype bonuses."""
    card_incineroar = {"species": "Incineroar", "card_id": "vgc-incineroar", "stats": {"hp": 95, "atk": 115, "def": 90, "spa": 80, "spd": 90, "spe": 60}}
    bonus = draft.archetype_bonus(card_incineroar, [])
    assert bonus >= 3.0
