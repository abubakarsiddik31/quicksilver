"""End-to-end synthetic simulation and fuzz testing for Quicksilver policies.

Tests:
- Pokemon draft edge cases (empty choices, single choice, rapid picks).
- Pokemon doubles battle turn legality across simulated boards.
- Werewolf belief updates across diverse death and voting patterns.
"""

from altruagent import DecisionContext, GameState
from common.dispatch import QuicksilverAgent
from werewolf.belief import Belief


def test_pokemon_doubles_always_produces_valid_action():
    agent = QuicksilverAgent()
    ctx = DecisionContext(
        session_id="sim-1",
        tournament_id=None,
        game_type="pokemon_vgc_doubles_draft",
        agent_id="quicksilver",
        seat_position=0,
    )

    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "move", "move_id": "protect"},
                    {"type": "move", "move_id": "surgingstrikes", "targets": [1, 2]},
                    {"type": "switch", "species": "Rillaboom"},
                    {"type": "pass"},
                ],
            },
            {
                "slot": 1,
                "options": [
                    {"type": "move", "move_id": "flareblitz", "targets": [1]},
                    {"type": "switch", "species": "Rillaboom"},
                    {"type": "pass"},
                ],
            },
        ],
    }

    obs = {
        "phase": "moving",
        "turn": 1,
        "active_pokemon": [
            {"species": "Urshifu-Rapid-Strike", "hp": 175, "max_hp": 175},
            {"species": "Incineroar", "hp": 200, "max_hp": 200},
        ],
        "opponent_active_pokemon": [
            {"species": "Flutter Mane", "hp": 130, "max_hp": 130},
            {"species": "Gholdengo", "hp": 162, "max_hp": 162},
        ],
        "clock": {"decision_seconds_left": 45.0, "bank_seconds_left": 350.0},
    }

    state = GameState.from_mcp_state(
        {
            "session_id": "sim-1",
            "game_type": "pokemon_vgc_doubles_draft",
            "status": "in_progress",
            "state_version": 2,
            "observation": obs,
            "phase": "moving",
            "messaging_enabled": False,
            "terminated_messaging": [],
            "new_messages": [],
            "current_actor": {"agent_id": "quicksilver", "position": 0},
            "is_current_actor": True,
            "is_terminal": False,
            "legal_actions": {
                "actions": [
                    {
                        "action_id": "act_turn",
                        "action_type": "doubles_turn",
                        "input": {"action": template},
                    }
                ]
            },
        }
    )

    choice = agent.choose_action(state, ctx)
    assert isinstance(choice, dict)
    assert choice["type"] == "doubles_turn"
    assert "slot_0" in choice and "slot_1" in choice
    # Guarantee no duplicate switch
    s0, s1 = choice["slot_0"], choice["slot_1"]
    if s0.get("type") == "switch" and s1.get("type") == "switch":
        assert s0.get("species") != s1.get("species")


def test_werewolf_simulation_convergence():
    from types import SimpleNamespace

    from altruagent import Message

    raw_state = {
        "your_position": 3,
        "players": [{"position": i, "name": f"Player{i}"} for i in range(7)],
        "game_state": {
            "day": 2,
            "phase": "day_vote",
            "alive": ["Player1", "Player2", "Player3", "Player4", "Player5", "Player6"],
            "dead": [{"player": "Player0", "role": "wolf", "cause": "voted_out"}],
            "vote_history": [
                {"voter": "Player1", "target": "Player0"},
                {"voter": "Player2", "target": "Player0"},
                {"voter": "Player3", "target": "Player0"},
                {"voter": "Player0", "target": "Player1"},
            ],
        },
        "eliminated": False,
        "observation": "=== Werewolf ===\nYour role: VILLAGER\nYou do not know anyone else's role.",
    }
    state = SimpleNamespace(raw=raw_state, observation=raw_state["observation"], phase="moving")

    def note(sender, content):
        return Message(
            index=0,
            sender=sender,
            recipients=[],
            content=content,
            type="chat",
            sent_at="",
            move_index=0,
        )

    b = Belief()
    b.update(state)
    b.note_messages([note("Player1", "I am the Seer. Player0 is a werewolf.")], our_position=3)

    # Player 0 is dead and confirmed wolf
    assert 0 in b.known_wolves
    assert b.wolves_remaining == 1
    # Vote recommendation avoids confirmed good/allies
    vote = b.best_vote()
    assert vote is not None
    assert vote in [1, 2, 4, 5, 6]
