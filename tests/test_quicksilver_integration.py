"""End-to-end: Quicksilver's agent driven by the platform's real runner.

The unit tests prove the pieces in isolation. This proves the *contract*: that
``create_agent()`` is accepted by ``altruagent.runner.run_game``, that our
dispatcher is resolved correctly for ``choose_action`` and ``choose_message``,
and that the structured payload we build is the one the runner actually sends.

That matters because the runner has non-obvious resolution rules — a returned
object is used via its ``choose_action`` method, ``choose_message`` is looked up
on the *original* object, and a ``dict`` return is submitted as the ``action``
payload rather than as ``play_action`` arguments. A unit test can't catch a
mismatch in any of those.
"""

from __future__ import annotations

from altruagent import DecisionContext, GameState, run_game

SESSION = "session-1"
DOUBLES = "pokemon_vgc_doubles_draft"


class FakeGame:
    """Minimal scripted stand-in for ``MCPGameSession``."""

    def __init__(self, *, states, action_results=None, results=None, messages=None):
        self._states = list(states)
        self._action_results = list(action_results or [])
        self._results = list(results or [])
        self._messages = list(messages or [])
        self.play_action_calls: list[dict] = []
        self.send_message_calls: list[dict] = []

    def get_state(self) -> GameState:
        return self._states.pop(0)

    def wait_for_update(self, **kwargs) -> GameState:
        return self._states.pop(0)

    def get_legal_actions(self) -> dict:
        raise AssertionError("the runner should not need a separate legal-actions read")

    def play_action(self, *, action_id=None, action=None, state_version) -> dict:
        self.play_action_calls.append(
            {"action_id": action_id, "action": action, "state_version": state_version}
        )
        return self._action_results.pop(0)

    def send_message(self, *, message_type, content=None, recipients=None) -> dict:
        self.send_message_calls.append(
            {"message_type": message_type, "content": content, "recipients": recipients}
        )
        return self._messages.pop(0)

    def resign(self) -> dict:
        raise AssertionError("not resigning in these tests")

    def get_result(self) -> dict:
        return self._results.pop(0)


def mcp_state(**overrides) -> GameState:
    payload = {
        "session_id": SESSION,
        "game_type": DOUBLES,
        "status": "in_progress",
        "state_version": 1,
        "observation": {"phase": "moving", "is_doubles": True},
        "phase": "moving",
        "messaging_enabled": False,
        "terminated_messaging": [],
        "new_messages": [],
        "current_actor": {"agent_id": "us", "position": 1},
        "is_current_actor": True,
        "is_terminal": False,
        "legal_actions": {"actions": []},
    }
    payload.update(overrides)
    return GameState.from_mcp_state(payload)


def terminal_state(**overrides) -> GameState:
    overrides.setdefault("is_terminal", True)
    overrides.setdefault("is_current_actor", False)
    overrides.setdefault("legal_actions", {"actions": []})
    return mcp_state(**overrides)


def context(game_type: str = DOUBLES) -> DecisionContext:
    return DecisionContext(
        session_id=SESSION,
        tournament_id=None,
        game_type=game_type,
        agent_id="us",
        seat_position=1,
    )


def doubles_turn_state(*, decision_seconds=50, bank_seconds=400) -> GameState:
    """A real doubles turn: one legal action whose ``input.action`` is a
    template for us to fill in."""
    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "board_position": 1,
                "options": [
                    {"type": "move", "move_id": "fakeout", "targets": [1, 2]},
                    {"type": "switch", "species": "gholdengo"},
                ],
            },
            {
                "slot": 1,
                "board_position": 2,
                "options": [{"type": "move", "move_id": "protect", "targets": [0]}],
            },
        ],
    }
    return mcp_state(
        observation={
            "phase": "moving",
            "is_doubles": True,
            "clock": {
                "decision_seconds_left": decision_seconds,
                "bank_seconds_left": bank_seconds,
            },
        },
        legal_actions={
            "actions": [
                {"action_id": "doubles_turn", "label": "Turn", "input": {"action": template}}
            ]
        },
    )


def result_dict() -> dict:
    return {
        "session_id": SESSION,
        "is_terminal": True,
        "status": "completed",
        "returns": {"us": 1.0},
        "termination_reason": "completed",
    }


# -- the doubles turn -------------------------------------------------------


def test_the_real_runner_submits_our_doubles_turn():
    from agent.agent import create_agent

    game = FakeGame(
        states=[doubles_turn_state(), terminal_state()],
        action_results=[{}],  # no "state" in the reply, so the runner re-reads
        results=[result_dict()],
    )

    final = run_game(game, context(), create_agent(), sleep=lambda _s: None)

    assert final.is_terminal is True
    assert len(game.play_action_calls) == 1
    call = game.play_action_calls[0]
    # A dict decision is submitted as the structured `action`, with no action_id.
    assert call["action_id"] is None
    action = call["action"]
    assert action["type"] == "doubles_turn"
    assert action["slot_0"] == {"type": "move", "move_id": "fakeout", "target": 1}
    assert action["slot_1"] == {"type": "move", "move_id": "protect", "target": 0}
    assert call["state_version"] == 1


def test_a_pokemon_draft_pick_goes_through_as_an_action_id():
    from agent.agent import create_agent

    game = FakeGame(
        states=[
            mcp_state(
                observation={"phase": "draft", "available_card_ids": ["vgc-gyarados"]},
                phase="draft",
                legal_actions={
                    "actions": [
                        {
                            "action_id": "draft_pick:vgc-gyarados",
                            "label": "Gyarados",
                            "input": {"action_id": "draft_pick:vgc-gyarados"},
                        }
                    ]
                },
            ),
            terminal_state(),
        ],
        action_results=[{}],
        results=[result_dict()],
    )

    run_game(game, context(), create_agent(), sleep=lambda _s: None)

    call = game.play_action_calls[0]
    # An enumerated action is normalized to its `action_id` (the runner matches
    # it against the live legal actions), so the structured `action` stays None.
    assert call["action_id"] == "draft_pick:vgc-gyarados"
    assert call["action"] is None


# -- messaging --------------------------------------------------------------


def test_the_dispatcher_auto_terminates_the_werewolf_discussion():
    """Our dispatcher *does* define choose_message, so the runner will always
    call it. In a discussion window it must answer with a terminate — whether
    that comes from the Werewolf delegate's own (currently silent) policy or
    from the runtime default the dispatcher supplies when a delegate has none
    (covered in test_quicksilver.py)."""
    from agent.agent import create_agent

    game = FakeGame(
        states=[
            mcp_state(
                game_type="werewolf",
                phase="messaging",
                messaging_enabled=True,
                is_current_actor=False,
                legal_actions={"actions": []},
            ),
            terminal_state(game_type="werewolf"),
        ],
        messages=[{"accepted": True, "session_id": SESSION, "phase": "moving"}],
        results=[result_dict()],
    )

    run_game(game, context("werewolf"), create_agent(), sleep=lambda _s: None)

    assert len(game.send_message_calls) == 1
    assert game.send_message_calls[0]["message_type"] == "terminate"
    assert game.play_action_calls == []


# -- the draft --------------------------------------------------------------

FLUTTER_MANE = {
    "card_id": "c-flutter",
    "species": "Flutter Mane",
    "item": "Booster Energy",
    "ability": "Protosynthesis",
    "nature": "Timid",
    "level": 50,
    "evs": {},
    "ivs": {},
    "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
}
CATERPIE = {
    "card_id": "c-caterpie",
    "species": "Caterpie",
    "item": "Leftovers",
    "ability": "Shield Dust",
    "nature": "Bold",
    "level": 50,
    "evs": {},
    "ivs": {},
    "moves": ["Tackle", "String Shot"],
}


def test_the_runner_plays_the_draft_card_our_policy_chose(tmp_path, monkeypatch):
    """End to end through run_game: a weak card is offered *first*, and the
    policy must still take the strong one. This is the test that would catch
    the draft logic silently degrading back to 'first legal action'."""
    from agent.agent import create_agent

    # The agent banks the card pool as a side effect of seeing a draft. Point
    # that at a temp file so the test never writes the repo's real capture.
    monkeypatch.setenv("QUICKSILVER_POOL_PATH", str(tmp_path / "pool.json"))

    draft_state = mcp_state(
        observation={
            "phase": "draft",
            "pick_number": 1,
            "first_drafter": "us",
            "available_cards": [CATERPIE, FLUTTER_MANE],
            "rosters": {"us": [], "them": []},
            "picks": [],
        },
        phase="draft",
        legal_actions={
            "actions": [
                {
                    "action_id": "draft_pick:c-caterpie",
                    "label": "Caterpie",
                    "input": {"action_id": "draft_pick:c-caterpie"},
                },
                {
                    "action_id": "draft_pick:c-flutter",
                    "label": "Flutter Mane",
                    "input": {"action_id": "draft_pick:c-flutter"},
                },
            ]
        },
    )
    game = FakeGame(
        states=[draft_state, terminal_state()],
        action_results=[{}],
        results=[result_dict()],
    )

    run_game(game, context(), create_agent(), sleep=lambda _s: None)

    assert game.play_action_calls[0]["action_id"] == "draft_pick:c-flutter"


# -- the battle -------------------------------------------------------------


def battle_mon(species, *, hp, max_hp):
    return {
        "species": species,
        "current_hp": hp,
        "max_hp": max_hp,
        "current_hp_fraction": hp / max_hp,
        "boosts": {},
        "fainted": hp <= 0,
    }


def test_the_runner_plays_the_move_our_battle_policy_chose():
    """End to end through run_game: Aqua Jet is offered first and Close Combat
    is 2x into the target, so the policy must pick Close Combat. This catches
    the battle logic silently degrading back to 'first legal option'."""
    from agent.agent import create_agent

    template = {
        "type": "doubles_turn",
        "slots": [
            {
                "slot": 0,
                "options": [
                    {"type": "move", "move_id": "Aqua Jet", "targets": [1, 2]},
                    {"type": "move", "move_id": "Close Combat", "targets": [1, 2]},
                ],
            },
            {
                "slot": 1,
                "options": [{"type": "move", "move_id": "Knock Off", "targets": [1, 2]}],
            },
        ],
    }
    state = mcp_state(
        observation={
            "phase": "moving",
            "is_doubles": True,
            "turn": 2,
            "active_pokemon": [
                battle_mon("Urshifu-Rapid-Strike", hp=176, max_hp=176),
                battle_mon("Incineroar", hp=202, max_hp=202),
            ],
            "opponent_active_pokemon": [battle_mon("Incineroar", hp=202, max_hp=202)],
        },
        legal_actions={
            "actions": [
                {"action_id": "doubles_turn", "label": "Turn", "input": {"action": template}}
            ]
        },
    )
    game = FakeGame(
        states=[state, terminal_state()],
        action_results=[{}],
        results=[result_dict()],
    )

    run_game(game, context(), create_agent(), sleep=lambda _s: None)

    action = game.play_action_calls[0]["action"]
    assert action["type"] == "doubles_turn"
    assert action["slot_0"]["move_id"] == "Close Combat"
    assert action["slot_0"]["target"] == 1


def test_the_runner_plays_the_lineup_our_policy_chose(monkeypatch, tmp_path):
    """Team Preview through run_game: the Grass Pokémon cannot hurt a Fire team,
    so the four brought must not include it."""
    from agent.agent import create_agent
    from pokemon import dex

    monkeypatch.setenv("QUICKSILVER_POOL_PATH", str(tmp_path / "pool.json"))

    roster = [
        "gyarados",
        "urshifurapidstrike",
        "rillaboom",
        "amoonguss",
        "fluttermane",
        "barraskewda",
    ]
    moves = {
        "gyarados": ["Waterfall", "Crunch", "Protect"],
        "urshifurapidstrike": ["Surging Strikes", "Close Combat", "Protect"],
        "rillaboom": ["Grassy Glide", "Knock Off", "Protect"],
        "amoonguss": ["Sludge Bomb", "Pollen Puff", "Protect"],
        "fluttermane": ["Moonblast", "Shadow Ball", "Protect"],
        "barraskewda": ["Liquidation", "Aqua Jet", "Protect"],
    }

    def entry(species, with_moves=False):
        block = {"species": species, "name": species, "types": list(dex.types_of(species))}
        if with_moves:
            block["moves"] = moves[species]
        return block

    template = {"type": "select_lineup", "roster": roster}
    state = mcp_state(
        observation={
            "phase": "team_preview",
            "your_roster": [entry(s, with_moves=True) for s in roster],
            "opponent_roster": [
                entry(s) for s in ("charizard", "talonflame", "incineroar", "armarouge")
            ],
        },
        phase="team_preview",
        legal_actions={
            "actions": [
                {"action_id": "select_lineup", "label": "Lineup", "input": {"action": template}}
            ]
        },
    )
    game = FakeGame(
        states=[state, terminal_state()],
        action_results=[{}],
        results=[result_dict()],
    )

    run_game(game, context(), create_agent(), sleep=lambda _s: None)

    action = game.play_action_calls[0]["action"]
    assert action["type"] == "select_lineup"
    assert len(action["bring"]) == 4
    assert len(action["leads"]) == 2
    # The ids must come from the server's own roster, not be invented.
    assert set(action["bring"]) <= set(roster)
    assert set(action["leads"]) <= set(action["bring"])
    assert "rillaboom" not in action["bring"]
