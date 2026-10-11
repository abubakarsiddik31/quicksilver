"""Example: the plain choose_action/create_agent contract, nothing else.

This is what you get by default — no messaging, no per-match state, just a
function that returns `state.legal_actions[0]`. It is a placeholder: it
finishes a Werewolf game (each `LegalAction.action_id` is a seat number as
a string, "7" = abstain) and gets through a Pokémon draft
(`"draft_pick:<id>"`), but it can't finish a Pokémon match — Team Preview
and each doubles turn are templates you fill in as a `dict`, and the server
refuses the bare action — or a Red Alert match, which has no
`legal_actions` (moves are batches of orders). See `smoke_agent.py` for
valid Pokémon moves and `llm_agent.py` for all three games.

Copy this over agent/agent.py as a starting point, or as the
"moves only" half of a messaging game (see messaging_agent.py for the other
half): a contestant that never defines choose_message still plays
Werewolf's discussion windows just fine — the runtime auto-terminates each
window on its behalf.

Run it as your agent with:

    python -m agent --tournament --agent examples.basic_agent

or copy this file's contents into agent/agent.py (the default) and run
`python -m agent --tournament`.
"""

from altruagent import DecisionContext, GameState, LegalAction


def choose_action(state: GameState, context: DecisionContext) -> LegalAction:
    return state.legal_actions[0]


def create_agent():
    return choose_action
