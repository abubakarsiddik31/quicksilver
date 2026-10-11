"""Example: a stateful Werewolf agent that actually talks.

Shows the messaging half of the contract that basic_agent.py's default
behavior skips:

- create_agent() returns a fresh object per match (not a bare function),
  so per-match state (here: which day we've already spoken on) never leaks
  between matches.
- choose_message is optional, next to choose_action, on that same object.
- SendMessage(content, recipients=None) to chat (empty/no recipients ==
  broadcast to every other player; one seat number == a private message);
  altruagent.TERMINATE_MESSAGING to vote the current window closed.
- state.new_messages holds what's been said in the current window.

Werewolf opens one discussion window per day, before the vote, with up to 5
chats per agent, a 50-word cap and a 2-minute limit. This starter doesn't
pass your remaining quota to choose_message, so track your own usage; this
agent sends one broadcast per day, keyed by the day number in the public
game record (state.raw["game_state"]["day"]), then terminates.

Once your agent is eliminated the runtime stops asking it to act or chat and
just waits for the game to end, so neither method needs to check for that.

Run it as your agent with:

    python -m agent --tournament --agent examples.messaging_agent

or copy this file's contents into agent/agent.py (the default) and run
`python -m agent --tournament`.
"""

from altruagent import DecisionContext, GameState, LegalAction, SendMessage, TERMINATE_MESSAGING


class WerewolfAgent:
    def __init__(self) -> None:
        self._spoke_on_day: int | None = None

    def choose_action(self, state: GameState, context: DecisionContext) -> LegalAction:
        # Action ids are seat numbers ("7" = abstain, day vote only); the
        # label says what it means ("Kill Player3", "Investigate Player3",
        # "Vote to eliminate Player3", "Abstain").
        # Swap this for real strategy — e.g. vote for whoever the seer
        # accused, using state.observation (your role and private info) and
        # state.raw["game_state"] (deaths with true roles, past votes).
        return state.legal_actions[0]

    def choose_message(self, state: GameState, context: DecisionContext):
        for message in state.new_messages:
            print(f"[{context.session_id}] Player{message.sender} says: {message.content!r}")

        day = (state.raw.get("game_state") or {}).get("day")
        if self._spoke_on_day == day:
            return TERMINATE_MESSAGING

        self._spoke_on_day = day
        return SendMessage("I'm a villager. Let's hear from everyone before we vote.")


def create_agent():
    return WerewolfAgent()
