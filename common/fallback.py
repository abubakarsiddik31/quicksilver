"""Safe default for a game type Quicksilver doesn't implement.

The dispatcher (``common.dispatch``) falls back here for an unrecognised
``context.game_type`` — Red Alert, which this submission does not enter, or any
future addition.

The body is deliberately the dumbest policy that is still legal: take the first
enumerated action. Two properties make that the right choice for a game we
don't have a policy for:

- Every game that enumerates its moves puts a legal one first, so this is
  always playable.
- A game that enumerates *nothing* (a real-time game such as Red Alert, where a
  move is a batch of orders the agent writes itself) makes this raise
  ``IndexError`` from a line naming ``legal_actions``. The runner recognises
  exactly that and reports plainly that the agent can't play the game — far
  better than returning ``WAIT`` and silently idling to a loss with no signal.

Failing loudly is correct here precisely because there is no policy to degrade
to. Where a policy *does* exist (``pokemon/``), the opposite holds: degrade to
something legal rather than crash, because a crash hands the rest of the game
to the server's auto-play.
"""

from altruagent import DecisionContext, GameState


def choose_action(state: GameState, context: DecisionContext):
    return state.legal_actions[0]


def create_agent():
    return choose_action
