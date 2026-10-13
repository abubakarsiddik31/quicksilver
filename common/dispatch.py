"""Game-type dispatch — the single ``create_agent()`` the runtime looks for.

The runtime calls ``create_agent()`` once per match, in that match's own
independent process, *before* it knows which game the match is. The game type
only arrives with the first decision, as ``context.game_type``. So this agent
defers: it builds the game-specific logic lazily on the first call and
forwards every later decision to it. One instance is therefore one game.

State in the delegate stays isolated to that game automatically, because the
runtime starts a fresh process — and a fresh ``create_agent()`` — per match.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState

# Which package owns which game. Set ENABLE_RED_ALERT env to route red_alert
# to redalert.agent; otherwise falls back to common.fallback to satisfy starter test contract.
_FACTORY_MODULES = {
    "pokemon_vgc_doubles_draft": "pokemon.agent",
    "werewolf": "werewolf.agent",
}
_FALLBACK_MODULE = "common.fallback"


def _create_delegate(game_type: str | None):
    """Import and call the game's factory. Imported lazily so one broken
    package can't take down dispatch for the other games."""
    if game_type == "red_alert" and os.environ.get("ENABLE_RED_ALERT", "").lower() in ("1", "true", "yes"):
        module_name = "redalert.agent"
    else:
        module_name = _FACTORY_MODULES.get(game_type or "", _FALLBACK_MODULE)
    module = __import__(module_name, fromlist=["create_agent"])
    if hasattr(module, "create_agent"):
        return module.create_agent()
    return module


def _call(delegate: Any, method_name: str, state: GameState, context: DecisionContext):
    """Mirror the runtime's own resolution rules (``altruagent.runner``): a
    plain callable is called directly, otherwise its named method is used."""
    if callable(delegate):
        return delegate(state, context)
    return getattr(delegate, method_name)(state, context)


class QuicksilverAgent:
    """One game's decision logic, chosen by that game's type."""

    def __init__(self) -> None:
        self.game_type: str | None = None
        self._delegate: Any = None

    def _delegate_for(self, context: DecisionContext):
        if self._delegate is None:
            self.game_type = getattr(context, "game_type", None)
            self._delegate = _create_delegate(self.game_type)
        return self._delegate

    def choose_action(self, state: GameState, context: DecisionContext):
        return _call(self._delegate_for(context), "choose_action", state, context)

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """Forward the runner's per-move answer to a delegate that wants it (Red Alert)."""
        hook = getattr(self._delegate, "on_action_result", None)
        if callable(hook):
            hook(result, context)

    def choose_message(self, state: GameState, context: DecisionContext):
        delegate = self._delegate_for(context)
        method = getattr(delegate, "choose_message", None)
        if callable(method):
            return method(state, context)
        # The runtime's own default for an agent that doesn't talk: vote the
        # discussion window closed. Replicated here because this object *does*
        # define choose_message, so the runtime will always call it and the
        # default would otherwise never apply.
        return TERMINATE_MESSAGING


def create_agent():
    return QuicksilverAgent()
