"""Quicksilver's agent entry point — the one name the runtime looks for.

The runtime (``python -m agent``) imports this module and calls
``create_agent()`` once per match, in that match's own process. This module is
deliberately thin: all decisions live in the game packages, reached through
``common.dispatch``, which routes by ``context.game_type``.

- ``common/``    game-type dispatch and the safe fallback
- ``pokemon/``   Pokémon VGC doubles draft policy (primary track)
- ``werewolf/``  Werewolf policy (secondary track)

``altruagent/`` is the platform's vendored starter SDK and is not ours to
edit. Run a different agent without touching this file using
``python -m agent --match --agent <module>``.
"""

from common.dispatch import create_agent

__all__ = ["create_agent"]
