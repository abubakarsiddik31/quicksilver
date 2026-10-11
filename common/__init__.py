"""Shared infrastructure for the Quicksilver agent.

Layout::

    common/dispatch.py   game-type dispatch behind the runtime's create_agent()
    common/fallback.py   safe default for a game type Quicksilver doesn't implement
    pokemon/             Pokémon VGC doubles draft policy
    werewolf/            Werewolf policy

``altruagent/`` is the platform's vendored starter SDK (HTTP, auth, seat
grants, leases, MCP transport, the state -> decide -> submit loop). It is not
ours to edit: everything contestant-specific lives in the packages above.
"""
