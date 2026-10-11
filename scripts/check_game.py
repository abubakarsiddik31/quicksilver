"""Manual smoke test for one already-known GameAPI match.

RETIRED for contestants: this signs in with the platform API key
(ALTRUAGENT_API_KEY), which the platform turned off. Kept for SDK developers
as a reference for the REST debug path; `python -m agent --tournament` (and
`--match` for test matches) is how games are played now.

Reads a session you already know about (e.g. one you joined by hand via
curl — see backend/skill/03-competitions.md in Agent_ACP) and shows its
current state. By default this is READ-ONLY: it fetches state and prints it,
nothing else.

Run:
    python scripts/check_game.py

To submit exactly one move (the first entry in legal_actions — for testing
only, not a real strategy):
    python scripts/check_game.py --step-first-legal

To resign instead:
    python scripts/check_game.py --resign

``--step-first-legal`` and ``--resign`` are mutually exclusive.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running this script directly without having pip-installed the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from altruagent.client import AltruAgentClient  # noqa: E402
from altruagent.errors import AltruAgentError, ConfigurationError  # noqa: E402
from altruagent.models import GameState  # noqa: E402


def _print_state(state: GameState) -> None:
    print(f"session_id:      {state.session_id}")
    print(f"game_name:       {state.game_name}")
    print(f"phase:           {state.phase}")
    print(f"is_terminal:     {state.is_terminal}")
    print(f"current_player:  {state.current_player.name if state.current_player else None}")
    print(f"legal_actions:   {state.legal_actions}")
    print(f"legal_action_labels: {[a.label for a in state.legal_actions]}")
    print(f"next_actions:    {[a.action for a in state.next_actions]}")
    for a in state.next_actions:
        print(f"  - {a.action}: {a.hint}")
    if state.is_terminal:
        print(f"returns:         {state.returns}")
        print(f"termination_reason: {state.termination_reason}")


def _read_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigurationError(
            f"{name} is not set. Set it in your environment or .env file."
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step-first-legal",
        action="store_true",
        help="Submit the first action in legal_actions (testing only).",
    )
    group.add_argument(
        "--resign",
        action="store_true",
        help="Resign from the game.",
    )
    args = parser.parse_args()

    try:
        client = AltruAgentClient()
        game_server_url = _read_required_env("ALTRUAGENT_GAME_SERVER_URL")
        session_id = _read_required_env("ALTRUAGENT_SESSION_ID")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    try:
        session = client.game(session_id=session_id, game_server_url=game_server_url)
        state = session.state()
        print("Current state:")
        _print_state(state)

        if state.is_terminal:
            print("\nGame is already finished; nothing to submit.")
            return 0

        if args.step_first_legal:
            action_names = {a.action for a in state.next_actions}
            if "make_move" not in action_names:
                print(
                    "\nRefusing to step: next_actions does not include "
                    f"'make_move' right now ({sorted(action_names)}). It is "
                    "not this agent's turn (or messaging/another phase is "
                    "active)."
                )
                return 1
            if not state.legal_actions:
                print("\nRefusing to step: legal_actions is empty.")
                return 1
            action = state.legal_actions[0]
            try:
                action_value = int(action.action_id)
            except ValueError:
                print(
                    "\nRefusing to step: REST debug stepping only supports "
                    f"integer action ids, got {action.action_id!r}. Use MCP "
                    "via `python -m agent` for structured games."
                )
                return 1
            print(f"\nSubmitting action {action_value} (first of {state.legal_actions})...")
            new_state = session.step(action_value)
            print("Updated state:")
            _print_state(new_state)
        elif args.resign:
            print("\nResigning...")
            new_state = session.resign()
            print("Updated state:")
            _print_state(new_state)
    except AltruAgentError as exc:
        print(f"Request failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
