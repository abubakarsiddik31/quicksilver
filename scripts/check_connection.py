"""Check that your agent can connect: your Official Agent Key, the control
plane, game assignments and your agent factory. Never plays anything.

This used to check the platform API key (ALTRUAGENT_API_KEY), which was
retired. It now runs the same checks as:

    python -m agent --check-tournament

Run:
    python scripts/check_connection.py [--agent MODULE[:FACTORY]]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (``python scripts/check_connection.py``)
# without having pip-installed the project first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import main as agent_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return agent_main(["--check-tournament", *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
