"""RETIRED: listed a platform agent's matches (GET /agents/me/sessions),
which the platform turned off together with the platform API key.

Your games now reach your agent through your Official Agent Key. To see how
many games are assigned to it right now, run:

    python -m agent --check-tournament

The tournament dashboard shows your agent's current game too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from altruagent.notices import AGENT_GUIDE_URL, DASHBOARD_URL  # noqa: E402


def main() -> int:
    print("scripts/check_sessions.py was retired with the platform API key (ALTRUAGENT_API_KEY).")
    print("To see your agent's assigned games, run: python -m agent --check-tournament")
    print(f"Your dashboard: {DASHBOARD_URL}")
    print(f"Guide: {AGENT_GUIDE_URL}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
