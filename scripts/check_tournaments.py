"""RETIRED: listed and inspected platform tournaments as a platform agent,
which the platform turned off together with the platform API key.

Tournaments are now run on the UCLA tournament site. You register your agent
on the tournament dashboard, and `python -m agent --tournament` plays its
tournament games automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from altruagent.notices import AGENT_GUIDE_URL, DASHBOARD_URL  # noqa: E402


def main() -> int:
    print("scripts/check_tournaments.py was retired with the platform API key (ALTRUAGENT_API_KEY).")
    print(f"Register your agent for tournaments on your dashboard: {DASHBOARD_URL}")
    print("Then keep `python -m agent --tournament` running; it plays your tournament games.")
    print(f"Guide: {AGENT_GUIDE_URL}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
