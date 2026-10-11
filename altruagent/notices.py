"""Plain-language notices for the retired ways of connecting an agent.

AltruAgent now runs on the UCLA tournament site. Every game reaches a
self-hosted agent through its Official Agent Key
(``ALTRUAGENT_OFFICIAL_AGENT_KEY``): ``python -m agent --tournament`` plays
tournament games, ``python -m agent --match`` plays test matches (Testing),
and both flags together play both. Two older ways of connecting were turned
off on the platform:

- the platform API-key mode (``ALTRUAGENT_API_KEY``, ``sk_agent_...``,
  ``/auth/agent/*``), which the platform now answers with HTTP 410
  ``platform_agents_retired``;
- Testing claim codes (``--claim seatclaim_...``), answered with HTTP 410
  ``claim_codes_retired``.

This module has no imports so every other module can use it.
"""

from __future__ import annotations

AGENT_GUIDE_URL = "https://platform.altruagent-game.com/tournament/agent-guide"
DASHBOARD_URL = "https://platform.altruagent-game.com/tournament/dashboard"
TOURNAMENT_COMMAND = "python -m agent --tournament"
MATCH_COMMAND = "python -m agent --match"

CLAIM_CODES_RETIRED = "Testing claim codes were retired; run with --match and your Official Agent Key"
PLATFORM_KEY_RETIRED = (
    "The platform API-key mode (ALTRUAGENT_API_KEY) was retired; run with --tournament "
    "and your Official Agent Key"
)

_HOW_TO = (
    "Set ALTRUAGENT_OFFICIAL_AGENT_KEY to your Official Agent Key (generate it on the "
    "tournament dashboard's Agent Configuration page), then run:\n"
    f"    {TOURNAMENT_COMMAND}           your tournament games\n"
    f"    {MATCH_COMMAND}                your test matches\n"
    f"    {TOURNAMENT_COMMAND} --match   both\n"
    "It picks up those games automatically.\n"
    f"Guide: {AGENT_GUIDE_URL}"
)

CLAIM_CODES_RETIRED_NOTICE = f"{CLAIM_CODES_RETIRED}.\n{_HOW_TO}"
PLATFORM_KEY_RETIRED_NOTICE = f"{PLATFORM_KEY_RETIRED}.\n{_HOW_TO}"
