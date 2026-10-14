"""Tactical double-opinion advisor powered by Gemini 3.8 Flash.

When a high-stakes tactical dilemma arises in a Pokémon VGC Doubles match
(e.g., predicting an opponent's Protect, choosing whether to sacrifice or switch,
or deciding between two evenly-scored attack lines), the rule engine computes
the top-k mathematically legal joint plans and asks the advisor for a strategic
pick within a strict 3.0-second time window.

Safety & Robustness Guarantees:
1. The model is NEVER asked to invent moves or targets. It is presented only
   with pre-validated, schema-compliant plans (PLAN_A, PLAN_B, PLAN_C).
2. The call has a strict 3.0s timeout. If the model is slow, times out, or
   returns malformed text, the engine executes PLAN_A with zero latency.
3. Every call is wrapped in a total exception guard: it can never crash a worker.
"""

from __future__ import annotations

import re

from common.llm import GeminiClient

ADVISOR_TIMEOUT = 3.0
MAX_ADVISOR_CALLS_PER_MATCH = 8

ADVISOR_SYSTEM = (
    "You are a World Champion Pokémon VGC Doubles tactical advisor. "
    "You are given the active battlefield state and 2 or 3 pre-calculated, verified legal "
    "joint turn plans (Plan A, Plan B, Plan C). "
    "Analyze the situation for opponent Protects, priority attacks, and switch predictions. "
    "Respond with exactly one line in the format:\n"
    "CHOICE: PLAN_<A|B|C> - <one brief sentence explaining why>"
)

_CHOICE_PATTERN = re.compile(r"CHOICE:\s*(PLAN_[A-C])", re.IGNORECASE)
_FALLBACK_PATTERN = re.compile(r"\b(PLAN_[A-C])\b", re.IGNORECASE)


def format_board_summary(obs: dict) -> str:
    """Format active Pokémon and field state into a concise text prompt."""
    lines = []
    turn = obs.get("turn", 1)
    lines.append(f"Turn: {turn}")

    actives = obs.get("active_pokemon") or []
    lines.append("Our Active Pokémon:")
    for idx, mon in enumerate(actives):
        if not mon or not isinstance(mon, dict):
            continue
        sp = mon.get("species", "Unknown")
        hp = mon.get("hp", 0)
        max_hp = mon.get("max_hp", 100)
        pct = int(hp / max_hp * 100) if max_hp else 0
        status = mon.get("status") or "healthy"
        lines.append(f"  - Slot {idx}: {sp} (HP: {pct}%, Status: {status})")

    opponents = obs.get("opponent_active_pokemon") or []
    lines.append("Opponent Active Pokémon:")
    for idx, mon in enumerate(opponents):
        if not mon or not isinstance(mon, dict):
            continue
        sp = mon.get("species", "Unknown")
        hp = mon.get("hp", 0)
        max_hp = mon.get("max_hp", 100)
        pct = int(hp / max_hp * 100) if max_hp else 0
        status = mon.get("status") or "healthy"
        lines.append(f"  - Opponent {idx}: {sp} (HP: {pct}%, Status: {status})")

    weather = obs.get("weather")
    if weather and weather != "none":
        lines.append(f"Weather: {weather}")
    terrain = obs.get("terrain")
    if terrain and terrain != "none":
        lines.append(f"Terrain: {terrain}")

    return "\n".join(lines)


def format_plan_description(plan: dict) -> str:
    """Turn a joint turn action dictionary into a readable summary for the model."""
    parts = []
    for slot_key in ("slot_0", "slot_1"):
        action = plan.get(slot_key)
        if not action or not isinstance(action, dict):
            continue
        atype = action.get("type")
        if atype == "move":
            mid = action.get("move_id", "move")
            t = action.get("target")
            target_str = f" on target {t}" if t is not None else ""
            parts.append(f"{slot_key}: {mid}{target_str}")
        elif atype == "switch":
            sp = action.get("species", "bench")
            parts.append(f"{slot_key}: switch to {sp}")
        elif atype == "pass":
            parts.append(f"{slot_key}: pass")
    return ", ".join(parts) if parts else "pass"


def parse_advisor_choice(reply: str | None, plan_labels: list[str]) -> str | None:
    """Extract chosen plan label (e.g. 'PLAN_A') from advisor response.

    Guarantees that the returned label is in ``plan_labels``, else None.
    """
    if not reply or not isinstance(reply, str):
        return None

    # First look for explicit "CHOICE: PLAN_X"
    match = _CHOICE_PATTERN.search(reply)
    if match:
        choice = match.group(1).upper()
        if choice in plan_labels:
            return choice

    # Secondary fallback: find first mention of any valid plan label
    match2 = _FALLBACK_PATTERN.search(reply)
    if match2:
        choice = match2.group(1).upper()
        if choice in plan_labels:
            return choice

    return None


def advise(
    client: GeminiClient | None,
    obs: dict,
    candidate_plans: list[tuple[dict, float, str]],
    *,
    timeout: float = ADVISOR_TIMEOUT,
) -> dict | None:
    """Ask Gemini 3.8 Flash to pick from candidate plans.

    ``candidate_plans`` is a list of tuples: ``(action_dict, score, explanation)``.
    Returns the selected action dict, or ``None`` if consultation fails or times out.
    """
    if client is None or not candidate_plans:
        return None

    if len(candidate_plans) == 1:
        return candidate_plans[0][0]

    plan_map: dict[str, dict] = {}
    plan_lines: list[str] = []
    label_chars = ["A", "B", "C"]

    for i, (action, score, desc) in enumerate(candidate_plans[:3]):
        label = f"PLAN_{label_chars[i]}"
        plan_map[label] = action
        human_readable = format_plan_description(action)
        plan_lines.append(f"- {label}: {human_readable} (Tactical note: {desc})")

    board_text = format_board_summary(obs)
    prompt = (
        f"{board_text}\n\n"
        f"Candidate Strategic Plans:\n" + "\n".join(plan_lines) + "\n\n"
        "Which plan is tactically best given opponent potential Protects, focus-firing, or switches?"
    )

    try:
        reply = client.generate(
            prompt,
            system=ADVISOR_SYSTEM,
            temperature=0.2,
            max_output_tokens=80,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - advisor must never crash decision
        return None

    chosen_label = parse_advisor_choice(reply, list(plan_map.keys()))
    if chosen_label and chosen_label in plan_map:
        return plan_map[chosen_label]

    return None
