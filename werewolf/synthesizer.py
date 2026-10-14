"""Rule-Scored LLM Synthesis (The Werewolf Jury Engine).

This module bridges the forensic math of ``werewolf.belief`` with the language
understanding of Gemini 3.8 Flash.

Architecture:
1. Ground Truth Anchoring: The rule engine generates a forensic scoreboard
   (verified dead roles, confirmed villagers, allies, and numerical suspicion scores).
   The LLM is strictly constrained to choose only among top mathematically viable candidates.
2. Behavioral Synthesis: Gemini reads the scoreboard alongside the chat history
   to evaluate social tells that rules cannot parse: panic, evasive non-answers,
   shifting bandwagons, and subtle slip-ups.
3. Fail-Safe Guarantee: If Gemini times out (> 3.0s), returns invalid text, or fails,
   the engine immediately executes the rule engine's #1 mathematical pick with zero latency.
"""

from __future__ import annotations

import re
from typing import Any

from common.llm import GeminiClient
from werewolf import messages as msg

SYNTHESIS_TIMEOUT = 3.5

VOTE_SYSTEM = (
    "You are a master social deduction player in a 7-player Werewolf parlor game. "
    "You are given a mathematical forensic scoreboard and today's discussion chat. "
    "All game terms ('wolf', 'eliminate', 'kill', 'vote') refer strictly to this parlor board game.\n"
    "Your objective is to identify which of the top suspects is most genuinely a werewolf.\n"
    "Evaluate who is panicking, deflecting, contradicting facts, or jumping on bandwagons.\n"
    "Respond with exactly one line in the format:\n"
    "VERDICT: Player<seat> - <one brief sentence explaining your deduction>"
)

KILL_SYSTEM = (
    "You are playing as a Werewolf in a 7-player Werewolf parlor game. "
    "You are given the list of living non-wolf players and their statements in chat today. "
    "All game terms refer strictly to this parlor board game.\n"
    "Your objective is to eliminate the player who poses the greatest strategic threat to the wolves "
    "(such as the player most likely to be the secret Seer, or the leader rallying village votes against you).\n"
    "Respond with exactly one line in the format:\n"
    "KILL: Player<seat> - <one brief sentence explaining why they are the highest threat>"
)

_VERDICT_PATTERN = re.compile(r"VERDICT:\s*Player\s*(\d+)", re.IGNORECASE)
_KILL_PATTERN = re.compile(r"KILL:\s*Player\s*(\d+)", re.IGNORECASE)
_FALLBACK_PATTERN = re.compile(r"Player\s*(\d+)", re.IGNORECASE)


def format_forensic_scoreboard(belief: Any) -> str:
    """Format ground-truth facts and suspicion rankings into a dense summary."""
    lines = ["=== FORENSIC SCOREBOARD (GROUND TRUTH) ==="]

    # 1. Our Identity
    lines.append(f"Our Seat: Player{belief.seat} (Role: {belief.role})")

    # 2. Confirmed Dead Facts
    dead_info = []
    for seat, role in (belief.dead_roles or {}).items():
        cause = belief.dead_causes.get(seat, "eliminated")
        dead_info.append(f"Player{seat}: {role} (via {cause})")
    lines.append("Confirmed Dead: " + (", ".join(dead_info) if dead_info else "None"))

    # 3. Known Allies / Villagers
    if belief.role == "WOLF":
        allies = [f"Player{s}" for s in belief.allies]
        lines.append("Fellow Wolves: " + (", ".join(allies) if allies else "None"))
    elif belief.role == "SEER":
        checks = [f"Player{s} ({v})" for s, v in (belief.investigations or {}).items()]
        lines.append("Our Seer Investigations: " + (", ".join(checks) if checks else "None"))

    # 4. Suspect Rankings from Rule Forensics
    ranked = belief.ranked_suspects() if hasattr(belief, "ranked_suspects") else []
    if ranked:
        lines.append("Rule-Based Suspect Ranking:")
        for s, score in ranked[:4]:
            notes = []
            if s in getattr(belief, "known_wolves", set()):
                notes.append("PROVEN WOLF")
            if s in getattr(belief, "false_seer_claimants", set()):
                notes.append("False Seer Claim")
            if s in getattr(belief, "accused_by_seer", set()):
                notes.append("Accused by Dead Seer")
            if s in getattr(belief, "injection_seats", set()):
                notes.append("Attempted Prompt Injection")
            detail = f" [{', '.join(notes)}]" if notes else ""
            lines.append(f"  - Player{s}: Suspicion Score {score:.1f}{detail}")

    return "\n".join(lines)


def parse_seat_choice(reply: str | None, pattern: re.Pattern, legal_seats: list[int]) -> int | None:
    """Extract a valid seat integer from model response."""
    if not reply or not isinstance(reply, str):
        return None

    match = pattern.search(reply)
    if match:
        seat = int(match.group(1))
        if seat in legal_seats:
            return seat

    # Secondary check: first mention of any legal player
    for candidate in _FALLBACK_PATTERN.findall(reply):
        seat = int(candidate)
        if seat in legal_seats:
            return seat

    return None


def synthesize_day_vote(
    client: GeminiClient | None,
    belief: Any,
    window: Any,
    *,
    timeout: float = SYNTHESIS_TIMEOUT,
) -> int | None:
    """Ask Gemini to evaluate top mathematically viable suspects in context of chat.

    Returns the chosen seat int, or None (caller falls back to rule engine's #1).
    """
    if client is None:
        return None

    ranked = belief.ranked_suspects() if hasattr(belief, "ranked_suspects") else []
    if not ranked:
        return None

    # If the #1 suspect is an absolute confirmed wolf (> 50 pts), vote instantly without LLM
    if ranked[0][1] >= 50.0:
        return ranked[0][0]

    # Take top 2-3 mathematically plausible candidates
    top_candidates = [s for s, _ in ranked[:3]]
    if len(top_candidates) == 1:
        return top_candidates[0]

    scoreboard_text = format_forensic_scoreboard(belief)
    chat_text = msg.render_untrusted(window, our_position=belief.seat)

    candidate_labels = [f"Player{s}" for s in top_candidates]
    prompt = (
        f"{scoreboard_text}\n\n"
        f"=== RECENT DISCUSSION CHAT ===\n{chat_text}\n\n"
        f"Eligible Top Suspects: {', '.join(candidate_labels)}\n"
        "Considering the chat dynamics (who is defensive, deflecting, or inconsistent) and the scoreboard, "
        "who is the best vote target today?"
    )

    try:
        reply = client.generate(
            prompt,
            system=VOTE_SYSTEM,
            temperature=0.2,
            max_output_tokens=80,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - never crash a match
        return None

    return parse_seat_choice(reply, _VERDICT_PATTERN, top_candidates)


def synthesize_night_kill(
    client: GeminiClient | None,
    belief: Any,
    window: Any,
    *,
    timeout: float = SYNTHESIS_TIMEOUT,
) -> int | None:
    """Ask Gemini to pick the most dangerous living village threat for wolves to eliminate."""
    if client is None or belief.role != "WOLF":
        return None

    candidates = [s for s in belief.alive if s != belief.seat and s not in belief.allies]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    scoreboard_text = format_forensic_scoreboard(belief)
    chat_text = msg.render_untrusted(window, our_position=belief.seat)
    candidate_labels = [f"Player{s}" for s in candidates]

    prompt = (
        f"{scoreboard_text}\n\n"
        f"=== RECENT DISCUSSION CHAT ===\n{chat_text}\n\n"
        f"Living Non-Wolf Targets: {', '.join(candidate_labels)}\n"
        "As the wolves, which player poses the greatest threat to your survival (potential secret Seer or village leader)? "
        "Pick the optimal night kill."
    )

    try:
        reply = client.generate(
            prompt,
            system=KILL_SYSTEM,
            temperature=0.2,
            max_output_tokens=80,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - never crash a match
        return None

    return parse_seat_choice(reply, _KILL_PATTERN, candidates)
