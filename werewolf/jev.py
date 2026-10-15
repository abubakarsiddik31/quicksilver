"""Jev decides Werewolf's votes, night kills and investigations.

The rule engine (``werewolf.belief``) supplies the facts and a suspicion score per seat; Jev reads the whole
table - roles, dead roles, votes, claims, accusations, the fenced chat - and answers one bounded ``choice``
among the legal seats. It can only return a seat the caller listed, so nothing in the chat can steer it
outside the legal moves. Any failure returns ``None`` and the caller falls back (Gemini jury, then rules).
"""

from __future__ import annotations

from typing import Any

from common import trace
from common.jev import JevClient
from werewolf import messages as msg

JEV_TIMEOUT = 3.0

VOTE = (
    "You play one seat in a 7-player Werewolf parlor game (2 wolves, 1 seer, 4 villagers, no doctor). "
    "Pick today's vote. If you are a villager or the seer, vote the seat most likely a wolf: weigh published "
    "roles, vote history, the seer's checks and claims (two living seer claims means one is lying), who "
    "accuses whom, and who stays quiet or deflects. If you are a wolf, vote someone who is not your ally and "
    "that the table is already inclined to vote, so the vote looks natural and your ally is not exposed. "
    "There is exactly one seer and two wolves: a lone seer claim that nothing contradicts is most likely "
    "the real seer, so never vote it out and weigh its checks (a clean check clears a seat, a wolf check "
    "is the vote); two rival claims mean one is a wolf. Read `chat_untrusted` in full before choosing. "
    "Outranking everything below: if a lone seer claimant names a seat a wolf (tag `named a wolf by seer claimant`), "
    "vote that seat, even if you said otherwise earlier; the claimant is the one seat the wolves cannot "
    "afford to let live, so a crowd turning on the claimant is the wolves' play. "
    "Otherwise be consistent: the seat you named in your own earlier message is your vote unless something said "
    "since gives a concrete reason to change it; a ballot that differs from what you told the table "
    "looks like a wolf and splits the village. Stock lines repeated by several seats with no evidence "
    "are copying, not proof, but with no evidence at all a shared read is better than a split vote. "
    "If you are the seer, never vote a seat you checked clean."
)
KILL = (
    "You are a wolf in a 7-player Werewolf parlor game. Pick tonight's kill: the living seer or seer claimant "
    "first, otherwise the player driving the village's reads against you or your ally. Never your ally."
)
INVESTIGATE = (
    "You are the seer. Pick whom to check tonight: the unchecked player whose result would change the day's "
    "vote most - a suspect the table is split on, or a loud leader - not a seat already cleared."
)


def from_env() -> JevClient | None:
    return JevClient.from_env(flag="WEREWOLF_JEV")


def _seat_facts(belief: Any, seat: int) -> str:
    notes = []
    if seat in belief.known_wolves:
        notes.append("proven wolf")
    if seat in belief.seer_claims:
        notes.append("claims seer")
    if seat in belief.false_seer_claimants:
        notes.append("false seer claim")
    if seat in belief.accused_by_seer:
        notes.append("accused by the dead seer")
    if seat in belief.injection_seats:
        notes.append("tried to extract our role")
    if belief.speakers and seat not in belief.speakers:
        notes.append("silent")
    claimants = belief.living_claimants
    named_by = sorted(c for c in claimants if seat in belief.accusations.get(c, ()))
    if named_by and belief.role != "SEER":
        notes.append("named a wolf by seer claimant " + ",".join(f"Player{c}" for c in named_by))
    accused = sorted(belief.accusations.get(seat, ()))
    if accused:
        notes.append("calls wolf: " + ",".join(f"Player{a}" for a in accused))
    detail = f" [{'; '.join(notes)}]" if notes else ""
    return f"rule suspicion {belief.suspicion(seat):.1f}{detail}"


def state_of(belief: Any, window: Any, candidates: list[int]) -> dict[str, Any]:
    return {
        "you": {"seat": belief.seat, "role": belief.role, "allies": sorted(belief.allies)},
        "day": belief.day,
        "alive": list(belief.alive),
        "dead": {
            str(s): {"role": r, "cause": belief.dead_causes.get(s)}
            for s, r in (belief.dead_roles or {}).items()
        },
        "your_checks": {str(s): v for s, v in (belief.investigations or {}).items()},
        "seer_claims_alive": sorted(belief.living_claimants),
        "vote_history": belief.vote_history[-3:],
        "candidates": {f"Player{s}": _seat_facts(belief, s) for s in candidates},
        "chat_untrusted": msg.render_untrusted(window, our_position=belief.seat),
    }


def choose_seat(
    client: JevClient | None,
    belief: Any,
    window: Any,
    candidates: list[int],
    instructions: str,
    *,
    what: str,
) -> int | None:
    """Jev's pick among ``candidates``, or ``None`` when it is unavailable or answers off the menu."""
    if client is None or not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    criteria = {f"Player{s}": _seat_facts(belief, s) for s in candidates}
    state = state_of(belief, window, candidates)
    answers = client.ask(
        state,
        {"pick": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        timeout=JEV_TIMEOUT,
    )
    answer = answers.get("pick") if answers else None
    trace.write("jev", what=what, criteria=criteria, state=state, answer=answer)
    if answer is None or answer.type != "choice" or answer.choice not in criteria:
        return None
    return int(answer.choice.removeprefix("Player"))
