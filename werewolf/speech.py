"""What to say in the discussion window.

Silence is a losing Werewolf strategy: the table learns nothing from you, and a
village that never compares notes cannot find the wolves. But a language model
is not required to say something useful — most of the value is in *reporting
what you know* and *proposing a vote*, which the belief tracker already
computes.

The priorities, in order:

1. **The seer should claim.** The seer's investigations are the only hard
   information in the game, they die with the seer, and the role is published on
   death anyway — so a seer that stays silent is throwing away its single
   advantage. Claim, report every result, and name the vote.
2. **A wolf should blend.** Claim villager, push a target that is not an ally,
   and prefer one the table already suspects so the push looks reasonable.
3. **Everyone else should contribute a read.** Name the top suspect from the
   belief ranking and give the reason, so other villagers have something to
   compare against their own notes.

Everything is capped by the protocol's limits — 50 words per message counted the
way the server counts them, and 5 messages per window — and enforced with
:func:`werewolf.messages.shorten` rather than hoped for.
"""

from __future__ import annotations

from typing import Any

from werewolf import messages as msg

# Say one thing per window. The cap is 5, but a single report plus a proposed
# vote is what a villager can actually act on, and spending the budget slowly
# leaves room to answer what the table says back.
SEER_CLAIM = "seer-claim"
WOLF_COVER = "wolf-cover"
ACCUSE = "accuse"


def seat_label(seat: Any) -> str:
    return f"Player{seat}" if isinstance(seat, int) and not isinstance(seat, bool) else "someone"


def _list_seats(seats: Any) -> str:
    ordered = sorted(s for s in seats or () if isinstance(s, int) and not isinstance(s, bool))
    return ", ".join(seat_label(s) for s in ordered)


def seer_claim_text(belief: Any, vote: int | None) -> str:
    """The seer's report. Says what it found and who to vote for."""
    wolves = sorted(
        seat for seat, verdict in (belief.investigations or {}).items() if verdict == "wolf"
    )
    villagers = sorted(
        seat for seat, verdict in (belief.investigations or {}).items() if verdict == "villager"
    )
    parts = ["I am the seer."]
    if wolves:
        parts.append(f"{_list_seats(wolves)} {'is' if len(wolves) == 1 else 'are'} a wolf.")
    if villagers:
        parts.append(f"{_list_seats(villagers)} checked clean.")
    if not wolves and not villagers:
        parts.append("No checks yet.")
    if vote is not None:
        parts.append(f"Vote {seat_label(vote)}.")
    return msg.shorten(" ".join(parts))


def wolf_cover_text(belief: Any, vote: int | None) -> str:
    """A wolf's cover story: a villager claim and a plausible push."""
    if vote is None:
        return msg.shorten("I am a villager. Nothing solid yet, but I am watching.")
    return msg.shorten(
        f"I am a villager. {seat_label(vote)} is my read — quiet when it matters. "
        f"Vote {seat_label(vote)}."
    )


def accusation_text(belief: Any, vote: int | None) -> str:
    """A villager's read, with the reason where there is one."""
    if vote is None:
        return msg.shorten("Nothing to go on yet. Tell me what you have seen.")
    reason = ""
    if getattr(belief, "injection_seats", None) and vote in belief.injection_seats:
        reason = " They tried to get me to reveal my role."
    elif vote in getattr(belief, "known_wolves", set()):
        reason = " The record already proves it."
    return msg.shorten(
        f"{seat_label(vote)} is my top suspect.{reason} Vote {seat_label(vote)}."
    )


def compose(belief: Any, *, already_sent: Any = None) -> tuple[str, str] | None:
    """The one message to send this window, as ``(kind, text)``, or ``None``.

    ``already_sent`` is the set of kinds already used in this game, so a claim
    is made once rather than every day, and a window we have already spoken in
    yields nothing — which is what tells the caller to terminate.

    ``None`` also means "nothing worth saying", and that is the common answer
    when the board is unreadable: a generic line tells the table nothing and
    only draws attention, so silence is the better move.
    """
    sent = already_sent if isinstance(already_sent, set) else set()
    vote = _best_vote(belief)
    role = getattr(belief, "role", None)

    # No concrete target means no concrete claim. Babbling is worse than quiet.
    if vote is None:
        return None

    if role == "SEER":
        # A claim with no results just paints a target on the seer; one with
        # results is the strongest information in the game.
        if not getattr(belief, "investigations", None):
            return None
        if SEER_CLAIM in sent:
            return None
        kind, text = SEER_CLAIM, seer_claim_text(belief, vote)
    elif role == "WOLF":
        if WOLF_COVER in sent:
            return None
        kind, text = WOLF_COVER, wolf_cover_text(belief, vote)
    else:
        if ACCUSE in sent:
            return None
        kind, text = ACCUSE, accusation_text(belief, vote)

    if not msg.fits(text):
        return None
    return kind, text


def _best_vote(belief: Any) -> int | None:
    try:
        vote = belief.best_vote()
    except Exception:  # noqa: BLE001 - speech must never break a decision
        return None
    return vote if isinstance(vote, int) and not isinstance(vote, bool) else None


def should_terminate(quota: Any) -> bool:
    """Whether to stop talking. Always terminate rather than idle: the window
    only closes when everyone has, and the server terminates for us anyway."""
    return not quota.can_send() if quota is not None else True
