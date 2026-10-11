"""Team Preview: bringing 4 of 6 and choosing the two that lead.

After the draft both players see **each other's full six**, so this is the one
decision in the game made with complete information about the opposing team.
That is what makes it worth doing properly: we can pick the four whose matchups
are best against *their* six, and lead the two that punish whatever they are
most likely to open with.

The cards themselves come from two sources, in order of quality:

1. The captured draft pool, keyed by species — this carries the real item,
   ability, nature, moves and EVs.
2. The Team Preview roster block, which publishes species, types and base stats
   but no set. Used as a fallback so this still works before a pool is banked.

Opposing *moves* are not published, so their threat is estimated from their
species' own types — the moves a Pokémon actually carries are almost always
drawn from its typing. That is an approximation, and it is the only one
available.

The returned species ids are taken from the template's own ``roster`` list,
because that is the vocabulary the server accepts: case and punctuation matter
to us but are ignored by the server, and echoing its own strings avoids the
mismatch entirely.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from pokemon import actions, dex, observation
from pokemon.battle import card_for_summary
from pokemon.damage import Card

BRING = 4
LEADS = 2

# How much an opponent's ability to hurt us counts against a matchup.
RISK_WEIGHT = 0.8
# A team of four that shares one attacking type can be walled — but this must
# stay small relative to a matchup, or a pile of irrelevant attack types beats
# actually being good against the team in front of you. At 1.0 per type it did
# exactly that: four Water/Grass Pokémon with ten attack types outscored the
# correct answer against a Fire team.
TYPE_DIVERSITY_BONUS = 0.25
# Both damage roles present.
ROLE_DIVERSITY_BONUS = 2.0
# Speed wins Team Previews: the faster lead moves first.
LEAD_SPEED_WEIGHT = 0.05
# Fake Out on a lead is close to a free turn.
LEAD_FAKE_OUT_BONUS = 6.0


# -- matchup scoring --------------------------------------------------------


def _attack_types(card: Card | None) -> set[str]:
    if card is None:
        return set()
    return {
        t
        for t in (dex.move_type(m) for m in card.moves if dex.is_damaging(m))
        if t
    }


def _stab_types(card: Card | None) -> set[str]:
    return set(card.types) if card is not None else set()


def _best_effectiveness(attacking: set[str], defending: tuple[str, ...]) -> float:
    if not attacking or not defending:
        return 1.0
    return max(dex.effectiveness(t, defending) for t in attacking)


def offensive_pressure(card: Card | None, opponent: Card | None) -> float:
    """How hard this card can hit that opponent, as a multiplier."""
    return _best_effectiveness(_attack_types(card), opponent.types if opponent else ())


def defensive_risk(card: Card | None, opponent: Card | None) -> float:
    """How hard that opponent can hit this card, estimated from its own types."""
    return _best_effectiveness(_stab_types(opponent), card.types if card else ())


def matchup(card: Card | None, opponents: list[Card | None]) -> float:
    """This card's average matchup across the opponents' whole six."""
    real = [o for o in opponents if o is not None]
    if card is None or not real:
        return 0.0
    offence = sum(offensive_pressure(card, o) for o in real) / len(real)
    risk = sum(defensive_risk(card, o) for o in real) / len(real)
    return offence - RISK_WEIGHT * risk


def _diversity(cards: list[Card]) -> float:
    types: set[str] = set()
    for card in cards:
        types |= _attack_types(card)
    bonus = len(types) * TYPE_DIVERSITY_BONUS

    roles: set[str] = set()
    for card in cards:
        for move in card.moves:
            category = dex.move_category(move)
            if dex.is_damaging(move) and category:
                roles.add(category)
    if len(roles) >= 2:
        bonus += ROLE_DIVERSITY_BONUS
    return bonus


def lineup_score(cards: list[Card], opponents: list[Card | None]) -> float:
    return sum(matchup(card, opponents) for card in cards) + _diversity(cards)


def lead_score(card: Card, opponents: list[Card | None]) -> float:
    """How good a lead this card is: matchup, speed, and Fake Out."""
    score = matchup(card, opponents)
    score += card.stats.get("spe", 0) * LEAD_SPEED_WEIGHT
    if any(dex.to_id(move) == "fakeout" for move in card.moves):
        score += LEAD_FAKE_OUT_BONUS
    return score


# -- building the candidate cards -------------------------------------------


def _cards_for_roster(
    roster_ids: list[str],
    summaries_by_id: dict[str, dict],
    pool_cards: Any,
) -> list[tuple[str, Card]]:
    """``[(template species id, card)]`` for each roster entry we can read.

    The template's id is paired with its card so the choice can be expressed in
    the server's own vocabulary.
    """
    out: list[tuple[str, Card]] = []
    for roster_id in roster_ids:
        summary = summaries_by_id.get(dex.to_id(roster_id))
        card = card_for_summary(summary, pool_cards) if summary else None
        if card is None and pool_cards:
            # No summary: fall back to the pool entry for this species.
            card = card_for_summary({"species": roster_id}, pool_cards)
        if card is None:
            card = card_for_summary({"species": roster_id}, None)
        if card is not None and card.species:
            out.append((roster_id, card))
    return out


def _summaries(entries: Any) -> dict[str, dict]:
    """``{normalized species: summary}``."""
    out: dict[str, dict] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        species = entry.get("species")
        if isinstance(species, str) and species:
            out[dex.to_id(species)] = entry
    return out


# -- choosing ---------------------------------------------------------------


def choose_lineup(
    template: Any, obs: dict, *, pool_cards: Any = None
) -> dict | None:
    """A filled-in ``select_lineup`` action, or ``None`` if it can't be read.

    Returns ids copied from the template's roster, which is what the server
    validates against.
    """
    if not isinstance(template, dict):
        return None
    roster = actions.roster_of(template)
    if len(roster) < BRING:
        return None

    ours = _cards_for_roster(roster, _summaries(observation.your_roster(obs)), pool_cards)
    if len(ours) < BRING:
        return None

    theirs: list[Card] = []
    for summary in _summaries(observation.opponent_roster(obs)).values():
        card = card_for_summary(summary, pool_cards)
        if card is not None:
            theirs.append(card)

    # Every legal way to bring BRING of our six. Ties break on the sorted
    # roster ids, so the same preview always yields the same lineup.
    best_combo: tuple[tuple[str, Card], ...] | None = None
    best_score: float | None = None
    best_key: tuple[str, ...] = ()
    for combo in itertools.combinations(ours, BRING):
        score = lineup_score([card for _, card in combo], theirs)
        key = tuple(sorted(roster_id for roster_id, _ in combo))
        if best_score is None or score > best_score or (score == best_score and key < best_key):
            best_score, best_key, best_combo = score, key, combo

    if best_combo is None:
        return None

    chosen = list(best_combo)
    leads = sorted(
        chosen, key=lambda pair: (-lead_score(pair[1], theirs), pair[0])
    )[:LEADS]

    return {
        "type": "select_lineup",
        "bring": [roster_id for roster_id, _ in chosen],
        "leads": [roster_id for roster_id, _ in leads],
    }


# -- multi-candidate lineup generation & LLM team preview advisor ----------


def generate_candidate_lineups(
    template: Any,
    obs: dict,
    *,
    pool_cards: Any = None,
    max_candidates: int = 3,
) -> list[tuple[dict, float, str]]:
    """Compute top-k viable 4-bring and 2-lead lineups ``(lineup_dict, score, explanation)``.

    Enforces all Showdown VGC Team Preview rules:
    - Bring exactly 4 unique Pokemon.
    - Leads are a subset of the 4 bring.
    """
    roster = actions.roster_of(template)
    if len(roster) < BRING:
        return []

    ours = _cards_for_roster(roster, _summaries(observation.your_roster(obs)), pool_cards)
    if len(ours) < BRING:
        return []

    theirs: list[Card] = []
    for summary in _summaries(observation.opponent_roster(obs)).values():
        card = card_for_summary(summary, pool_cards)
        if card is not None:
            theirs.append(card)

    scored_combos: list[tuple[float, tuple[tuple[str, Card], ...]]] = []
    for combo in itertools.combinations(ours, BRING):
        score = lineup_score([card for _, card in combo], theirs)
        scored_combos.append((score, combo))

    scored_combos.sort(key=lambda item: item[0], reverse=True)

    candidates: list[tuple[dict, float, str]] = []
    seen_keys: set[tuple[str, ...]] = set()

    for score, combo in scored_combos:
        chosen = list(combo)
        leads = sorted(
            chosen, key=lambda pair: (-lead_score(pair[1], theirs), pair[0])
        )[:LEADS]

        bring_ids = [roster_id for roster_id, _ in chosen]
        lead_ids = [roster_id for roster_id, _ in leads]
        key = tuple(sorted(bring_ids) + lead_ids)

        if key not in seen_keys:
            seen_keys.add(key)
            lineup_dict = {
                "type": "select_lineup",
                "bring": bring_ids,
                "leads": lead_ids,
            }
            desc = f"Leads: {', '.join(lead_ids)}; Backbench: {', '.join([b for b in bring_ids if b not in lead_ids])}"
            candidates.append((lineup_dict, score, desc))

        if len(candidates) >= max_candidates:
            break

    return candidates


LINEUP_ADVISOR_SYSTEM = (
    "You are a World Champion Pokémon VGC Doubles team preview tactician. "
    "In Pokémon VGC Doubles, selecting the 4 to bring and the 2 opening leads "
    "determines 70% of match outcome. "
    "You will be given our 6-Pokémon roster, the opponent's 6-Pokémon roster, "
    "and the top 3 mathematically evaluated candidate lineups (Plan A, Plan B, Plan C). "
    "Evaluate defensive typing, speed advantage, Fake Out pressure, and weather/trick room counters. "
    "Respond with exactly one line in the format:\n"
    "CHOICE: PLAN_<A|B|C> - <one brief sentence explaining your strategic win condition>"
)

_CHOICE_PATTERN = re.compile(r"CHOICE:\s*(PLAN_[A-C])", re.IGNORECASE)
_FALLBACK_PATTERN = re.compile(r"\b(PLAN_[A-C])\b", re.IGNORECASE)


def advise_lineup(
    client: Any,
    obs: dict,
    candidate_lineups: list[tuple[dict, float, str]],
    *,
    timeout: float = 4.0,
) -> dict | None:
    """Ask Gemini 3.8 Flash to select the optimal 4-bring and 2-leads.

    Takes advantage of the 90-second Team Preview window.
    Returns the chosen lineup dict or None (caller defaults to Plan A).
    """
    if client is None or not candidate_lineups:
        return None
    if len(candidate_lineups) == 1:
        return candidate_lineups[0][0]

    plan_map: dict[str, dict] = {}
    plan_lines: list[str] = []
    labels = ["A", "B", "C"]

    for i, (lineup_dict, score, desc) in enumerate(candidate_lineups[:3]):
        label = f"PLAN_{labels[i]}"
        plan_map[label] = lineup_dict
        plan_lines.append(f"- {label}: {desc} (Matchup Score: {score:.1f})")

    # Format opponent team summary
    opp_summary = []
    for mon in observation.opponent_roster(obs) or []:
        if isinstance(mon, dict):
            sp = mon.get("species", "Unknown")
            item = mon.get("item") or "Item Unknown"
            opp_summary.append(f"{sp} ({item})")

    prompt = (
        f"Opponent's 6-Pokémon Roster:\n"
        f"{', '.join(opp_summary) if opp_summary else 'Unknown'}\n\n"
        f"Candidate Lineups (Bring 4, Lead 2):\n" + "\n".join(plan_lines) + "\n\n"
        "Which lineup establishes the strongest opening board control and covers opponent threats?"
    )

    try:
        reply = client.generate(
            prompt,
            system=LINEUP_ADVISOR_SYSTEM,
            temperature=0.2,
            max_output_tokens=80,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - never crash a decision
        return None

    if not reply or not isinstance(reply, str):
        return None

    m = _CHOICE_PATTERN.search(reply)
    if m and m.group(1).upper() in plan_map:
        return plan_map[m.group(1).upper()]

    m2 = _FALLBACK_PATTERN.search(reply)
    if m2 and m2.group(1).upper() in plan_map:
        return plan_map[m2.group(1).upper()]

    return None
