"""The draft: choosing 6 of the 18 shared cards, as a team rather than as cards.

Both players pick from the same 18 fixed, pre-built cards in a snake order —
``A B B A A B B A A B B A``, twelve picks, six each. Who starts is decided by
the pool seed and published as ``first_drafter``.

The strategy the docs give is the one implemented here: *"You are picking a
team, not six individually strong cards. Cover each other's weaknesses, keep
both physical and special attackers, and deny your opponent the card that best
completes their roster."*

So a card is not scored in isolation. The score is the **swing** it creates:

    (what it adds to us)  +  weight × (what it would have added to them)

The second term is denial, and it *adds*: taking a card does not merely fail to
help the opponent, it removes the card from their reach for the rest of the
draft. The weight is below 1 because they might not have taken it anyway, and it
rises when they pick immediately after us — the documented snake order is what
decides that.

**Item Clause** is enforced by the server, not here: cards whose item already
duplicates one on our roster are simply never offered, so an illegal pick
cannot be constructed from the offer list.

Everything here is total and deterministic. A draft decision that raised would
stop the match's worker, and an unclear pick is far better than a lost one.
"""

from __future__ import annotations

from typing import Any

from pokemon import dex, observation

# -- scoring weights --------------------------------------------------------

# Base-stat total is the coarsest quality signal. 600 is a strong VGC set.
QUALITY_PER_BST = 1.0 / 200.0
# Speed matters disproportionately in doubles: the fast Pokemon move first and
# can remove a threat before it acts.
SPEED_WEIGHT = 1.0 / 120.0
# Breadth of super-effective coverage, per newly-covered defending type.
COVERAGE_WEIGHT = 1.2
# A resistance is worth less than a coverage type, but not nothing.
RESIST_WEIGHT = 0.5
# Overlapping weaknesses lose games; penalise repeats.
WEAKNESS_OVERLAP_PENALTY = 1.5
# Having both a physical and a special attacker keeps us un-wallable.
ROLE_BALANCE_BONUS = 2.0
# Denial: how much of the opponent's would-be gain counts towards our own
# score for taking the card away from them. Below 1 because they might not have
# picked it anyway; higher when they pick immediately after us.
DENIAL_WEIGHT_WHEN_THEY_PICK_NEXT = 0.8
DENIAL_WEIGHT_OTHERWISE = 0.4

_DAMAGE_CATEGORIES = ("Physical", "Special")


# -- snake order ------------------------------------------------------------


def pick_owner(pick_number: int, *, we_are_first: bool) -> str:
    """``"us"`` or ``"them"`` for a given pick number."""
    if pick_number <= 1:
        return "us" if we_are_first else "them"
    index = (pick_number - 2) // 2
    owner_is_first = index % 2 == 1
    first = "us" if we_are_first else "them"
    second = "them" if we_are_first else "us"
    return first if owner_is_first else second


def we_are_first(obs: dict, agent_id: str | None) -> bool | None:
    """Whether we hold the first pick. ``None`` when it is not published."""
    first = observation.first_drafter(obs)
    if first is None or agent_id is None:
        return None
    return first == agent_id


def denial_weight(obs: dict, agent_id: str | None) -> float:
    """Higher when the opponent picks immediately after us, because taking a
    card then denies it for longer."""
    number = observation.pick_number(obs)
    first = we_are_first(obs, agent_id)
    if number is None or first is None:
        return DENIAL_WEIGHT_OTHERWISE
    if pick_owner(number + 1, we_are_first=first) == "them":
        return DENIAL_WEIGHT_WHEN_THEY_PICK_NEXT
    return DENIAL_WEIGHT_OTHERWISE


# -- card profiles ----------------------------------------------------------


def card_species(card: dict) -> str:
    value = card.get("species")
    return value if isinstance(value, str) else ""


def card_moves(card: dict) -> tuple[str, ...]:
    value = card.get("moves")
    return tuple(m for m in value if isinstance(m, str)) if isinstance(value, list) else ()


def offensive_types(card: dict) -> set[str]:
    """Every type this card can hit for damage."""
    return {t for t in (dex.move_type(m) for m in card_moves(card) if dex.is_damaging(m)) if t}


def coverage(card: dict) -> set[str]:
    """Defending types this card hits super-effectively."""
    out: set[str] = set()
    for attacking in offensive_types(card):
        for defending in dex.ALL_TYPES:
            if dex.effectiveness(attacking, (defending,)) >= 2.0:
                out.add(defending)
    return out


def resistances(card: dict) -> set[str]:
    """Attacking types this card takes reduced (or no) damage from."""
    types = dex.types_of(card_species(card))
    return {
        attacking
        for attacking in dex.ALL_TYPES
        if types and dex.effectiveness(attacking, types) < 1.0
    }


def weaknesses(card: dict) -> set[str]:
    """Attacking types this card takes super-effective damage from."""
    types = dex.types_of(card_species(card))
    return {
        attacking
        for attacking in dex.ALL_TYPES
        if types and dex.effectiveness(attacking, types) > 1.0
    }


def damage_roles(card: dict) -> set[str]:
    """Which of Physical/Special this card actually attacks with."""
    return {
        category
        for category in (dex.move_category(m) for m in card_moves(card) if dex.is_damaging(m))
        if category in _DAMAGE_CATEGORIES
    }


def card_quality(card: dict) -> float:
    """Intrinsic strength only: bulk and speed.

    Team-level properties — coverage, resistances, role balance — deliberately
    belong to ``team_value`` instead, so they are counted once for the team
    rather than once per card. Folding them in here as well would double-count
    every benefit and make a second copy of a card look as good as the first.
    """
    stats = dex.base_stats(card_species(card))
    bst = sum(v for v in stats.values() if isinstance(v, int))
    speed = stats.get("spe")
    return (
        bst * QUALITY_PER_BST
        + (speed if isinstance(speed, int) and not isinstance(speed, bool) else 0) * SPEED_WEIGHT
    )


# -- team evaluation --------------------------------------------------------


def with_card_sets(roster: list[dict], pool_cards: Any) -> list[dict]:
    """Roster entries arrive as species-only stubs; the team maths wants each
    card's real moves, items and roles. Join on species against the captured
    pool, keeping the stub when the species is unknown."""
    if not pool_cards:
        return list(roster)
    by_species: dict[str, dict] = {}
    for card in pool_cards:
        if not isinstance(card, dict) or not card.get("moves"):
            continue
        species = card_species(card)
        if species:
            by_species.setdefault(species.casefold(), card)
    out = []
    for entry in roster:
        species = card_species(entry) if isinstance(entry, dict) else ""
        out.append(by_species.get(species.casefold(), entry) if species else entry)
    return out


def role_bonus(roster: list[dict]) -> float:
    """Reward a roster that can attack from both sides, so a single
    high-Defence or high-SpDef wall cannot blank the whole team."""
    roles: set[str] = set()
    for card in roster:
        roles |= damage_roles(card)
    return ROLE_BALANCE_BONUS if roles >= set(_DAMAGE_CATEGORIES) else 0.0


def covered_types(roster: list[dict]) -> set[str]:
    """Defending types the team hits super-effectively, counted once."""
    return set().union(*(coverage(card) for card in roster)) if roster else set()


def resisted_types(roster: list[dict]) -> set[str]:
    return set().union(*(resistances(card) for card in roster)) if roster else set()


def shared_weaknesses(roster: list[dict]) -> dict[str, int]:
    """``{attacking type: how many team members are weak to it}``."""
    counts: dict[str, int] = {}
    for card in roster:
        for attacking in weaknesses(card):
            counts[attacking] = counts.get(attacking, 0) + 1
    return counts


def team_value(roster: list[dict]) -> float:
    """The value of a set of cards *as a team*.

    Beyond the sum of the cards' own strength, this rewards a wide offensive
    spread and a broad set of resistances, and penalises a weakness that two or
    more members share — a hole the opponent can aim at.
    """
    if not roster:
        return 0.0
    total = sum(card_quality(card) for card in roster)
    total += len(covered_types(roster)) * COVERAGE_WEIGHT
    total += len(resisted_types(roster)) * RESIST_WEIGHT
    total -= sum(
        (count - 1) * WEAKNESS_OVERLAP_PENALTY
        for count in shared_weaknesses(roster).values()
        if count > 1
    )
    total += role_bonus(roster)
    return total


# -- pool archetype synergies -----------------------------------------------

# Tier-1 S-rank tournament picks in the captured pool
_S_TIER_SPECIES = {
    "Incineroar",
    "Flutter Mane",
    "Urshifu-Rapid-Strike",
    "Rillaboom",
    "Chien-Pao",
    "Landorus-Therian",
    "Great Tusk",
    "Basculegion",
}
# High-priority archetype pivots
_SPEED_CONTROL_SPECIES = {"Talonflame", "Tornadus", "Whimsicott"}
_WEATHER_SETTERS = {"Tyranitar", "Pelipper"}


def archetype_bonus(card: dict, roster: list[dict]) -> float:
    """Extra valuation for high-synergy VGC tournament archetypes."""
    species = card.get("species", "")
    bonus = 0.0

    # S-Tier staples: Intimidate/Fake Out pivot (Incineroar/Rillaboom), top sweepers
    if species in _S_TIER_SPECIES:
        bonus += 3.0

    # Speed control priority: having at least 1 Tailwind setter is crucial in VGC
    has_speed_control = any(c.get("species") in _SPEED_CONTROL_SPECIES for c in roster)
    if species in _SPEED_CONTROL_SPECIES and not has_speed_control:
        bonus += 2.5

    # Weather/Terrain synergy: Baxcalibur pairs with Sand / Ice
    if species == "Baxcalibur" and any(c.get("species") == "Tyranitar" for c in roster):
        bonus += 2.0

    return bonus


def marginal_value(card: dict, roster: list[dict]) -> float:
    """How much this card adds to *this* roster."""
    base_marginal = team_value([*roster, card]) - team_value(roster)
    return base_marginal + archetype_bonus(card, roster)


def score_card(
    card: dict,
    *,
    our_roster: list[dict],
    their_roster: list[dict],
    denial: float = DENIAL_WEIGHT_OTHERWISE,
) -> float:
    """The swing from taking this card.

    Denial adds rather than subtracts: a card the opponent would gain a lot
    from is *more* attractive to us, because taking it both improves our team
    and removes it from theirs. Subtracting would do the opposite — steer us
    away from exactly the cards worth denying.
    """
    ours = marginal_value(card, our_roster)
    theirs = marginal_value(card, their_roster)
    return ours + denial * theirs


def choose_card(
    obs: dict,
    *,
    agent_id: str | None,
    pool_cards: Any = None,
) -> str | None:
    """The ``card_id`` to take, or ``None`` when nothing is on offer.

    Deterministic: equal scores resolve to the lowest ``card_id`` so a replay
    of the same state picks the same card.
    """
    cards = [c for c in observation.pool_of(obs) if isinstance(c.get("card_id"), str)]
    if not cards:
        return None

    rosters = observation.rosters(obs)
    our_roster = with_card_sets(rosters.get(agent_id or "", []), pool_cards)
    theirs = with_card_sets(
        [entry for key, entries in rosters.items() if key != agent_id for entry in entries],
        pool_cards,
    )
    denial = denial_weight(obs, agent_id)

    scored = [
        (score_card(card, our_roster=our_roster, their_roster=theirs, denial=denial), card)
        for card in cards
    ]
    best_score = max(score for score, _ in scored)
    best = min(
        (card for score, card in scored if score == best_score),
        key=lambda card: card["card_id"],
    )
    return best["card_id"]


def choose_from_actions(obs: dict, *, agent_id: str | None, legal_actions: Any, pool_cards: Any = None) -> Any:
    """The legal action matching our chosen card, or ``None`` if unavailable.

    Draft picks arrive as enumerated actions (``draft_pick:<card_id>``), so the
    decision has to be mapped back onto the server's own list rather than
    constructed — anything else is refused as ``INVALID_ACTION``.
    """
    chosen = choose_card(obs, agent_id=agent_id, pool_cards=pool_cards)
    if chosen is None:
        return None
    for action in legal_actions or []:
        action_id = getattr(action, "action_id", None)
        if isinstance(action_id, str) and action_id == f"draft_pick:{chosen}":
            return action
    return None


# -- ranked candidates -------------------------------------------------------


def top_cards(
    obs: dict,
    *,
    agent_id: str | None,
    k: int = 3,
    pool_cards: Any = None,
) -> list[tuple[str, float, dict]]:
    """The ``k`` best cards on offer by swing score, best first.

    ``[(card_id, score, card)]`` — the same math ``choose_card`` runs, kept
    visible so the Jev tactician can offer the top three as options while the
    deterministic #1 stays the fallback. Ties break on the lowest
    ``card_id``, exactly like ``choose_card``.
    """
    cards = [c for c in observation.pool_of(obs) if isinstance(c.get("card_id"), str)]
    if not cards:
        return []
    rosters = observation.rosters(obs)
    our_roster = with_card_sets(rosters.get(agent_id or "", []), pool_cards)
    theirs = with_card_sets(
        [entry for key, entries in rosters.items() if key != agent_id for entry in entries],
        pool_cards,
    )
    denial = denial_weight(obs, agent_id)
    scored = [
        (score_card(card, our_roster=our_roster, their_roster=theirs, denial=denial), card)
        for card in cards
    ]
    scored.sort(key=lambda item: (-item[0], item[1]["card_id"]))
    return [(card["card_id"], score, card) for score, card in scored[:k]]
