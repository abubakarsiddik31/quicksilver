"""Doubles turn selection: spending the damage numbers.

Each active slot independently picks its best option. A slot's options are moves
(each with legal targets), switches, or a pass; each is scored and the best is
taken, subject to the server's two structural rules — two slots may not switch
into the same Pokémon, and both slots may not pass.

**An honest limitation, stated up front.** The battle observation publishes
``base_stats``, ``current_hp``/``max_hp``, boosts, item and ability, but *not*
either side's EV spread or defensive stats. Cards built here therefore assume 31
IVs, zero EVs and the observed HP. That is inaccurate in absolute terms, and the
error largely cancels:

- For ranking our own moves of the *same* category, the defender's Defence (or
  Sp. Def) multiplies every candidate equally, so it cannot change which wins.
- It *does* matter when comparing a physical move against a special one, and for
  absolute claims like "this is a guaranteed KO" — which is why the KO term is a
  nudge, not a guarantee.

Where our own Pokémon appears in the captured draft pool, its real set is used,
which removes most of the error on the attacking side.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from pokemon import dex, lookahead, observation, threat
from pokemon.damage import Card, parse_card

# -- scoring weights --------------------------------------------------------

DAMAGE_WEIGHT = 1.0  # per percent of the target's max HP
KO_BONUS = 60.0  # removing a Pokémon beats any amount of chip damage
FAKE_OUT_TURN_ONE_BONUS = 50.0
PROTECT_BASE = 12.0
PROTECT_WHEN_LOW = 45.0  # below the threshold, stalling out a turn is good
PROTECT_RISK_WEIGHT = 40.0  # per unit of expected-damage / remaining-HP
PROTECT_DOOMED_BONUS = 10.0  # a faster foe can KO us before we move
PROTECT_REPEAT = -20.0  # Protect fails when used twice in a row
DOOMED_FACTOR = 0.6  # an attack from a Pokémon that may be KO'd first
OVERKILL_PENALTY = 30.0
FOCUS_FIRE_BONUS = 15.0
DOUBLE_FIELD_MOVE_PENALTY = 40.0
DOUBLE_PROTECT_PENALTY = 15.0
SWITCH_RISK_WEIGHT = 35.0
SWITCH_LOW_BONUS = 15.0
REPLACEMENT_WEIGHT = 0.5     # forced replacements: matchup percent -> score
LOW_HP_FRACTION = 0.35
SWITCH_BASE = 4.0
SWITCH_WHEN_LOW = 30.0
PASS_PENALTY = -1000.0  # taken only when nothing else is legal
# A duplicate switch must score strictly worse than a pass, or the pair tie and
# the move/switch/pass tie-break quietly lets the duplicate through.
DUPLICATE_SWITCH_PENALTY = PASS_PENALTY - 1.0
ALLY_DAMAGE_WEIGHT = 1.0  # damage we deal our own partner counts against us

# Spread moves. `allAdjacentFoes` hits only the opponents; `allAdjacent` also
# hits our own partner, which is why the two are tracked separately.
_SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
_ALLY_HITTING_TARGETS = {"allAdjacent"}
# Status moves valuation
_PROTECT_LIKE = threat.PROTECT_LIKE | {"endure"}
_SPEED_CONTROL = {"tailwind", "trickroom"}
_SCREENS = {"reflect", "lightscreen", "auroraveil"}
_SETUP_STAT = {"nastyplot": "spa", "swordsdance": "atk", "calmmind": "spa"}
_STATUS_IMMUNE_TYPES = {
    "thunderwave": {"Ground", "Electric"},
    "willowisp": {"Fire"},
    "leechseed": {"Grass"},
}
_CRIPPLING_STATUS = {"willowisp", "thunderwave", "taunt", "partingshot", "leechseed"}


# -- one Pokémon on the board -----------------------------------------------


@dataclass(frozen=True)
class Active:
    """A board slot: the card we can compute with, plus the live summary.

    Both are needed and they are not the same thing — the card carries stats,
    while the summary carries the *current* boosts, HP and fainted state that
    change every turn.
    """

    card: Card | None
    summary: dict

    @property
    def fainted(self) -> bool:
        return bool(self.summary.get("fainted"))

    @property
    def boosts(self) -> dict:
        value = self.summary.get("boosts")
        return value if isinstance(value, dict) else {}

    @property
    def hp_fraction(self) -> float:
        value = self.summary.get("current_hp_fraction")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(1.0, float(value)))
        current, maximum = self.summary.get("current_hp"), self.summary.get("max_hp")
        if isinstance(current, int) and isinstance(maximum, int) and maximum > 0:
            return max(0.0, min(1.0, current / maximum))
        return 1.0

    @property
    def current_hp(self) -> int:
        cur = self.summary.get("current_hp")
        if isinstance(cur, int) and not isinstance(cur, bool):
            return max(0, cur)
        max_hp = self.card.max_hp if self.card else 100
        return max(1, int(self.hp_fraction * max_hp))

    @property
    def is_low(self) -> bool:
        return self.hp_fraction < LOW_HP_FRACTION

    @property
    def usable(self) -> bool:
        return self.card is not None and not self.fainted


def card_for_summary(summary: Any, pool_cards: Any, side: str | None = None) -> Card | None:
    """A card for a board Pokémon: its real set if the pool has it, otherwise
    the assumed spread."""
    if not isinstance(summary, dict):
        return None
    species = summary.get("species")
    if not isinstance(species, str) or not species:
        return None

    exact = _find_in_pool(species, pool_cards, side)
    raw = (
        exact
        if exact is not None
        else {
            "species": species,
            "item": summary.get("item"),
            "ability": summary.get("ability"),
            "level": 50,
            "evs": {},
            "ivs": {},
            "moves": summary.get("moves") if isinstance(summary.get("moves"), list) else [],
        }
    )
    card = parse_card(raw)

    # The observed HP is authoritative — it reveals the real HP investment.
    maximum = summary.get("max_hp")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 and card.stats:
        card = dataclasses.replace(card, stats={**card.stats, "hp": maximum})
    return card


def _find_in_pool(species: str, pool_cards: Any, side: str | None = None) -> dict | None:
    """The set for a species. When two cards share it (one on each team), the
    one held by ``side`` wins, then an unowned one, then the other side's."""
    if not pool_cards:
        return None
    wanted = dex.to_id(species)
    found = [
        entry
        for entry in pool_cards
        if isinstance(entry, dict) and dex.to_id(entry.get("species")) == wanted
    ]
    if not found:
        return None
    for want in (side, None):
        for entry in found:
            if entry.get("_owner") == want:
                return entry
    return found[0]


def board(obs: dict, *, pool_cards: Any = None) -> tuple[list[Active], list[Active]]:
    """``(ours, theirs)`` for the active slots, aligned by slot index."""
    ours = [
        Active(
            card=card_for_summary(s, pool_cards, "ours"), summary=s if isinstance(s, dict) else {}
        )
        for s in observation.active_pokemon(obs)
    ]
    theirs = [
        Active(
            card=card_for_summary(s, pool_cards, "theirs"), summary=s if isinstance(s, dict) else {}
        )
        for s in observation.opponent_active_pokemon(obs)
    ]
    return ours, theirs


# -- move scoring -----------------------------------------------------------


def is_spread(move_id: Any) -> bool:
    entry = dex.move(move_id) or {}
    return entry.get("target") in _SPREAD_TARGETS


def target_index(target: Any) -> int | None:
    """A template target mapped to an opposing slot: ``1`` → their slot 0,
    ``2`` → their slot 1. ``0`` and negatives are not opposing slots."""
    if isinstance(target, int) and not isinstance(target, bool) and target in (1, 2):
        return target - 1
    return None


def hits_own_ally(move_id: Any) -> bool:
    """Whether a spread move also damages our partner (Earthquake, Surf)."""
    entry = dex.move(move_id) or {}
    return entry.get("target") in _ALLY_HITTING_TARGETS


# -- the board, once per decision ---------------------------------------------


@dataclass(frozen=True)
class Ctx:
    """Everything a score needs beyond one slot's own damage: the field, the
    danger to each of our slots, and what we did last turn."""

    ours: list[Active]
    theirs: list[Active]
    turn: int | None = None
    field: threat.Field = threat.Field()
    danger: threat.Danger = threat.Danger()
    memory: threat.Memory | None = None
    pool_cards: Any = None

    def replacement_value(self, species: Any) -> float:
        """How good a reserve is as a forced replacement: the damage it deals the
        foes on the board minus the damage they deal it (percent of max HP, averaged)."""
        card = card_for_summary({"species": species}, self.pool_cards, "ours") if isinstance(species, str) else None
        foes = [f for f in self.theirs if f.usable]
        if card is None or not card.stats or not foes:
            return 0.0
        mine = Active(card=card, summary={"species": species})
        offence = sum(threat.best_hit(mine, f, self.field, defender_is_ours=False) for f in foes)
        defence = sum(threat.best_hit(f, mine, self.field, defender_is_ours=True) for f in foes)
        return REPLACEMENT_WEIGHT * (offence - defence) / len(foes)

    def switch_resilience(self, species: Any) -> float:
        """How well a benched Pokémon takes the foes' best hits, in ``[0, 1]``
        (0.5 when we cannot read its set)."""
        if not isinstance(species, str):
            return 0.5
        card = card_for_summary({"species": species}, self.pool_cards, "ours")
        if card is None or not card.stats:
            return 0.5
        incoming = max(
            (
                threat.best_hit(
                    foe, Active(card=card, summary={}), self.field, defender_is_ours=True
                )
                for foe in self.theirs
                if foe.usable
            ),
            default=0.0,
        )
        return 1.0 - min(1.0, incoming / 100.0)


def make_ctx(obs: dict, *, pool_cards: Any = None, memory: threat.Memory | None = None) -> Ctx:
    ours, theirs = board(obs, pool_cards=pool_cards)
    fld = threat.field_of(obs)
    return Ctx(
        ours=ours,
        theirs=theirs,
        turn=observation.turn(obs),
        field=fld,
        danger=threat.danger_of(ours, theirs, fld),
        memory=memory,
        pool_cards=pool_cards,
    )


def _species_of(active: Active | None) -> str | None:
    value = active.summary.get("species") if active is not None else None
    return value if isinstance(value, str) else None


# -- status moves ---------------------------------------------------------------


def _status_score(
    move_id: str,
    *,
    attacker: Active | None,
    defenders: list[Active],
    targets: list,
    turn: int | None,
    ctx: Ctx | None,
    slot_index: int | None,
) -> tuple[float, int | None]:
    mid = dex.to_id(move_id)
    fld = ctx.field if ctx else threat.Field()
    risk = ctx.danger.risk_of(slot_index) if ctx else 0.0
    doomed = ctx.danger.doomed_of(slot_index) if ctx else False
    low = attacker is not None and attacker.is_low
    early = turn in (1, 2)

    if mid in _PROTECT_LIKE:
        # Protect fails when used twice in a row.
        if ctx and ctx.memory and ctx.memory.protected_last(_species_of(attacker)):
            return PROTECT_REPEAT, None
        score = max(
            PROTECT_WHEN_LOW if low else PROTECT_BASE, PROTECT_BASE + PROTECT_RISK_WEIGHT * risk
        )
        return score + (PROTECT_DOOMED_BONUS if doomed else 0.0), None

    if mid == "tailwind":
        if fld.tailwind_us:
            return 0.0, None
        score = 42.0 if early else 20.0
        return (score * 0.3 if fld.trick_room else score), None

    if mid == "trickroom":
        us = [
            s
            for a in (ctx.ours if ctx else [])
            if a.usable and (s := threat.raw_speed(a, fld, ours=True)) is not None
        ]
        them = [
            s
            for d in defenders
            if d.usable and (s := threat.raw_speed(d, fld, ours=False)) is not None
        ]
        if not us or not them:
            return (0.0 if fld.trick_room else 20.0), None
        slower = sum(us) / len(us) < sum(them) / len(them)
        if fld.trick_room:
            # Using it again ends it: only worth it when we are the faster side.
            return (0.0 if slower else 30.0), None
        return ((42.0 if early else 30.0) if slower else 4.0), None

    if mid in _SCREENS:
        if fld.screen_up(mid) or (mid == "auroraveil" and fld.reflect_us and fld.light_screen_us):
            return 0.0, None
        return (32.0 if early else 15.0), None

    if mid in _SETUP_STAT:
        stage = attacker.boosts.get(_SETUP_STAT[mid], 0) if attacker is not None else 0
        stage = stage if isinstance(stage, int) else 0
        if stage >= 2:
            return 1.0, None
        if low or risk >= 0.6 or doomed:
            return 8.0, None
        return (35.0 if stage <= 0 else 15.0), None

    if mid in _CRIPPLING_STATUS:
        return _crippling_score(mid, defenders, targets, ctx)

    return 5.0, None


def _crippling_score(
    mid: str, defenders: list[Active], targets: list, ctx: Ctx | None
) -> tuple[float, int | None]:
    if mid == "partingshot":
        valid = [t for t in targets or [] if isinstance(t, int) and t > 0]
        return 28.0, (valid[0] if valid else None)
    best: tuple[float, int | None] = (0.0, None)
    for target in targets or []:
        index = target_index(target)
        if index is None or index >= len(defenders):
            continue
        foe = defenders[index]
        if not foe.usable or foe.card is None:
            continue
        status = foe.summary.get("status")
        if mid in ("thunderwave", "willowisp", "leechseed") and status and mid != "leechseed":
            continue  # a Pokémon holds one major status
        if set(foe.card.types) & _STATUS_IMMUNE_TYPES.get(mid, set()):
            continue
        score = 28.0
        if mid == "willowisp":
            physical = any(
                dex.move_category(m) == "Physical" and dex.is_damaging(m) for m in foe.card.moves
            )
            score = 28.0 if physical else 6.0
        if ctx is not None:
            # Prefer crippling the foe that hurts us most.
            score += 0.1 * max(
                (
                    threat.best_hit(foe, mine, ctx.field, defender_is_ours=True)
                    for mine in ctx.ours
                    if mine.usable
                ),
                default=0.0,
            )
        if score > best[0]:
            best = (score, target)
    return best


# -- move scoring (continued) -------------------------------------------------


def move_score(
    move_id: Any,
    *,
    attacker: Active | None,
    defenders: list[Active],
    targets: list,
    turn: int | None,
    partner: Active | None = None,
    ctx: Ctx | None = None,
    slot_index: int | None = None,
) -> tuple[float, int | None]:
    """``(score, target to send)`` for one move option in one slot."""
    if not isinstance(move_id, str) or not move_id:
        return 0.0, None

    if dex.move_category(move_id) == "Status":
        return _status_score(
            move_id,
            attacker=attacker,
            defenders=defenders,
            targets=targets,
            turn=turn,
            ctx=ctx,
            slot_index=slot_index,
        )

    if attacker is None or attacker.card is None:
        return 0.0, None

    fld = ctx.field if ctx else threat.Field()
    accuracy = dex.move_accuracy(move_id)
    spread = is_spread(move_id)
    live_foes = sum(1 for d in defenders if d.usable)
    indices: list[int] = []
    for target in targets or []:
        index = target_index(target)
        if spread:
            indices = [0, 1]
            break
        if index is not None and index not in indices:
            indices.append(index)

    per_target: list[tuple[float, int]] = []
    any_hittable = False
    for index in indices:
        if index >= len(defenders):
            continue
        defender = defenders[index]
        if not defender.usable or defender.card is None:
            continue
        result = threat.calc(
            attacker,
            defender,
            move_id,
            fld,
            defender_is_ours=False,
            spread=spread and live_foes > 1,
        )
        if result is None:
            continue
        any_hittable = any_hittable or not result.immune
        score = result.max_percent * DAMAGE_WEIGHT * accuracy
        # True KO check against current remaining HP (finishing off wounded foes)
        rem_hp = defender.current_hp
        if result.min_damage >= rem_hp:
            score += KO_BONUS * accuracy
        elif result.max_damage >= rem_hp:
            score += KO_BONUS / 2 * accuracy
        per_target.append((score, index))

    best_score = 0.0
    best_target: int | None = None
    if per_target:
        top_score, top_index = max(per_target, key=lambda pair: pair[0])
        # A spread move lands on every foe it can reach; a single-target move on one.
        best_score = sum(s for s, _ in per_target) if spread else top_score
        if top_score > 0 or best_score > 0:
            best_target = next((t for t in targets or [] if target_index(t) == top_index), None)

    # An attack that never happens is worth little: a faster foe can KO us first.
    # Priority moves still go.
    if ctx is not None and ctx.danger.doomed_of(slot_index) and dex.move_priority(move_id) <= 0:
        best_score *= DOOMED_FACTOR

    # A move that also hits our partner is only as good as what it does to them.
    if hits_own_ally(move_id) and partner is not None and partner.usable and partner.card:
        friendly_fire = threat.calc(
            attacker, partner, move_id, fld, defender_is_ours=True, spread=spread and live_foes > 1
        )
        if friendly_fire is not None:
            best_score -= friendly_fire.max_percent * ALLY_DAMAGE_WEIGHT

    if dex.to_id(move_id) == "fakeout" and any_hittable:
        species = _species_of(attacker)
        fresh = ctx.memory.fake_out_ok(species, turn) if (ctx and ctx.memory) else turn == 1
        # Check if any opposing active has Armor Tail (Farigiraf) or Dazzling which blocks priority
        blocks_priority = any(
            d.usable and d.card and d.card.ability in ("Armor Tail", "Dazzling", "Queenly Majesty")
            for d in defenders
        )
        if fresh and not blocks_priority:
            best_score += FAKE_OUT_TURN_ONE_BONUS
    return best_score, best_target


def _option_score(
    option: dict,
    *,
    slot_index: int,
    ours: list[Active],
    theirs: list[Active],
    turn: int | None,
    taken_switches: set[str],
    ctx: Ctx | None = None,
) -> float:
    kind = option.get("type")
    attacker = ours[slot_index] if slot_index < len(ours) else None
    partner = _partner(ours, slot_index)
    if kind == "move":
        score, _ = move_score(
            option.get("move_id"),
            attacker=attacker,
            defenders=theirs,
            targets=option.get("targets") or [],
            turn=turn,
            partner=partner,
            ctx=ctx,
            slot_index=slot_index,
        )
        return score
    if kind == "switch":
        species = option.get("species")
        if isinstance(species, str) and species in taken_switches:
            return DUPLICATE_SWITCH_PENALTY  # the server rejects two slots taking one
        if ctx is None:
            return SWITCH_WHEN_LOW if (attacker is not None and attacker.is_low) else SWITCH_BASE
        if attacker is None or not attacker.usable:
            # A forced replacement: nothing is being given up, so pick the best matchup.
            return SWITCH_BASE + ctx.replacement_value(species)
        # Worth switching when we are about to be hit hard and the Pokémon coming
        # in takes that hit well; pointless to rotate a healthy Pokémon in no danger.
        urgency = SWITCH_RISK_WEIGHT * ctx.danger.risk_of(slot_index)
        if attacker is not None and attacker.is_low:
            urgency += SWITCH_LOW_BONUS
        return SWITCH_BASE + urgency * ctx.switch_resilience(species)
    if kind == "pass":
        return PASS_PENALTY
    return 0.0


def _clean_options(slot: dict) -> list[dict]:
    raw = slot.get("options")
    return [clean_option(o) for o in raw if isinstance(o, dict)] if isinstance(raw, list) else []


def clean_option(option: dict) -> dict:
    """An option with ``targets`` guaranteed to be a list of ints (junk becomes ``[]``)."""
    if "targets" not in option:
        return option
    raw = option["targets"]
    targets = [t for t in raw if isinstance(t, int) and not isinstance(t, bool)] if isinstance(raw, list) else []
    return {**option, "targets": targets}


def _partner(ours: list[Active], slot_index: int) -> Active | None:
    other = 1 - slot_index
    if 0 <= other < len(ours):
        return ours[other]
    return None


# -- choosing a whole turn --------------------------------------------------


def choose_turn(
    template: Any, obs: dict, *, pool_cards: Any = None, memory: threat.Memory | None = None
) -> dict | None:
    """A filled-in ``doubles_turn`` action for this board.

    ``template`` is the one the server put in the legal action's
    ``input['action']`` — its ``slots`` are the only options that will be
    accepted. Returns ``None`` when the template is unreadable, so the caller
    can fall back to the server's own shape rather than send a guess.

    The joint plan search is the single scorer (it knows about overkill, focus
    fire and doubled-up status moves); the per-slot greedy pick below only runs
    if it finds nothing.
    """
    if not isinstance(template, dict):
        return None
    slots = template.get("slots")
    if not isinstance(slots, list) or not slots:
        return None

    plans = generate_candidate_plans(
        template, obs, pool_cards=pool_cards, max_candidates=1, memory=memory
    )
    if plans:
        return plans[0][0]

    ctx = make_ctx(obs, pool_cards=pool_cards, memory=memory)
    ours, theirs, turn = ctx.ours, ctx.theirs, ctx.turn

    # Decline when there is nothing to reason about. Without a readable board
    # every score is zero, moves lose their required target, and the choice
    # becomes noise — the caller's simpler baseline is better than that, and a
    # move sent without its target is refused outright.
    if not any(active.usable for active in theirs):
        return None
    if not any(active.usable for active in ours):
        return None

    picks: dict[int, dict] = {}
    taken_switches: set[str] = set()
    for position, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        number = slot.get("slot", position)
        if number not in (0, 1):
            continue
        options = _clean_options(slot)
        if not options:
            continue
        picks[number] = _pick(
            options,
            slot_index=number,
            ours=ours,
            theirs=theirs,
            turn=turn,
            taken_switches=taken_switches,
            ctx=ctx,
        )

    if not picks:
        return None

    # Both slots may not pass. If scoring produced that, keep the pass only for
    # the slot that is genuinely forced and give the other a real option.
    if len(picks) == 2 and all(pick.get("type") == "pass" for pick in picks.values()):
        for number, pick in picks.items():
            slot = next((s for s in slots if isinstance(s, dict) and s.get("slot") == number), None)
            options = [o for o in ((slot or {}).get("options") or []) if isinstance(o, dict)]
            real = [o for o in options if o.get("type") != "pass"]
            if real:
                picks[number] = _as_choice(real[0])
                break

    action: dict[str, Any] = {"type": "doubles_turn"}
    for number, pick in sorted(picks.items()):
        action[f"slot_{number}"] = pick
    return action


def _pick(
    options: list[dict],
    *,
    slot_index: int,
    ours: list[Active],
    theirs: list[Active],
    turn: int | None,
    taken_switches: set[str],
    ctx: Ctx | None = None,
) -> dict:
    scored = [
        (
            _option_score(
                option,
                slot_index=slot_index,
                ours=ours,
                theirs=theirs,
                turn=turn,
                taken_switches=taken_switches,
                ctx=ctx,
            ),
            option,
        )
        for option in options
    ]
    best_score = max(score for score, _ in scored)
    # Deterministic tie-break: prefer a move, then a switch, then a pass, then
    # by name — so the same board always yields the same turn.
    ordered = sorted(
        (option for score, option in scored if score == best_score),
        key=lambda o: (
            {"move": 0, "switch": 1, "pass": 2}.get(str(o.get("type")), 3),
            str(o.get("move_id") or o.get("species") or ""),
        ),
    )
    chosen = ordered[0]

    if chosen.get("type") == "switch" and isinstance(chosen.get("species"), str):
        taken_switches.add(chosen["species"])
    return _resolve_single_choice(
        chosen, slot_index=slot_index, ours=ours, theirs=theirs, turn=turn, ctx=ctx
    )


def _as_choice(option: dict) -> dict:
    if option.get("type") == "switch" and isinstance(option.get("species"), str):
        return {"type": "switch", "species": option["species"]}
    if option.get("type") == "move" and isinstance(option.get("move_id"), str):
        choice: dict[str, Any] = {"type": "move", "move_id": option["move_id"]}
        if "target" in option:
            choice["target"] = option["target"]
        return choice
    return {"type": "pass"}


# -- candidate plans generator & criticality check -------------------------

_FIELD_MOVES = {"tailwind", "trickroom", "reflect", "lightscreen", "auroraveil"}
_TYPE_RANK = {"move": 0, "switch": 1, "pass": 2}


def generate_candidate_plans(
    template: Any,
    obs: dict,
    *,
    pool_cards: Any = None,
    max_candidates: int = 3,
    memory: threat.Memory | None = None,
    diverse: bool = False,
) -> list[tuple[dict, float, str]]:
    """Compute top-k joint turn action candidates ``(plan_dict, score, explanation)``.

    Enforces all VGC Doubles constraints:
    - No duplicate switches across slots.
    - No simultaneous passes unless forced.
    - Resolves exact legal target IDs for all moves.

    With ``diverse``, the best plan using Protect and the best plan using a
    switch are appended when the top-k left them out, so a reader (Jev) can
    weigh defence against the attack the engine's score prefers.
    """
    if not isinstance(template, dict):
        return []
    slots = template.get("slots")
    if not isinstance(slots, list) or not slots:
        return []

    ctx = make_ctx(obs, pool_cards=pool_cards, memory=memory)
    ours, theirs, turn = ctx.ours, ctx.theirs, ctx.turn

    if not any(active.usable for active in theirs) or not any(active.usable for active in ours):
        return []

    # Map options per slot
    slot_options: dict[int, list[dict]] = {}
    for pos, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        num = slot.get("slot", pos)
        if num in (0, 1):
            opts = _clean_options(slot)
            if opts:
                slot_options[num] = opts

    if not slot_options:
        return []

    def option_score(o: dict, num: int, taken: set[str]) -> float:
        return _option_score(
            o, slot_index=num, ours=ours, theirs=theirs, turn=turn, taken_switches=taken, ctx=ctx
        )

    def resolve(o: dict, num: int) -> dict:
        return _resolve_single_choice(
            o, slot_index=num, ours=ours, theirs=theirs, turn=turn, ctx=ctx
        )

    # If only 1 slot is active (e.g. 1 mon left)
    if len(slot_options) == 1:
        num = next(iter(slot_options))
        scored_opts = []
        for o in slot_options[num]:
            sc = option_score(o, num, set())
            choice_dict = resolve(o, num)
            action = {"type": "doubles_turn", f"slot_{num}": choice_dict}
            scored_opts.append((action, sc, _describe_single_choice(choice_dict, sc)))
        scored_opts.sort(
            key=lambda item: (-item[1], _TYPE_RANK.get(item[0][f"slot_{num}"].get("type"), 3))
        )
        return scored_opts[:max_candidates]

    # Both slots active: generate joint pairs
    s0_opts = slot_options.get(0, [])
    s1_opts = slot_options.get(1, [])
    all_pass = all(o.get("type") == "pass" for o in s0_opts) and all(
        o.get("type") == "pass" for o in s1_opts
    )
    joint_candidates: list[tuple[dict, float, str]] = []
    look = lookahead.Lookahead(ctx)

    for o0 in s0_opts:
        choice0 = resolve(o0, 0)
        taken_switches: set[str] = set()
        if choice0.get("type") == "switch" and isinstance(choice0.get("species"), str):
            taken_switches.add(choice0["species"])

        sc0 = option_score(o0, 0, set())

        for o1 in s1_opts:
            # Enforce switch exclusivity
            if o1.get("type") == "switch":
                sp1 = o1.get("species")
                if isinstance(sp1, str) and sp1 in taken_switches:
                    continue  # Invalid: both slots cannot switch to the same mon

            choice1 = resolve(o1, 1)

            # Prevent both passing unless forced
            if choice0.get("type") == "pass" and choice1.get("type") == "pass" and not all_pass:
                continue

            sc1 = option_score(o1, 1, taken_switches)
            plan_for_eval = {"type": "doubles_turn", "slot_0": choice0, "slot_1": choice1}
            swing = look.value(plan_for_eval)
            if swing is not None:
                # The resolved turn prices every attack, Protect and switch; only
                # what it cannot see (status, setup, field moves) keeps its heuristic.
                joint_score, note = _joint_adjust(choice0, choice1, ctx, combat=False)
                joint_score += swing + _residual(o0, sc0, not ours[0].usable) + _residual(o1, sc1, len(ours) > 1 and not ours[1].usable)
            else:
                joint_score, note = _joint_adjust(choice0, choice1, ctx)
                joint_score += sc0 + sc1

            explanation = (
                (f"{note}; " if note else "")
                + f"Slot 0: {choice0.get('move_id') or choice0.get('species') or 'pass'}; Slot 1: {choice1.get('move_id') or choice1.get('species') or 'pass'}"
            )

            plan_dict = {
                "type": "doubles_turn",
                "slot_0": choice0,
                "slot_1": choice1,
            }
            joint_candidates.append((plan_dict, joint_score, explanation))

    joint_candidates.sort(
        key=lambda item: (
            -item[1],
            _TYPE_RANK.get(item[0]["slot_0"].get("type"), 3)
            + _TYPE_RANK.get(item[0]["slot_1"].get("type"), 3),
            item[2],
        )
    )

    # Filter out duplicate functional action dicts
    unique_candidates: list[tuple[dict, float, str]] = []
    seen_keys: set[str] = set()
    for plan, score, desc in joint_candidates:
        key = "|".join(
            f"{plan[s].get('type')}:{plan[s].get('move_id')}:{plan[s].get('species')}:{plan[s].get('target')}"
            for s in ("slot_0", "slot_1")
        )
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidates.append((plan, score, desc))

    chosen = unique_candidates[:max_candidates]
    if diverse:
        for wanted in (_uses_protect, _uses_switch):
            if not any(wanted(p) for p, _, _ in chosen):
                extra = next((c for c in unique_candidates[max_candidates:] if wanted(c[0])), None)
                if extra is not None:
                    chosen.append(extra)
    return chosen


def _uses_protect(plan: dict) -> bool:
    return any(
        isinstance(plan.get(s), dict) and dex.to_id(plan[s].get("move_id")) in _PROTECT_LIKE
        for s in ("slot_0", "slot_1")
    )


def _uses_switch(plan: dict) -> bool:
    return any(
        isinstance(plan.get(s), dict) and plan[s].get("type") == "switch"
        for s in ("slot_0", "slot_1")
    )


def _residual(option: dict, heuristic: float, forced: bool = False) -> float:
    """The part of an option's heuristic score the turn lookahead does not price.

    Attacks, Protect and switches are resolved by the lookahead; passes keep their
    penalty; status, setup and field moves keep their heuristic value.
    """
    kind = option.get("type")
    if kind == "pass":
        return heuristic
    if kind == "switch":
        return heuristic if forced else 0.0  # a forced replacement is not on the battlefield to resolve
    move = option.get("move_id")
    if dex.is_damaging(move) or dex.to_id(move) in _PROTECT_LIKE:
        return 0.02 * heuristic  # tie-break only
    return heuristic


def _joint_adjust(choice0: dict, choice1: dict, ctx: Ctx, *, combat: bool = True) -> tuple[float, str]:
    """Score changes that only exist for the pair: focus fire, overkill, and
    two slots doing the same one-shot thing."""
    ours, theirs = ctx.ours, ctx.theirs
    m0, m1 = dex.to_id(choice0.get("move_id")), dex.to_id(choice1.get("move_id"))

    if choice0.get("type") == "move" and choice1.get("type") == "move":
        # Two Trick Rooms cancel out; two Tailwinds or screens are one wasted turn.
        if m0 == m1 and m0 in _FIELD_MOVES:
            return -DOUBLE_FIELD_MOVE_PENALTY, f"Both slots use {m0}"
        if m0 == m1 and m0 in _PROTECT_LIKE:
            # Fine when something of ours is genuinely about to die, a waste otherwise.
            worst = max(ctx.danger.risk, default=0.0)
            return -DOUBLE_PROTECT_PENALTY * (1.0 - worst), "Both slots Protect"

        t0, t1 = choice0.get("target"), choice1.get("target")
        if combat and isinstance(t0, int) and t0 > 0 and t0 == t1:
            idx = target_index(t0)
            if idx is not None and idx < len(theirs) and theirs[idx].usable and theirs[idx].card:
                foe = theirs[idx]
                results = [
                    threat.calc(ours[i], foe, choice["move_id"], ctx.field, defender_is_ours=False)
                    for i, choice in ((0, choice0), (1, choice1))
                    if i < len(ours)
                ]
                results = [r for r in results if r is not None and not r.immune]
                others_alive = any(o.usable for i, o in enumerate(theirs) if i != idx)
                if any(r.min_damage >= foe.current_hp for r in results) and others_alive:
                    return (
                        -OVERKILL_PENALTY,
                        f"Overkill on {foe.card.species} (one slot already guarantees the KO)",
                    )
                return FOCUS_FIRE_BONUS, f"Focus fire on {foe.card.species}"
    return 0.0, ""


def _resolve_single_choice(
    option: dict,
    *,
    slot_index: int,
    ours: list[Active],
    theirs: list[Active],
    turn: int | None,
    ctx: Ctx | None = None,
) -> dict:
    """Resolve a raw option dict into a legal, formatted choice dict with target."""
    if option.get("type") == "move":
        attacker = ours[slot_index] if slot_index < len(ours) else None
        _, target = move_score(
            option.get("move_id"),
            attacker=attacker,
            defenders=theirs,
            targets=option.get("targets") or [],
            turn=turn,
            partner=_partner(ours, slot_index),
            ctx=ctx,
            slot_index=slot_index,
        )
        choice: dict[str, Any] = {"type": "move", "move_id": option.get("move_id")}
        legal_targets = option.get("targets")
        if isinstance(legal_targets, list) and legal_targets:
            if target in legal_targets:
                choice["target"] = target
            elif legal_targets == [0] and is_spread(option.get("move_id")):
                # Spread moves like Dazzling Gleam / Bleakwind Storm targeting all foes
                choice["target"] = 0
            elif legal_targets != [0]:
                # Nothing to aim at scored (an immune target, say), but the server still
                # refuses a move without its target: send the first one it offered.
                choice["target"] = legal_targets[0]
        elif target is not None:
            choice["target"] = target
        return choice
    return _as_choice(option)


def _describe_single_choice(choice: dict, score: float) -> str:
    ctype = choice.get("type")
    if ctype == "move":
        return f"Move {choice.get('move_id')} (score {score:.1f})"
    if ctype == "switch":
        return f"Switch to {choice.get('species')}"
    return "Pass"
