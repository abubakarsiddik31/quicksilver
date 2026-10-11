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

from pokemon import dex, observation
from pokemon.damage import Card, damage, parse_card

# -- scoring weights --------------------------------------------------------

DAMAGE_WEIGHT = 1.0          # per percent of the target's max HP
KO_BONUS = 60.0              # removing a Pokémon beats any amount of chip damage
FAKE_OUT_TURN_ONE_BONUS = 50.0
PROTECT_BASE = 12.0
PROTECT_WHEN_LOW = 45.0      # below the threshold, stalling out a turn is good
LOW_HP_FRACTION = 0.35
SWITCH_BASE = 4.0
SWITCH_WHEN_LOW = 30.0
PASS_PENALTY = -1000.0       # taken only when nothing else is legal
# A duplicate switch must score strictly worse than a pass, or the pair tie and
# the move/switch/pass tie-break quietly lets the duplicate through.
DUPLICATE_SWITCH_PENALTY = PASS_PENALTY - 1.0
ALLY_DAMAGE_WEIGHT = 1.0     # damage we deal our own partner counts against us

# Spread moves. `allAdjacentFoes` hits only the opponents; `allAdjacent` also
# hits our own partner, which is why the two are tracked separately.
_SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
_ALLY_HITTING_TARGETS = {"allAdjacent"}
_PROTECT_LIKE = {"protect", "detect", "spikyshield", "banefulbunker", "endure"}


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
    def is_low(self) -> bool:
        return self.hp_fraction < LOW_HP_FRACTION

    @property
    def usable(self) -> bool:
        return self.card is not None and not self.fainted


def card_for_summary(summary: Any, pool_cards: Any) -> Card | None:
    """A card for a board Pokémon: its real set if the pool has it, otherwise
    the assumed spread."""
    if not isinstance(summary, dict):
        return None
    species = summary.get("species")
    if not isinstance(species, str) or not species:
        return None

    exact = _find_in_pool(species, pool_cards)
    raw = exact if exact is not None else {
        "species": species,
        "item": summary.get("item"),
        "ability": summary.get("ability"),
        "level": 50,
        "evs": {},
        "ivs": {},
        "moves": summary.get("moves") if isinstance(summary.get("moves"), list) else [],
    }
    card = parse_card(raw)

    # The observed HP is authoritative — it reveals the real HP investment.
    maximum = summary.get("max_hp")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 and card.stats:
        card = dataclasses.replace(card, stats={**card.stats, "hp": maximum})
    return card


def _find_in_pool(species: str, pool_cards: Any) -> dict | None:
    if not pool_cards:
        return None
    wanted = dex.to_id(species)
    for entry in pool_cards:
        if isinstance(entry, dict) and dex.to_id(entry.get("species")) == wanted:
            return entry
    return None


def board(obs: dict, *, pool_cards: Any = None) -> tuple[list[Active], list[Active]]:
    """``(ours, theirs)`` for the active slots, aligned by slot index."""
    ours = [
        Active(card=card_for_summary(s, pool_cards), summary=s if isinstance(s, dict) else {})
        for s in observation.active_pokemon(obs)
    ]
    theirs = [
        Active(card=card_for_summary(s, pool_cards), summary=s if isinstance(s, dict) else {})
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


def move_score(
    move_id: Any,
    *,
    attacker: Active | None,
    defenders: list[Active],
    targets: list,
    turn: int | None,
    partner: Active | None = None,
) -> tuple[float, int | None]:
    """``(score, target to send)`` for one move option in one slot."""
    if not isinstance(move_id, str) or not move_id:
        return 0.0, None

    if dex.move_category(move_id) == "Status":
        if dex.to_id(move_id) in _PROTECT_LIKE:
            low = attacker is not None and attacker.is_low
            return (PROTECT_WHEN_LOW if low else PROTECT_BASE), None
        return 0.0, None

    if attacker is None or attacker.card is None:
        return 0.0, None

    spread = is_spread(move_id)
    indices: list[int] = []
    for target in targets or []:
        index = target_index(target)
        if spread:
            indices = [0, 1]
            break
        if index is not None and index not in indices:
            indices.append(index)

    best_score = 0.0
    best_target: int | None = None
    for index in indices:
        if index >= len(defenders):
            continue
        defender = defenders[index]
        if not defender.usable or defender.card is None:
            continue
        result = damage(
            attacker.card,
            defender.card,
            move_id,
            attacker_boosts=attacker.boosts,
            defender_boosts=defender.boosts,
        )
        score = result.max_percent * DAMAGE_WEIGHT
        if result.min_damage >= defender.card.max_hp:
            score += KO_BONUS
        elif result.max_damage >= defender.card.max_hp:
            score += KO_BONUS / 2
        if score > best_score:
            best_score = score
            match = next((t for t in targets or [] if target_index(t) == index), None)
            best_target = match

    # A move that also hits our partner is only as good as what it does to them.
    if hits_own_ally(move_id) and partner is not None and partner.usable and partner.card:
        friendly_fire = damage(
            attacker.card,
            partner.card,
            move_id,
            attacker_boosts=attacker.boosts,
            defender_boosts=partner.boosts,
        )
        best_score -= friendly_fire.max_percent * ALLY_DAMAGE_WEIGHT

    if dex.to_id(move_id) == "fakeout" and turn == 1:
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
        )
        return score
    if kind == "switch":
        species = option.get("species")
        if isinstance(species, str) and species in taken_switches:
            return DUPLICATE_SWITCH_PENALTY  # the server rejects two slots taking one
        return SWITCH_WHEN_LOW if (attacker is not None and attacker.is_low) else SWITCH_BASE
    if kind == "pass":
        return PASS_PENALTY
    return 0.0


def _partner(ours: list[Active], slot_index: int) -> Active | None:
    other = 1 - slot_index
    if 0 <= other < len(ours):
        return ours[other]
    return None


# -- choosing a whole turn --------------------------------------------------


def choose_turn(
    template: Any, obs: dict, *, pool_cards: Any = None
) -> dict | None:
    """A filled-in ``doubles_turn`` action for this board.

    ``template`` is the one the server put in the legal action's
    ``input['action']`` — its ``slots`` are the only options that will be
    accepted. Returns ``None`` when the template is unreadable, so the caller
    can fall back to the server's own shape rather than send a guess.
    """
    if not isinstance(template, dict):
        return None
    slots = template.get("slots")
    if not isinstance(slots, list) or not slots:
        return None

    ours, theirs = board(obs, pool_cards=pool_cards)
    turn = observation.turn(obs)

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
        options = [o for o in (slot.get("options") or []) if isinstance(o, dict)]
        if not options:
            continue
        picks[number] = _pick(
            options,
            slot_index=number,
            ours=ours,
            theirs=theirs,
            turn=turn,
            taken_switches=taken_switches,
        )

    if not picks:
        return None

    # Both slots may not pass. If scoring produced that, keep the pass only for
    # the slot that is genuinely forced and give the other a real option.
    if len(picks) == 2 and all(pick.get("type") == "pass" for pick in picks.values()):
        for number, pick in picks.items():
            slot = next(
                (s for s in slots if isinstance(s, dict) and s.get("slot") == number), None
            )
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
    if chosen.get("type") == "move":
        attacker = ours[slot_index] if slot_index < len(ours) else None
        _, target = move_score(
            chosen.get("move_id"),
            attacker=attacker,
            defenders=theirs,
            targets=chosen.get("targets") or [],
            turn=turn,
            partner=_partner(ours, slot_index),
        )
        choice: dict[str, Any] = {"type": "move", "move_id": chosen.get("move_id")}
        legal_targets = chosen.get("targets")
        if isinstance(legal_targets, list) and legal_targets:
            if target in legal_targets:
                choice["target"] = target
            elif legal_targets == [0] and is_spread(chosen.get("move_id")):
                # Spread moves like Dazzling Gleam / Bleakwind Storm targeting all foes
                choice["target"] = 0
            elif target is not None:
                # If a positive target was chosen but not directly in targets (e.g. edge case), pick legal
                choice["target"] = legal_targets[0]
        elif target is not None:
            choice["target"] = target
        return choice
    return _as_choice(chosen)


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


def generate_candidate_plans(
    template: Any,
    obs: dict,
    *,
    pool_cards: Any = None,
    max_candidates: int = 3,
) -> list[tuple[dict, float, str]]:
    """Compute top-k joint turn action candidates ``(plan_dict, score, explanation)``.

    Enforces all VGC Doubles constraints:
    - No duplicate switches across slots.
    - No simultaneous passes unless forced.
    - Resolves exact legal target IDs for all moves.
    """
    if not isinstance(template, dict):
        return []
    slots = template.get("slots")
    if not isinstance(slots, list) or not slots:
        return []

    ours, theirs = board(obs, pool_cards=pool_cards)
    turn = observation.turn(obs)

    if not any(active.usable for active in theirs) or not any(active.usable for active in ours):
        return []

    # Map options per slot
    slot_options: dict[int, list[dict]] = {}
    for pos, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        num = slot.get("slot", pos)
        if num in (0, 1):
            opts = [o for o in (slot.get("options") or []) if isinstance(o, dict)]
            if opts:
                slot_options[num] = opts

    if not slot_options:
        return []

    # If only 1 slot is active (e.g. 1 mon left)
    if len(slot_options) == 1:
        num = next(iter(slot_options))
        opts = slot_options[num]
        scored_opts = []
        for o in opts:
            sc = _option_score(o, slot_index=num, ours=ours, theirs=theirs, turn=turn, taken_switches=set())
            choice_dict = _resolve_single_choice(o, slot_index=num, ours=ours, theirs=theirs, turn=turn)
            action = {"type": "doubles_turn", f"slot_{num}": choice_dict}
            desc = _describe_single_choice(choice_dict, sc)
            scored_opts.append((action, sc, desc))
        scored_opts.sort(key=lambda item: item[1], reverse=True)
        return scored_opts[:max_candidates]

    # Both slots active: generate joint pairs
    s0_opts = slot_options.get(0, [])
    s1_opts = slot_options.get(1, [])
    joint_candidates: list[tuple[dict, float, str]] = []

    for o0 in s0_opts:
        choice0 = _resolve_single_choice(o0, slot_index=0, ours=ours, theirs=theirs, turn=turn)
        taken_switches = set()
        if choice0.get("type") == "switch":
            sp = choice0.get("species")
            if isinstance(sp, str):
                taken_switches.add(sp)

        sc0 = _option_score(o0, slot_index=0, ours=ours, theirs=theirs, turn=turn, taken_switches=set())

        for o1 in s1_opts:
            # Enforce switch exclusivity
            if o1.get("type") == "switch":
                sp1 = o1.get("species")
                if isinstance(sp1, str) and sp1 in taken_switches:
                    continue  # Invalid: both slots cannot switch to the same mon

            choice1 = _resolve_single_choice(o1, slot_index=1, ours=ours, theirs=theirs, turn=turn)

            # Prevent both passing unless forced
            if choice0.get("type") == "pass" and choice1.get("type") == "pass":
                if not (all(o.get("type") == "pass" for o in s0_opts) and all(o.get("type") == "pass" for o in s1_opts)):
                    continue

            sc1 = _option_score(o1, slot_index=1, ours=ours, theirs=theirs, turn=turn, taken_switches=taken_switches)
            joint_score = sc0 + sc1

            # Focus fire synergy: if both hit the same target and achieve combined KO
            synergy_note = ""
            if (
                choice0.get("type") == "move"
                and choice1.get("type") == "move"
                and choice0.get("target") is not None
                and choice0.get("target") == choice1.get("target")
                and choice0.get("target") > 0
            ):
                t_idx = target_index(choice0["target"])
                if t_idx is not None and t_idx < len(theirs) and theirs[t_idx].card:
                    target_hp = theirs[t_idx].card.max_hp
                    joint_score += 15.0  # Focus-fire coordination bonus
                    synergy_note = f"Focus fire on {theirs[t_idx].card.species}"

            explanation = (
                f"{synergy_note}; " if synergy_note else ""
            ) + f"Slot 0: {choice0.get('move_id') or choice0.get('species') or 'pass'}; Slot 1: {choice1.get('move_id') or choice1.get('species') or 'pass'}"

            plan_dict = {
                "type": "doubles_turn",
                "slot_0": choice0,
                "slot_1": choice1,
            }
            joint_candidates.append((plan_dict, joint_score, explanation))

    joint_candidates.sort(key=lambda item: item[1], reverse=True)

    # Filter out duplicate functional action dicts
    unique_candidates: list[tuple[dict, float, str]] = []
    seen_keys: set[str] = set()
    for plan, score, desc in joint_candidates:
        s0_str = f"{plan.get('slot_0', {}).get('type')}:{plan.get('slot_0', {}).get('move_id')}:{plan.get('slot_0', {}).get('species')}:{plan.get('slot_0', {}).get('target')}"
        s1_str = f"{plan.get('slot_1', {}).get('type')}:{plan.get('slot_1', {}).get('move_id')}:{plan.get('slot_1', {}).get('species')}:{plan.get('slot_1', {}).get('target')}"
        key = f"{s0_str}|{s1_str}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidates.append((plan, score, desc))
        if len(unique_candidates) >= max_candidates:
            break

    return unique_candidates


def _resolve_single_choice(
    option: dict,
    *,
    slot_index: int,
    ours: list[Active],
    theirs: list[Active],
    turn: int | None,
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
        )
        choice: dict[str, Any] = {"type": "move", "move_id": option.get("move_id")}
        legal_targets = option.get("targets")
        if isinstance(legal_targets, list) and legal_targets:
            if target in legal_targets:
                choice["target"] = target
            elif legal_targets == [0] and is_spread(option.get("move_id")):
                choice["target"] = 0
            elif target is not None:
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


def is_tactical_dilemma(candidate_plans: list[tuple[dict, float, str]], obs: dict) -> bool:
    """Check if the board represents a critical tactical dilemma warranting LLM review.

    Criteria:
    1. At least 2 distinct viable candidates exist.
    2. The score difference between Plan 1 and Plan 2 is small (score delta < 18.0),
       meaning the mathematical difference is close.
    3. Neither candidate is an obvious Turn-1 Fake Out or blowout (> 50 pt swing).
    4. Clock bank is healthy (> 90.0 seconds).
    """
    if len(candidate_plans) < 2:
        return False

    clock = obs.get("clock") or {}
    bank_left = clock.get("bank_seconds_left")
    if bank_left is not None and bank_left < 90.0:
        return False  # Preserve clock bank under time pressure

    score_1 = candidate_plans[0][1]
    score_2 = candidate_plans[1][1]
    score_delta = score_1 - score_2

    # If Plan 1 is overwhelmingly dominant, take it instantly
    if score_delta > 18.0:
        return False

    return True


# -- LLM Turn Fallback ------------------------------------------------------


def choose_turn_with_model(
    template: Any,
    obs: dict,
    client: Any,
    *,
    timeout: float = 8.0,
) -> dict | None:
    """Ask a model to pick legal actions for both slots when the deterministic
    evaluator declines.

    Returns a valid ``doubles_turn`` dictionary or ``None`` on any failure.
    Enforces that both slots do not switch to the same Pokémon, and that both
    do not pass if an alternative is available.
    """
    if client is None or not isinstance(template, dict):
        return None
    raw_slots = template.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != 2:
        return None

    # Build discrete candidate options per slot
    slot_options: dict[int, list[dict]] = {}
    option_lookup: dict[int, dict[str, dict]] = {}
    allowed_by_slot: dict[int, list[str]] = {}

    for idx, slot_dict in enumerate(raw_slots):
        if not isinstance(slot_dict, dict):
            return None
        s_num = slot_dict.get("slot", idx)
        raw_opts = [o for o in (slot_dict.get("options") or []) if isinstance(o, dict)]
        if not raw_opts:
            return None

        slot_options[s_num] = []
        option_lookup[s_num] = {}
        allowed_by_slot[s_num] = []

        for o in raw_opts:
            o_type = o.get("type")
            if o_type == "move":
                move_id = o.get("move_id")
                targets = o.get("targets") or []
                if targets:
                    for t in targets:
                        opt_id = f"move:{move_id}:{t}"
                        c = {"type": "move", "move_id": move_id, "target": t}
                        slot_options[s_num].append(c)
                        option_lookup[s_num][opt_id] = c
                        allowed_by_slot[s_num].append(opt_id)
                else:
                    opt_id = f"move:{move_id}"
                    c = {"type": "move", "move_id": move_id}
                    slot_options[s_num].append(c)
                    option_lookup[s_num][opt_id] = c
                    allowed_by_slot[s_num].append(opt_id)
            elif o_type == "switch":
                sp = o.get("species")
                opt_id = f"switch:{sp}"
                c = {"type": "switch", "species": sp}
                slot_options[s_num].append(c)
                option_lookup[s_num][opt_id] = c
                allowed_by_slot[s_num].append(opt_id)
            elif o_type == "pass":
                opt_id = "pass"
                c = {"type": "pass"}
                slot_options[s_num].append(c)
                option_lookup[s_num][opt_id] = c
                allowed_by_slot[s_num].append(opt_id)

    # Flatten all legal option ids for the model prompt
    all_allowed = list(dict.fromkeys(allowed_by_slot[0] + allowed_by_slot[1]))

    from common.llm import choose_ids

    prompt_context = (
        f"Format: Pokémon VGC Doubles.\n"
        f"Available options for Slot 0: {', '.join(allowed_by_slot[0])}\n"
        f"Available options for Slot 1: {', '.join(allowed_by_slot[1])}\n"
        f"Choose the best strategic action for both slots."
    )

    chosen_ids = choose_ids(
        client,
        instruction="Select one action for slot_0 and one action for slot_1 from the legal option ids.",
        allowed=all_allowed,
        slots=2,
        context=prompt_context,
        timeout=timeout,
    )
    if not chosen_ids or len(chosen_ids) != 2:
        return None

    # Map back to action dicts, ensuring each picked id is legal for that slot
    p0_id, p1_id = chosen_ids[0], chosen_ids[1]
    choice_0 = option_lookup[0].get(p0_id)
    choice_1 = option_lookup[1].get(p1_id)
    if choice_0 is None or choice_1 is None:
        return None

    # Validation: cannot switch to the same Pokémon
    if (
        choice_0.get("type") == "switch"
        and choice_1.get("type") == "switch"
        and choice_0.get("species") == choice_1.get("species")
    ):
        return None

    return {
        "type": "doubles_turn",
        "slot_0": _as_choice(choice_0),
        "slot_1": _as_choice(choice_1),
    }
