"""Jev tactician for Pokémon: typed, calibrated decisions on the full game.

Same architecture as Red Alert's ``redalert.tactics``: the deterministic engine
computes the legal, damage-checked candidates, and Jev (TypeSafe System One)
sees the *whole* game — the rules, both boards, the exact damage numbers, the
speed order, both rosters — and answers one bounded ``choice`` question among
those candidates. The free-text Gemini advisors are gone from this game: Jev
answers in ~0.5 s against a 55 s decision clock and a 420 s battle bank, its
confidence rides with every answer, and it answers with a key the engine
handed it — never free text.

Safety, in layers, mirroring Red Alert:

- Jev only ever picks among candidates the engine itself built from the legal
  state; every plan is already a legal action the server will accept.
- Jev decides every turn the decision clock affords a call: its pick is played
  whatever its confidence. The deterministic #1 plays only when Jev is
  unavailable or out of time.
- Only engine strings and numbers go out — species names, move ids, numbers,
  the rules. There is no chat in this game, so there is nothing to inject.
- The client is total: on any failure the deterministic path answers with zero
  latency, so a model call can never cost a match.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from common import trace
from common.jev import Answer, JevClient
from pokemon import battle, dex, draft, observation, threat
from pokemon.damage import damage

# A Pokémon decision happens once per turn, so unlike Red Alert's 5 Hz loop
# there is nothing to background: ask, wait ~0.5 s, decide.
JEV_TIMEOUT = 2.5
# A battle is at most ~25 decisions; the clock budget does the real limiting.
MAX_CALLS_PER_MATCH = 60

LABELS = ("PLAN_A", "PLAN_B", "PLAN_C", "PLAN_D", "PLAN_E")
INDEX = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class Request:
    """One Jev request: the state it reads and the options it may pick from."""

    state: dict
    criteria: dict[str, str]
    instructions: str


# -- the rules, as part of the state -----------------------------------------
# Jev is asked to decide *for this game*, so the game's rules ride with every
# request — the same block Red Alert sends. Engine facts only.

DRAFT_RULES = {
    "win": "The draft builds a team of 6; from it only 4 are brought to each battle. The match is won in battle.",
    "snake": "Players alternate single picks from the same 18 fixed prebuilt cards; a card taken is gone for both sides.",
    "item_clause": "Each item may appear only once per team; the server never offers an item already on our roster.",
    "team_over_cards": "The six must cover each other: broad coverage, both Physical and Special attackers, no weakness "
    "shared by several members, and speed for doubles.",
    "denial": "A card left on the table may be exactly what the opponent's roster needs; taking it is worth part of the "
    "value it would have given them.",
}

PREVIEW_RULES = {
    "win": "KO all four of the opponent's Pokémon to win the battle.",
    "bring": "Bring exactly 4 of the 6 and lead 2; reserves switch in later when an active faints.",
    "leads": "Leads act first unless a priority move or Trick Room is in play; a fast lead with Fake Out or Intimidate "
    "shapes the opening.",
    "opponent": "The opponent's full six is visible; their moves are not published and are estimated from their typing.",
}

BATTLE_RULES = {
    "win": "KO all four of the opponent's Pokémon to win the battle.",
    "turns": "Both sides choose simultaneously; faster Pokémon act first unless a priority move is used.",
    "protect": "A protecting Pokémon takes no damage this turn; if a Protect is likely, attacking the other target is "
    "often better.",
    "spread": "Spread moves hit both opposing Pokémon; moves that also hit our partner say so.",
    "fake_out": "Fake Out flinches only on the user's first turn on the field.",
    "damage": "Damage ranges are min-max percent of the defender's max HP from the exact damage formula; a range marked "
    "KO reaches the HP that is left.",
    "clock": "55 seconds per decision and a 420-second bank for the whole battle; an empty bank forfeits.",
}

DRAFT_INSTRUCTIONS = (
    "You are drafting for our side. `our_roster` and `their_roster` are what each side holds; the options are the "
    "three highest-scoring remaining cards with their team math. Which do we take now — the best completion of our "
    "team (coverage, both damage categories, speed) or the card the opponent needs most?"
)

PREVIEW_INSTRUCTIONS = (
    "You command our side at Team Preview: choose which 4 of our 6 to bring and which 2 lead. Their full six is in "
    "`their_roster`. Which lineup wins the opening? Consider how our damage matches into their typing and theirs into "
    "ours, who outspeeds whom, Fake Out pressure, Intimidate, and keeping a fast attacker with a bulky pivot."
)

BATTLE_INSTRUCTIONS = (
    "You command our side for this turn of a Pokémon VGC doubles battle. `our_active` and `their_active` carry HP "
    "percent, boosts, items, abilities and known moves; `speed_order` is who acts first; the damage ranges in the "
    "options are exact percentages of the defender's max HP. Which plan should we commit? Consider finishing a KO, "
    "not overkilling one foe while the other threatens us, a likely Protect on the obvious target, and spreading "
    "damage."
)


# -- board facts -------------------------------------------------------------


def _speed_of(
    active: battle.Active, fld: threat.Field | None = None, *, ours: bool = True
) -> int | None:
    """Effective speed: the stat with speed boosts, Choice Scarf and Tailwind.
    Trick Room does not change the number; it reverses the order (see ``speed_order``)."""
    value = threat.raw_speed(active, fld or threat.Field(), ours=ours)
    return None if value is None else int(value)


def _active_entry(
    slot: int,
    active: battle.Active,
    *,
    side: str,
    fld: threat.Field | None = None,
    risk: float | None = None,
) -> dict:
    card = active.card
    species = card.species if card else str((active.summary or {}).get("species") or "unknown")
    return {
        "side": side,
        "slot": slot,
        "species": species,
        "types": list(card.types) if card else [],
        "hp_percent": round(active.hp_fraction * 100),
        "item": card.item if card else None,
        "ability": card.ability if card else None,
        "boosts": dict(active.boosts or {}),
        "moves": list(card.moves) if card else [],
        "speed": _speed_of(active, fld, ours=side == "us"),
        "status": (active.summary or {}).get("status"),
        "fainted": active.fainted,
        **(
            {"expected_damage_next_turn_percent_of_remaining_hp": round(risk * 100)}
            if risk is not None
            else {}
        ),
    }


def _card_entry(card: dict) -> dict:
    """A draft card as published: species, set, typing and base stats."""
    species = card.get("species")
    stats = dex.base_stats(species) if isinstance(species, str) else {}
    return {
        "species": species,
        "item": card.get("item"),
        "ability": card.get("ability"),
        "types": list(dex.types_of(species)) if isinstance(species, str) else [],
        "moves": [m for m in (card.get("moves") or []) if isinstance(m, str)][:4],
        "base_stat_total": sum(v for v in stats.values() if isinstance(v, int)),
        "speed": stats.get("spe"),
    }


# -- damage facts ------------------------------------------------------------


def _range_note(attacker: battle.Active, defender: battle.Active, move_id: str) -> str | None:
    """The exact damage span of one move into one defender, with the KO verdict."""
    if attacker is None or defender is None or attacker.card is None or defender.card is None:
        return None
    try:
        result = damage(
            attacker.card,
            defender.card,
            move_id,
            attacker_boosts=attacker.boosts,
            defender_boosts=defender.boosts,
        )
    except Exception:  # noqa: BLE001 - a description must never break a decision
        return None
    span = f"{result.min_percent:.0f}-{result.max_percent:.0f}%"
    if result.min_damage >= defender.current_hp:
        return f"{span} of its max HP: guaranteed KO"
    if result.max_damage >= defender.current_hp:
        return f"{span} of its max HP: KO on a high roll"
    return f"{span} of its max HP"


def describe_plan(plan: dict, ours: list[battle.Active], theirs: list[battle.Active]) -> str:
    """Engine facts only: what each slot does and the exact damage it deals."""
    parts: list[str] = []
    for key in ("slot_0", "slot_1"):
        action = plan.get(key)
        if not isinstance(action, dict):
            continue
        index = int(key[-1])
        attacker = ours[index] if index < len(ours) else None
        kind = action.get("type")
        if kind == "move":
            move_id = str(action.get("move_id") or "move")
            if battle.is_spread(move_id):
                notes = []
                for defender in theirs:
                    if defender.usable:
                        note = _range_note(attacker, defender, move_id)
                        if note:
                            notes.append(note)
                detail = f"hits both foes: {' & '.join(notes)}" if notes else "hits both foes"
                if battle.hits_own_ally(move_id):
                    partner = ours[1 - index] if 0 <= 1 - index < len(ours) else None
                    if partner is not None and partner.usable:
                        note = _range_note(attacker, partner, move_id)
                        if note:
                            detail += f"; also hits our partner for {note}"
            else:
                idx = battle.target_index(action.get("target"))
                defender = theirs[idx] if idx is not None and idx < len(theirs) else None
                note = _range_note(attacker, defender, move_id) if defender is not None else None
                if defender is not None and note:
                    detail = f"at {defender.card.species if defender.card else 'foe'}: {note}"
                elif defender is not None:
                    detail = f"at {defender.card.species if defender.card else 'foe'}"
                else:
                    detail = f"at target {action.get('target')}"
            parts.append(f"{key}: {move_id} — {detail}")
        elif kind == "switch":
            parts.append(f"{key}: switch to {action.get('species')} (gives up this turn's attack)")
        elif kind == "pass":
            parts.append(f"{key}: pass")
    return "; ".join(parts) or "no legal actions"


# -- request builders ---------------------------------------------------------


def draft_request(obs: dict, agent_id: str | None, top: list[tuple[str, float, dict]]) -> Request:
    """The draft as Jev sees it: rules, both rosters, and the top cards' math.

    ``top`` is ``draft.top_cards``' output — the swing-scored candidates.
    """
    rosters = observation.rosters(obs)
    our_roster = rosters.get(agent_id or "", [])
    their_roster = [c for key, cards in rosters.items() if key != agent_id for c in cards]

    criteria: dict[str, str] = {}
    for i, (_, score, card) in enumerate(top[: len(LABELS)]):
        ours_add = draft.marginal_value(card, our_roster)
        theirs_add = draft.marginal_value(card, their_roster)
        moves = (
            ", ".join([m for m in (card.get("moves") or []) if isinstance(m, str)][:4])
            or "no moves"
        )
        criteria[LABELS[i]] = (
            f"{card.get('species')} — {card.get('item')} / {card.get('ability')}; moves {moves}; "
            f"adds {ours_add:+.1f} to our team; would add {theirs_add:+.1f} to theirs if left for them "
            f"(denial); swing {score:.1f}"
        )

    state = {
        "phase": "draft",
        "format": "gen9vgc2025regi VGC doubles, snake draft from 18 fixed prebuilt cards, level 50",
        "rules": DRAFT_RULES,
        "pick_number": observation.pick_number(obs),
        "picks_remaining": observation.picks_remaining(obs),
        "our_roster": [_card_entry(c) for c in our_roster],
        "their_roster": [_card_entry(c) for c in their_roster],
        "candidates_detail": [_card_entry(card) for *_, card in top[: len(LABELS)]],
    }
    return Request(state=state, criteria=criteria, instructions=DRAFT_INSTRUCTIONS)


def preview_request(obs: dict, candidate_lineups: list[tuple[dict, float, str]]) -> Request:
    """Team Preview as Jev sees it: both full sixes and the top lineups."""

    def roster_entry(summary: Any) -> dict | None:
        if not isinstance(summary, dict):
            return None
        species = summary.get("species")
        stats = dex.base_stats(species) if isinstance(species, str) else {}
        spe = stats.get("spe")
        item = summary.get("item")
        # A Choice Scarf lead acts at 1.5x speed; that changes who leads.
        if item == "Choice Scarf" and isinstance(spe, int):
            spe = int(spe * 1.5)
        return {
            "species": species,
            "item": item,
            "ability": summary.get("ability"),
            "types": list(dex.types_of(species)) if isinstance(species, str) else [],
            "moves": [m for m in (summary.get("moves") or []) if isinstance(m, str)][:4],
            "speed": spe,
        }

    our_roster = [e for e in (roster_entry(s) for s in observation.your_roster(obs)) if e]
    their_roster = [e for e in (roster_entry(s) for s in observation.opponent_roster(obs)) if e]

    criteria = {
        LABELS[i]: f"{desc} (engine matchup score {score:.1f})"
        for i, (_, score, desc) in enumerate(candidate_lineups[: len(LABELS)])
    }
    state = {
        "phase": "team_preview",
        "format": "gen9vgc2025regi VGC doubles, level 50, no Terastallization",
        "rules": PREVIEW_RULES,
        "our_roster": our_roster,
        "their_roster": their_roster,
        "clock": observation.clock(obs),
    }
    return Request(state=state, criteria=criteria, instructions=PREVIEW_INSTRUCTIONS)


def battle_request(
    obs: dict,
    candidate_plans: list[tuple[dict, float, str]],
    ours: list[battle.Active],
    theirs: list[battle.Active],
    ctx: battle.Ctx | None = None,
) -> Request:
    """The board as Jev sees it: both actives in full, the speed order, the field,
    the damage coming back at us, and each candidate plan annotated with its
    exact damage ranges."""
    fld = ctx.field if ctx else threat.field_of(obs)
    danger = ctx.danger if ctx else threat.danger_of(ours, theirs, fld)
    our_entries = [
        _active_entry(i, a, side="us", fld=fld, risk=danger.risk_of(i) if a.usable else None)
        for i, a in enumerate(ours)
    ]
    their_entries = [_active_entry(i, a, side="them", fld=fld) for i, a in enumerate(theirs)]

    paced = [
        (entry["speed"], entry)
        for entry in our_entries + their_entries
        if entry["speed"] is not None and not entry["fainted"]
    ]
    paced.sort(key=lambda pair: pair[0] if fld.trick_room else -pair[0])
    speed_order = [f"{entry['species']} ({entry['side']}, {speed})" for speed, entry in paced]

    criteria = {
        LABELS[i]: f"{describe_plan(plan, ours, theirs)} (engine score {score:.1f})"
        for i, (plan, score, _) in enumerate(candidate_plans[: len(LABELS)])
    }

    state = {
        "phase": "moving",
        "format": "gen9vgc2025regi VGC doubles, level 50, no Terastallization",
        "rules": BATTLE_RULES,
        "turn": observation.turn(obs),
        "our_active": our_entries,
        "their_active": their_entries,
        "speed_order": speed_order,
        "clock": observation.clock(obs),
    }
    weather = observation.weather(obs)
    if weather:
        state["weather"] = weather
    if fld.trick_room:
        state["trick_room"] = "active: the SLOWEST Pokémon moves first"
    for key, value in (
        ("our_side", obs.get("side_conditions")),
        ("their_side", obs.get("opponent_side_conditions")),
    ):
        if value:
            state[key] = value
    if any(danger.doomed):
        state["doomed_slots"] = [i for i, d in enumerate(danger.doomed) if d]
    terrain = obs.get("terrain")
    if isinstance(terrain, str) and terrain and terrain != "none":
        state["terrain"] = terrain
    return Request(state=state, criteria=criteria, instructions=BATTLE_INSTRUCTIONS)


# -- the fallback turn, when the board is unreadable --------------------------

# Where a free-text model used to be asked to pick from option ids, Jev gets
# one choice question per slot against the same request.

_INSTRUCTIONS_PER_SLOT = "Choose this slot's action for this turn."


def _slot_menu(
    template: Any,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, dict]]] | None:
    """Per-slot Jev criteria and the id → choice table, from the template's own
    options — the only vocabulary the server accepts."""
    slots = template.get("slots") if isinstance(template, dict) else None
    if not isinstance(slots, list) or len(slots) != 2:
        return None
    menus: dict[int, dict[str, str]] = {}
    lookup: dict[int, dict[str, dict]] = {}
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            return None
        number = slot.get("slot", index)
        options = [o for o in (slot.get("options") or []) if isinstance(o, dict)]
        if not options or number not in (0, 1):
            return None
        criteria: dict[str, str] = {}
        table: dict[str, dict] = {}
        for option in options:
            kind = option.get("type")
            if kind == "move":
                move_id = option.get("move_id")
                if not isinstance(move_id, str) or not move_id:
                    continue
                targets = [
                    t
                    for t in (option.get("targets") or [])
                    if isinstance(t, int) and not isinstance(t, bool)
                ]
                if targets:
                    for t in targets:
                        oid = f"move:{move_id}:{t}"
                        table[oid] = {"type": "move", "move_id": move_id, "target": t}
                        label = "both foes" if battle.is_spread(move_id) else f"target {t}"
                        criteria[oid] = f"{move_id} at {label}"
                else:
                    oid = f"move:{move_id}"
                    table[oid] = {"type": "move", "move_id": move_id}
                    criteria[oid] = move_id
            elif kind == "switch":
                species = option.get("species")
                if not isinstance(species, str) or not species:
                    continue
                oid = f"switch:{species}"
                table[oid] = {"type": "switch", "species": species}
                criteria[oid] = f"switch to {species} (gives up this turn's attack)"
            elif kind == "pass":
                table["pass"] = {"type": "pass"}
                criteria["pass"] = "do nothing this turn"
        if not criteria:
            return None
        menus[number] = criteria
        lookup[number] = table
    return menus, lookup


def slots_fallback(template: Any, obs: dict, tactician: Tactician) -> dict | None:
    """A whole turn from Jev when the deterministic evaluator declined: one
    choice per slot, both answered against the same state in one request."""
    parsed = _slot_menu(template)
    if parsed is None or not tactician.enabled:
        return None
    menus, lookup = parsed
    state = {
        "phase": "moving",
        "format": "gen9vgc2025regi VGC doubles, level 50",
        "rules": BATTLE_RULES,
        "note": "the live board could not be read this turn; these are the legal options",
        "turn": observation.turn(obs),
        "clock": observation.clock(obs),
    }
    questions = {
        f"slot_{number}": {
            "type": "choice",
            "instructions": _INSTRUCTIONS_PER_SLOT,
            "criteria": menus[number],
        }
        for number in (0, 1)
    }
    answers = tactician.ask(state=state, questions=questions)
    if answers is None:
        return None
    picks: dict[int, dict] = {}
    for number in (0, 1):
        answer = answers.get(f"slot_{number}")
        if answer is None or answer.type != "choice" or answer.choice not in lookup[number]:
            return None
        picks[number] = lookup[number][answer.choice]
    # The server's two structural rules, re-checked here like the executor does.
    if (
        picks[0].get("type") == "switch"
        and picks[1].get("type") == "switch"
        and picks[0].get("species") == picks[1].get("species")
    ):
        return None
    if all(pick.get("type") == "pass" for pick in picks.values()):
        return None  # two passes are refused unless forced; the baseline handles that
    action: dict[str, Any] = {"type": "doubles_turn"}
    for number, pick in sorted(picks.items()):
        action[f"slot_{number}"] = pick
    return action


# -- the tactician ------------------------------------------------------------


class Tactician:
    """Jev, one instance per match.

    Synchronous by design: a Pokémon decision happens once per turn and Jev
    answers in ~0.5 s, so — unlike Red Alert's 5 Hz executor — there is
    nothing to background. Every entry point is total.
    """

    def __init__(self, client: JevClient | None = None, *, log: Any = None) -> None:
        self.client = client
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None
        self._announced = False

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _note(self, message: str) -> None:
        self._log(f"[jev-pokemon] {message}")

    def ask(
        self, *, state: dict, questions: dict[str, dict[str, Any]], timeout: float = JEV_TIMEOUT
    ) -> dict[str, Answer] | None:
        """One request, every question answered against ``state``; ``None`` on
        any failure (disabled, transport error, no answers)."""
        if self.client is None or not questions:
            return None
        if not self._announced:
            self._announced = True
            self._note(
                "tactician on: Jev reads the full game state; the damage math plays on any failure"
            )
        self.calls += 1
        answers = self.client.ask(state, questions, timeout=timeout)
        if not answers:
            self.errors += 1
            self.last_error = getattr(self.client, "last_error", None)
            if self.errors in (1, 5):
                self._note(f"no answer ({self.last_error}); using the damage math")
            return None
        return answers

    def choose(
        self,
        *,
        state: dict,
        instructions: str,
        criteria: dict[str, str],
        timeout: float = JEV_TIMEOUT,
    ) -> str | None:
        """Jev's pick among ``criteria``'s keys, or ``None`` — play the
        deterministic #1 — when the answer is missing, off-menu or a shrug."""
        if not criteria:
            return None
        answers = self.ask(
            state=state,
            questions={
                "pick": {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
            },
            timeout=timeout,
        )
        if answers is None:
            return None
        answer = answers.get("pick")
        trace.write("jev", instructions=instructions, criteria=criteria, state=state, answer=answer)
        if answer is None or answer.type != "choice" or answer.choice not in criteria:
            self.errors += 1
            return None
        return answer.choice
