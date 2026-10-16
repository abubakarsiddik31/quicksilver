"""A one-turn lookahead: play each of our plans out against what the foes are likely to do.

The per-option scores in :mod:`pokemon.battle` see only our own damage plus some
heuristics, so they cannot tell that a Protect eats a Close Combat, that a
slower attacker never gets its move off, or that a switch absorbs a hit meant
for someone else. This resolves the turn instead: priority and speed order,
Protect, Fake Out flinches, spread hits, KOs and redirected hits, using the
same damage calculator, and scores the result as the HP swing.

The foes' side is a distribution, not a guess: each of their slots keeps its two
most attractive options (their best hits on each of ours, Protect), weighted by
how attractive they are. Everything here is expected-value (average roll, times
accuracy) and total: anything unreadable makes ``value`` return ``None`` and the
caller keeps the plain heuristic score.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any

from pokemon import dex, threat

KO_TEMPO = 25.0          # a KO also removes every later action of that Pokémon
FOE_PROTECT_PRIOR = 14.0  # how attractive a foe finds Protect, on the damage-percent scale
FOE_SOFTMAX = 25.0
WORST_CASE_WEIGHT = 0.25
_PRIORITY = {"fakeout": 3, "protect": 4, "detect": 4, "spikyshield": 4, "banefulbunker": 4,
             "kingsshield": 4, "silktrap": 4, "obstruct": 4}
_SETUP = {"nastyplot": {"spa": 2}, "swordsdance": {"atk": 2}, "calmmind": {"spa": 1, "spd": 1}}


@dataclasses.dataclass
class _Unit:
    side: str
    slot: int
    active: Any
    hp: float
    start: float
    boosts: dict
    flinched: bool = False
    protected: bool = False
    alive_at_start: bool = True


class Lookahead:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.ours, self.theirs = ctx.ours, ctx.theirs
        self.fld = ctx.field
        self._hits: dict[tuple, float] = {}
        self._switch_cards: dict[str, Any] = {}
        try:
            self.foe_plans = self._foe_plans()
            readable = True
        except Exception:  # noqa: BLE001 - an unreadable board just means no lookahead
            self.foe_plans, readable = [({}, 1.0)], False
        self.ok = readable and (
            any(a.usable for a in self.ours)
            and all(a.card is not None for a in self.ours if not a.fainted and a.summary)
            and all(a.card is not None for a in self.theirs if not a.fainted and a.summary)
            and any(a.usable for a in self.theirs)
        )

    # -- damage ---------------------------------------------------------------

    def _fraction(self, att: _Unit, dfn: _Unit, move: str, spread: bool) -> float:
        """Expected share of the defender's max HP this hit removes."""
        key = (id(att.active), tuple(sorted(att.boosts.items())), id(dfn.active), tuple(sorted(dfn.boosts.items())),
               move, spread, dfn.hp >= 1.0, att.side)
        if key not in self._hits:
            attacker = dataclasses.replace(att.active, summary={**att.active.summary, "boosts": dict(att.boosts)})
            defender = dataclasses.replace(dfn.active, summary={**dfn.active.summary, "boosts": dict(dfn.boosts)})
            result = threat.calc(attacker, defender, move, self.fld, defender_is_ours=dfn.side == "us", spread=spread)
            if result is None:
                self._hits[key] = 0.0
            else:
                mean = (result.min_damage + result.max_damage) / 2
                maximum = max(1, result.defender_max_hp)
                self._hits[key] = mean / maximum * dex.move_accuracy(move)
        return self._hits[key]

    # -- the foes ---------------------------------------------------------------

    def _foe_options(self, slot: int) -> list[tuple[dict, float]]:
        foe = self.theirs[slot] if slot < len(self.theirs) else None
        if foe is None or not foe.usable:
            return []
        known = [m for m in foe.card.moves if dex.is_damaging(m)]
        moves = known or threat._guessed_moves(foe.card)
        scored: dict[str, tuple[dict, float]] = {}
        me = _Unit("them", slot, foe, foe.hp_fraction, foe.hp_fraction, dict(foe.boosts))
        targets = [
            _Unit("us", i, a, a.hp_fraction, a.hp_fraction, dict(a.boosts))
            for i, a in enumerate(self.ours) if a.usable
        ]
        for move in moves:
            kind = (dex.move(move) or {}).get("target")
            if kind in ("allAdjacentFoes", "allAdjacent"):
                total = sum(self._fraction(me, t, move, len(targets) > 1) * 100 for t in targets)
                scored[move] = ({"type": "move", "move_id": move, "target": 0}, total)
                continue
            for t in targets:
                pct = self._fraction(me, t, move, False) * 100
                if pct >= t.hp * 100 * 0.95:
                    pct += 30.0  # they would love the KO
                if dex.to_id(move) == "fakeout":
                    memory = getattr(self.ctx, "memory", None)
                    fresh = memory.foe_fake_out_ok(foe.summary.get("species"), self.ctx.turn) if memory else self.ctx.turn == 1
                    pct += 25.0 if fresh else -100.0
                key = f"{move}@{t.slot}"
                scored[key] = ({"type": "move", "move_id": move, "target": t.slot + 1}, pct)
        if any(dex.to_id(m) in threat.PROTECT_LIKE for m in foe.card.moves):
            scored["protect"] = ({"type": "move", "move_id": "Protect", "target": 0}, FOE_PROTECT_PRIOR)
        ranked = sorted(scored.values(), key=lambda pair: -pair[1])
        return ranked[:2]

    def _foe_plans(self) -> list[tuple[dict[int, dict], float]]:
        per_slot = [self._foe_options(s) for s in range(len(self.theirs))]
        live = [(s, opts) for s, opts in enumerate(per_slot) if opts]
        if not live:
            return [({}, 1.0)]
        combos = list(itertools.product(*[opts for _, opts in live]))
        means = [sum(score for _, score in combo) / len(combo) for combo in combos]
        top = max(means)
        plans = []
        for combo, mean in zip(combos, means):
            weight = math.exp(max(-50.0, (mean - top) / FOE_SOFTMAX))
            plans.append(({s: action for (s, _), (action, _) in zip(live, combo)}, weight))
        total = sum(w for _, w in plans)
        return [(p, w / total) for p, w in plans]

    # -- resolving a turn ---------------------------------------------------------

    def _unit(self, side: str, slot: int) -> _Unit | None:
        pool = self.ours if side == "us" else self.theirs
        if slot >= len(pool) or not pool[slot].usable:
            return None
        a = pool[slot]
        return _Unit(side, slot, a, a.hp_fraction, a.hp_fraction, dict(a.boosts))

    def _switch_in(self, species: Any) -> Any:
        if species not in self._switch_cards:
            from pokemon import battle  # local: battle imports this module

            card = battle.card_for_summary({"species": species}, self.ctx.pool_cards, "ours")
            self._switch_cards[species] = battle.Active(card=card, summary={"species": species}) if card else None
        return self._switch_cards[species]

    def swing(self, plan: dict, foe_plan: dict[int, dict]) -> float:
        units = {("us", s): self._unit("us", s) for s in (0, 1)}
        units.update({("them", s): self._unit("them", s) for s in (0, 1)})
        history = [u for u in units.values() if u is not None]
        queue = []
        for slot in (0, 1):
            action = plan.get(f"slot_{slot}")
            if isinstance(action, dict) and units[("us", slot)] is not None:
                queue.append(("us", slot, action))
        for slot, action in foe_plan.items():
            if units.get(("them", slot)) is not None:
                queue.append(("them", slot, action))

        def order(item):
            side, slot, action = item
            unit = units[(side, slot)]
            if action.get("type") == "switch":
                return (-10, 0.0)
            move = dex.to_id(action.get("move_id"))
            prio = _PRIORITY.get(move, dex.move_priority(action.get("move_id")))
            speed = threat.raw_speed(unit.active, self.fld, ours=side == "us") or 0.0
            return (-prio, speed if self.fld.trick_room else -speed)

        memory = self.ctx.memory
        for side, slot, action in sorted(queue, key=order):
            unit = units[(side, slot)]
            if unit is None or unit.hp <= 0:
                continue
            if action.get("type") == "switch":
                incoming = self._switch_in(action.get("species"))
                if incoming is None or incoming.card is None:
                    continue
                unit = _Unit(side, slot, incoming, 1.0, 1.0, {})
                units[(side, slot)] = unit
                history.append(unit)
                continue
            if action.get("type") != "move" or unit.flinched:
                continue
            move = action.get("move_id")
            key = dex.to_id(move)
            species = unit.active.summary.get("species")
            if key in threat.PROTECT_LIKE:
                failed = side == "us" and memory is not None and memory.protected_last(species)
                unit.protected = not failed
                continue
            if key in _SETUP:
                for stat, n in _SETUP[key].items():
                    unit.boosts[stat] = min(6, unit.boosts.get(stat, 0) + n)
                continue
            if not dex.is_damaging(move):
                continue
            if key == "fakeout":
                fresh = (memory.fake_out_ok(species, self.ctx.turn) if memory is not None else self.ctx.turn == 1) \
                    if side == "us" else (memory.foe_fake_out_ok(species, self.ctx.turn) if memory is not None else self.ctx.turn == 1)
                if not fresh:
                    continue
            self._strike(units, unit, action, move)

        gain = loss = 0.0
        for u in history:
            delta = (u.start - max(0.0, u.hp)) * 100
            ko = u.hp <= 0 < u.start
            if u.side == "them":
                gain += delta + (KO_TEMPO if ko else 0.0)
            else:
                loss += delta + (KO_TEMPO if ko else 0.0)
        return gain - loss

    def _strike(self, units: dict, att: _Unit, action: dict, move: str) -> None:
        kind = (dex.move(move) or {}).get("target")
        foe_side = "them" if att.side == "us" else "us"
        foes = [units.get((foe_side, s)) for s in (0, 1)]
        foes = [f for f in foes if f is not None and f.hp > 0]
        if kind in ("allAdjacentFoes", "allAdjacent"):
            targets = list(foes)
            if kind == "allAdjacent":
                ally = units.get((att.side, 1 - att.slot))
                if ally is not None and ally.hp > 0:
                    targets.append(ally)
        else:
            wanted = action.get("target")
            index = wanted - 1 if isinstance(wanted, int) and wanted > 0 else None
            pick = units.get((foe_side, index)) if index is not None else None
            if pick is None or pick.hp <= 0:
                pick = foes[0] if foes else None  # redirected: the original target is gone
            targets = [pick] if pick is not None else []
        spread = len(targets) > 1
        for target in targets:
            if target.protected and target.side != att.side:
                continue
            target.hp = max(0.0, target.hp - self._fraction(att, target, move, spread))
            if dex.to_id(move) == "fakeout":
                target.flinched = True

    # -- scoring --------------------------------------------------------------------

    def value(self, plan: dict) -> float | None:
        """The expected HP swing (percent of max HP, KOs weighted) of this plan, or
        ``None`` when the board is not readable."""
        if not self.ok:
            return None
        try:
            swings = [(self.swing(plan, fp), w) for fp, w in self.foe_plans]
        except Exception:  # noqa: BLE001 - scoring must never break a decision
            return None
        mean = sum(s * w for s, w in swings)
        worst = min(s for s, _ in swings)
        return (1 - WORST_CASE_WEIGHT) * mean + WORST_CASE_WEIGHT * worst
