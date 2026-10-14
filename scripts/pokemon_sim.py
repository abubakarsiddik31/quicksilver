"""A small VGC-doubles battle simulator for stress-testing the Pokémon policy.

It plays two decision policies against each other on the *same* observation and
action shapes the live server uses (``active_pokemon``, ``opponent_active_pokemon``,
a ``doubles_turn`` template with per-option ``targets``), so the agent's real
``_doubles_turn`` runs unmodified. It is strict: an illegal action (a target not
offered, two slots switching to one Pokémon, both slots passing, junk) raises
``SimError``, so a policy bug shows up as a crash rather than a quiet loss.

**What it models:** the damage formula (the project's own calculator, rolls and
accuracy included), speed order with priority, Tailwind, Trick Room, Protect
(fails back to back), Fake Out (first turn only, flinch), Swords Dance / Nasty
Plot / Calm Mind, Thunder Wave, Will-O-Wisp, spread moves (including hitting the
partner), switching and replacements.

**What it does not:** abilities (Intimidate, redirection…), most items, terrain,
weather, screens, Tera, hazards, and the server's real clock. The engine shares
its damage calculator with the policy, so a win rate here measures decision
quality under those assumptions — not strength against the live ladder.

Run:  PYTHONPATH=. python scripts/pokemon_sim.py [--matches 200] [--seed 1]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import types as pytypes
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokemon import dex
from pokemon.damage import damage, parse_card

POOL = json.loads((Path(__file__).resolve().parent.parent / "pokemon" / "pool.json").read_text())[
    "cards"
]
SINGLE = {"normal", "any", "adjacentFoe", "randomNormal"}
PRIORITY_BOOST = {
    "fakeout": 3,
    "protect": 4,
    "detect": 4,
    "spikyshield": 4,
    "banefulbunker": 4,
    "kingsshield": 4,
}
SETUP = {"nastyplot": {"spa": 2}, "swordsdance": {"atk": 2}, "calmmind": {"spa": 1, "spd": 1}}
PROTECT = {"protect", "detect", "spikyshield", "banefulbunker", "kingsshield"}


class SimError(Exception):
    """The policy sent something the server would have refused."""


@dataclass
class Mon:
    raw: dict
    card: Any
    hp: int
    max_hp: int
    boosts: dict = field(default_factory=dict)
    status: str | None = None
    protect_last: bool = False
    protected: bool = False
    turns_out: int = 0
    flinched: bool = False

    @property
    def name(self) -> str:
        return self.raw["species"]

    @property
    def alive(self) -> bool:
        return self.hp > 0


class Battle:
    def __init__(self, teams: list[list[dict]], rng: random.Random) -> None:
        self.rng = rng
        self.mons = [[self._mon(c) for c in team] for team in teams]
        self.active = [[0, 1], [0, 1]]
        self.tailwind = [0, 0]
        self.trick_room = 0
        self.turn = 1

    @staticmethod
    def _mon(raw: dict) -> Mon:
        card = parse_card(raw)
        return Mon(raw=raw, card=card, hp=card.stats["hp"], max_hp=card.stats["hp"])

    # -- views --------------------------------------------------------------

    def _slot(self, side: int, slot: int) -> Mon | None:
        index = self.active[side][slot]
        mon = self.mons[side][index] if index is not None else None
        return mon if mon is not None and mon.alive else None

    def _summary(self, mon: Mon | None, *, own: bool) -> dict | None:
        if mon is None:
            return None
        out = {
            "species": mon.name,
            "current_hp": mon.hp,
            "max_hp": mon.max_hp,
            "current_hp_fraction": mon.hp / mon.max_hp,
            "status": mon.status,
            "ability": mon.raw.get("ability"),
            "item": mon.raw.get("item"),
            "types": list(mon.card.types),
            "base_stats": dex.base_stats(mon.name),
            "boosts": dict(mon.boosts),
            "fainted": not mon.alive,
            "active": True,
        }
        if own:
            out["moves"] = list(mon.raw["moves"])
        return out

    def observation(self, side: int) -> dict:
        foe = 1 - side
        return {
            "phase": "moving",
            "is_doubles": True,
            "turn": self.turn,
            "weather": None,
            "fields": ["Trick Room"] if self.trick_room else [],
            "side_conditions": {"tailwind": self.tailwind[side]} if self.tailwind[side] else {},
            "opponent_side_conditions": {"tailwind": self.tailwind[foe]}
            if self.tailwind[foe]
            else {},
            "active_pokemon": [self._summary(self._slot(side, s), own=True) for s in (0, 1)],
            "opponent_active_pokemon": [
                self._summary(self._slot(foe, s), own=False) for s in (0, 1)
            ],
            "clock": {"decision_seconds_left": 50, "bank_seconds_left": 400},
        }

    def template(self, side: int) -> dict:
        foe = 1 - side
        foes = [s + 1 for s in (0, 1) if self._slot(foe, s)]
        bench = [
            m.name for i, m in enumerate(self.mons[side]) if m.alive and i not in self.active[side]
        ]
        slots = []
        for s in (0, 1):
            mon = self._slot(side, s)
            options: list[dict] = []
            if mon is not None:
                for move_id in mon.raw["moves"]:
                    target = (dex.move(move_id) or {}).get("target")
                    targets = foes if target in SINGLE else [0]
                    if targets:
                        options.append(
                            {"type": "move", "move_id": move_id, "targets": list(targets)}
                        )
                options += [{"type": "switch", "species": b} for b in bench]
            else:
                options.append({"type": "pass"})
            slots.append({"slot": s, "options": options})
        return {"type": "doubles_turn", "slots": slots}

    # -- validation -----------------------------------------------------------

    def validate(self, side: int, action: Any) -> list[dict | None]:
        if not isinstance(action, dict) or action.get("type") != "doubles_turn":
            raise SimError(f"not a doubles_turn: {action!r}")
        template, chosen, switched = self.template(side), [], set()
        for s in (0, 1):
            pick = action.get(f"slot_{s}")
            options = template["slots"][s]["options"]
            if not isinstance(pick, dict):
                raise SimError(f"slot_{s} missing: {action!r}")
            if pick.get("type") == "pass":
                if not any(o["type"] == "pass" for o in options):
                    raise SimError(f"slot_{s} passed while it had a Pokémon")
                chosen.append(None)
            elif pick.get("type") == "switch":
                if not any(
                    o == {"type": "switch", "species": pick.get("species")} for o in options
                ):
                    raise SimError(f"illegal switch {pick!r}")
                if pick["species"] in switched:
                    raise SimError("two slots switched into the same Pokémon")
                switched.add(pick["species"])
                chosen.append(pick)
            elif pick.get("type") == "move":
                option = next(
                    (
                        o
                        for o in options
                        if o["type"] == "move" and o["move_id"] == pick.get("move_id")
                    ),
                    None,
                )
                if option is None:
                    raise SimError(f"illegal move {pick!r}")
                target = pick.get("target", 0 if option["targets"] == [0] else None)
                if target not in option["targets"]:
                    raise SimError(f"illegal target {pick!r}; offered {option['targets']}")
                chosen.append({**pick, "target": target})
            else:
                raise SimError(f"junk slot {pick!r}")
        if chosen == [None, None]:
            raise SimError("both slots passed")
        return chosen

    # -- turn -----------------------------------------------------------------

    def speed(self, side: int, mon: Mon) -> float:
        value = mon.card.stats["spe"]
        stage = mon.boosts.get("spe", 0)
        value *= (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
        if mon.raw.get("item") == "Choice Scarf":
            value *= 1.5
        if self.tailwind[side]:
            value *= 2
        if mon.status == "par":
            value *= 0.5
        return value

    def step(self, actions: list[list[dict | None]]) -> None:
        for side in (0, 1):
            for mon in self.mons[side]:
                mon.protected = False
                mon.flinched = False
        queue = []
        for side in (0, 1):
            for slot in (0, 1):
                act = actions[side][slot]
                mon = self._slot(side, slot)
                if act is None or mon is None:
                    continue
                if act["type"] == "switch":
                    queue.append((10, 0.0, side, slot, act))
                    continue
                key = dex.to_id(act["move_id"])
                priority = PRIORITY_BOOST.get(key, dex.move_priority(act["move_id"]))
                speed = self.speed(side, mon) * (-1 if self.trick_room else 1)
                queue.append((priority, speed + self.rng.random() * 1e-3, side, slot, act))
        queue.sort(key=lambda q: (-q[0], -q[1]))
        for _, _, side, slot, act in queue:
            mon = self._slot(side, slot)
            if mon is None:
                continue
            if act["type"] == "switch":
                self._switch(side, slot, act["species"])
            elif mon.flinched:
                continue
            else:
                self._move(side, slot, mon, act)
        for side in (0, 1):
            for mon in self.mons[side]:
                if mon.status == "brn" and mon.alive:
                    mon.hp = max(0, mon.hp - mon.max_hp // 16)
                if not any(
                    dex.to_id(a["move_id"]) in PROTECT
                    for a in actions[side]
                    if a and a["type"] == "move" and a is not None
                ):
                    mon.protect_last = False
            for slot in (0, 1):
                mon = self._slot(side, slot)
                if mon:
                    mon.turns_out += 1
            self.tailwind[side] = max(0, self.tailwind[side] - 1)
        self.trick_room = max(0, self.trick_room - 1)
        self._replace()
        self.turn += 1

    def _switch(self, side: int, slot: int, species: str) -> None:
        index = next(i for i, m in enumerate(self.mons[side]) if m.name == species)
        mon = self.mons[side][self.active[side][slot]]
        mon.boosts, mon.protect_last = {}, False
        self.active[side][slot] = index
        self.mons[side][index].turns_out = 0

    def _replace(self) -> None:
        for side in (0, 1):
            for slot in (0, 1):
                if self._slot(side, slot) is None:
                    bench = [
                        i
                        for i, m in enumerate(self.mons[side])
                        if m.alive and i not in self.active[side]
                    ]
                    if bench:
                        self.active[side][slot] = bench[0]
                        self.mons[side][bench[0]].turns_out = 0

    def _targets(self, side: int, slot: int, act: dict) -> list[tuple[int, Mon]]:
        kind = (dex.move(act["move_id"]) or {}).get("target")
        foes = [(1 - side, s) for s in (0, 1) if self._slot(1 - side, s)]
        if kind == "allAdjacentFoes":
            return [(f, self._slot(f, s)) for f, s in foes]
        if kind == "allAdjacent":
            out = [(f, self._slot(f, s)) for f, s in foes]
            ally = self._slot(side, 1 - slot)
            return out + ([(side, ally)] if ally else [])
        if kind in SINGLE:
            wanted = act["target"] - 1
            pick = self._slot(1 - side, wanted) or next(
                (self._slot(1 - side, s) for s in (0, 1) if self._slot(1 - side, s)), None
            )
            return [(1 - side, pick)] if pick else []
        return []

    def _move(self, side: int, slot: int, mon: Mon, act: dict) -> None:
        move, key = act["move_id"], dex.to_id(act["move_id"])
        if key in PROTECT:
            if mon.protect_last or self.rng.random() < 0.0:
                mon.protect_last = False
            else:
                mon.protected, mon.protect_last = True, True
            return
        mon.protect_last = False
        if key == "tailwind":
            self.tailwind[side] = 4
        elif key == "trickroom":
            self.trick_room = 0 if self.trick_room else 5
        elif key in SETUP:
            for stat, n in SETUP[key].items():
                mon.boosts[stat] = min(6, mon.boosts.get(stat, 0) + n)
        if key == "fakeout" and mon.turns_out > 0:
            return
        targets = self._targets(side, slot, act)
        spread = len(targets) > 1
        for foe_side, target in targets:
            if target.protected and foe_side != side:
                continue
            accuracy = (dex.move(move) or {}).get("accuracy")
            if (
                accuracy is not True
                and accuracy is not None
                and self.rng.random() * 100 >= accuracy
            ):
                continue
            if (
                key == "thunderwave"
                and not target.status
                and not set(target.card.types) & {"Ground", "Electric"}
            ):
                target.status = "par"
            elif key == "willowisp" and not target.status and "Fire" not in target.card.types:
                target.status = "brn"
            if dex.is_damaging(move):
                result = damage(
                    mon.card,
                    target.card,
                    move,
                    attacker_boosts=mon.boosts,
                    defender_boosts=target.boosts,
                    spread=spread,
                    burned=mon.status == "brn",
                    defender_full_hp=target.hp == target.max_hp,
                )
                if result.max_damage > 0:
                    target.hp = max(
                        0, target.hp - self.rng.randint(result.min_damage, result.max_damage)
                    )
            if key == "fakeout":
                target.flinched = True

    def winner(self) -> int | None:
        alive = [any(m.alive for m in team) for team in self.mons]
        if all(alive):
            return None
        return 0 if alive[0] else 1 if alive[1] else -1


# -- policies ------------------------------------------------------------------

Policy = Callable[[dict, dict], dict]


def first_legal() -> Policy:
    """The repo's own baseline: each slot's first move, first target."""

    def policy(obs: dict, template: dict) -> dict:
        action: dict = {"type": "doubles_turn"}
        for slot in template["slots"]:
            opts = slot["options"]
            move = next((o for o in opts if o["type"] == "move"), None)
            if move:
                action[f"slot_{slot['slot']}"] = {
                    "type": "move",
                    "move_id": move["move_id"],
                    **({"target": move["targets"][0]} if move["targets"] != [0] else {}),
                }
            else:
                action[f"slot_{slot['slot']}"] = {"type": "pass"}
        return action

    return policy


def random_legal(seed: int) -> Policy:
    rng = random.Random(seed)

    def policy(obs: dict, template: dict) -> dict:
        action: dict = {"type": "doubles_turn"}
        switched: set[str] = set()
        for slot in template["slots"]:
            opts = [
                o for o in slot["options"] if o["type"] != "switch" or o["species"] not in switched
            ]
            o = rng.choice(opts)
            if o["type"] == "move":
                action[f"slot_{slot['slot']}"] = {
                    "type": "move",
                    "move_id": o["move_id"],
                    **({"target": rng.choice(o["targets"])} if o["targets"] != [0] else {}),
                }
            elif o["type"] == "switch":
                switched.add(o["species"])
                action[f"slot_{slot['slot']}"] = {"type": "switch", "species": o["species"]}
            else:
                action[f"slot_{slot['slot']}"] = {"type": "pass"}
        return action

    return policy


def quicksilver(cards_ours: list[dict], cards_theirs: list[dict]) -> Policy:
    """The real agent's ``_doubles_turn``, with Jev off and both rosters known."""
    from pokemon.agent import PokemonAgent

    agent = PokemonAgent(pool_path=Path("/nonexistent/pool.json"), jev_client=None)
    agent._agent_id = "me"
    agent._match_cards.observe({"rosters": {"me": cards_ours, "them": cards_theirs}}, agent_id="me")

    def policy(obs: dict, template: dict) -> dict:
        state = pytypes.SimpleNamespace(raw={"observation": obs}, observation="")
        return agent._doubles_turn(template, state)

    return policy


def old_quicksilver(cards_ours: list[dict], cards_theirs: list[dict]) -> Policy | None:
    """The pre-fix greedy scorer, loaded from git (``HEAD:pokemon/battle.py``)."""
    global _OLD
    if "_OLD" not in globals():
        try:
            source = subprocess.run(
                ["git", "show", "HEAD:pokemon/battle.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent.parent,
                check=True,
            ).stdout
            module = pytypes.ModuleType("battle_old")
            sys.modules["battle_old"] = module
            exec(compile(source, "battle_old", "exec"), module.__dict__)
            _OLD = module
        except Exception:  # noqa: BLE001
            _OLD = None
    if (
        _OLD is None
        or not hasattr(_OLD, "choose_turn")
        or "memory" in _OLD.choose_turn.__code__.co_varnames
    ):
        return None
    cards = cards_ours + cards_theirs

    def policy(obs: dict, template: dict) -> dict:
        return _OLD.choose_turn(template, obs, pool_cards=cards) or first_legal()(obs, template)

    return policy


# -- running ---------------------------------------------------------------------


def draw_teams(rng: random.Random) -> list[list[dict]]:
    picked = rng.sample(POOL, 8)
    return [picked[:4], picked[4:]]


def play(make_a, make_b, teams: list[list[dict]], rng: random.Random, max_turns: int = 40) -> int:
    """0 if A wins, 1 if B wins, -1 on a draw. ``make_*(ours, theirs)`` build a fresh policy."""
    battle = Battle(teams, rng)
    policies = [make_a(teams[0], teams[1]), make_b(teams[1], teams[0])]
    while battle.winner() is None and battle.turn <= max_turns:
        actions = []
        for side in (0, 1):
            obs, template = battle.observation(side), battle.template(side)
            actions.append(battle.validate(side, policies[side](obs, template)))
        battle.step(actions)
    result = battle.winner()
    if result is None:  # out of turns: more total HP left wins
        left = [sum(m.hp / m.max_hp for m in team) for team in battle.mons]
        return 0 if left[0] > left[1] else 1 if left[1] > left[0] else -1
    return result


def versus(make_a, make_b, matches: int, seed: int) -> tuple[int, int, int]:
    """(A wins, B wins, draws); each draw of teams is played with A on each of the two teams."""
    a = b = d = 0
    for i in range(matches):
        rng = random.Random(seed * 100003 + i)
        teams = draw_teams(rng)
        for swap in (False, True):
            ts = teams[::-1] if swap else teams
            r = play(make_a, make_b, ts, random.Random(seed * 7919 + i))
            if r == 0:
                a += 1
            elif r == 1:
                b += 1
            else:
                d += 1
    return a, b, d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--matches", type=int, default=100, help="team draws; each is played from both seats"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rivals = {
        "first-legal-move": lambda o, t: first_legal(),
        "random-legal": lambda o, t: random_legal(args.seed),
        "previous scorer (git HEAD)": old_quicksilver,
    }
    for name, make in rivals.items():
        if name.startswith("previous") and make([], []) is None:
            print(f"{name:28} skipped (HEAD has no older battle.py to load)")
            continue
        a, b, d = versus(quicksilver, make, args.matches, args.seed)
        n = a + b + d
        print(
            f"quicksilver vs {name:28} {a:4}-{b:<4} draws {d:<3} win rate {100 * a / n:5.1f}%  ({n} games)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
