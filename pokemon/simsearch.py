"""Search the battle on the real Showdown rules engine instead of scoring it with heuristics.

``pokemon/battle.py`` ranks a turn by hand-written scores, so every rule it was never taught (a Choice
lock, Focus Sash against Fake Out, Trick Room against Tailwind) is a blind spot that only a lost game
reveals. Here the rules come from ``@pkmn/sim`` itself, the engine Showdown runs on: the live battle is
rebuilt from the observation, each of our joint choices is played against the foe's likely replies for a
few turns, and the plan that wins most often is returned (``pokemon/sim/engine.js``).

Both teams come from the same fixed card pool and the team sheets are open, so the foe's sets are known:
the search is close to a perfect-information game.

Everything here is optional. No Node, no ``npm install``, a timeout or an unreadable board all return
``None`` and the caller falls back to the heuristic planner.
"""

from __future__ import annotations

import json
import os
import random
import select
import shutil
import subprocess
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from pokemon import actions, dex, observation

SIM_DIR = Path(__file__).with_name("sim")
PROTECT_LIKE = {
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "silktrap",
    "obstruct",
}
WEATHERS = {
    "raindance": "raindance",
    "sunnyday": "sunnyday",
    "sandstorm": "sandstorm",
    "snow": "snow",
    "hail": "hail",
}
TERRAINS = {
    "electricterrain": "electricterrain",
    "grassyterrain": "grassyterrain",
    "mistyterrain": "mistyterrain",
    "psychicterrain": "psychicterrain",
}
TRICK_ROOM_TURNS = 5
BRING = 4


def available() -> bool:
    """True when the engine can run here: Node is installed and ``npm install`` was run in ``pokemon/sim``."""
    if os.environ.get("POKEMON_SIM", "1") == "0":
        return False
    return bool(shutil.which("node")) and (SIM_DIR / "node_modules" / "@pkmn" / "sim").is_dir()


class Engine:
    """The Node search process. One per match; restarted on a crash, switched off after repeated ones."""

    MAX_FAILURES = 3

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._failures = 0
        self._lock = threading.Lock()
        self._next_id = 0
        self.last_error: str | None = None

    @property
    def usable(self) -> bool:
        return self._failures < self.MAX_FAILURES and available()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            ["node", str(SIM_DIR / "server.js")],
            cwd=str(SIM_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def search(self, state: dict, budget_ms: int, opts: dict | None = None) -> dict | None:
        """The engine's ranked plans for ``state`` within ``budget_ms`` (plus a safety margin), or ``None``."""
        return self._call({"cmd": "search", "state": state, "budget_ms": budget_ms, "opts": opts or {}}, budget_ms)

    def lineups(self, req: dict, budget_ms: int) -> dict | None:
        """Values of each candidate Team Preview lineup of ours against the foe's likely lineups, or ``None``."""
        return self._call({"cmd": "lineups", "req": {**req, "ms": budget_ms}}, budget_ms)

    def _call(self, payload: dict, budget_ms: int) -> dict | None:
        if not self.usable:
            return None
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._start()
                assert self._proc is not None and self._proc.stdin and self._proc.stdout
                self._next_id += 1
                request = {"id": self._next_id, **payload}
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
                deadline = time.monotonic() + budget_ms / 1000.0 + 5.0
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("search timed out")
                    ready, _, _ = select.select([self._proc.stdout], [], [], left)
                    if not ready:
                        raise TimeoutError("search timed out")
                    line = self._proc.stdout.readline()
                    if not line:
                        raise RuntimeError("engine closed")
                    reply = json.loads(line)
                    if reply.get("id") == self._next_id:
                        break
                if reply.get("error"):
                    raise RuntimeError(reply["error"])
                return reply.get("result")
            except Exception as exc:  # noqa: BLE001 - a broken engine must never cost a match
                self.last_error = str(exc)[:600]
                self._failures += 1
                self.close()
                return None


# -- the live state --------------------------------------------------------------------------


def _norm(name: Any) -> str:
    return dex.to_id(name)


def _card_for(species: Any, cards: list[dict]) -> dict | None:
    want = _norm(species)
    bare = None
    for card in cards or []:
        if _norm(card.get("species")) == want:
            if card.get("moves"):
                return card
            bare = bare or card
    if bare is None and "*" in str(species):
        # Showdown's wildcard forme ("Urshifu-*"): the team sheet shows which one, the preview does not
        for card in cards or []:
            if card.get("moves") and _norm(card.get("species")).startswith(want):
                return card
    return bare if bare is not None and bare.get("moves") else None


def _set_of(card: dict) -> dict:
    return {
        "species": card.get("species"),
        "item": card.get("item") or "",
        "ability": card.get("ability") or "",
        "moves": list(card.get("moves") or []),
        "nature": card.get("nature") or "Hardy",
        "evs": dict(card.get("evs") or {}),
        "ivs": dict(card.get("ivs") or {}),
        "level": int(card.get("level") or 50),
    }


class LogFacts:
    """What the battle log says that the observation does not: who moved since switching in, Choice locks,
    last turn's Protect, consumed items, boost-driven abilities, weather, terrain and Trick Room."""

    def __init__(self) -> None:
        self.turn = 0
        self.moves: dict[str, int] = {}
        self.last_move: dict[str, tuple[str, int]] = {}
        self.consumed: set[str] = set()
        self.volatiles: dict[str, set[str]] = {}
        self.weather: str | None = None
        self.terrain: str | None = None
        self.trick_room_since: int | None = None
        self.roster: dict[str, list[str]] = {"p1": [], "p2": []}

    @staticmethod
    def _key(ident: str) -> str:
        # "p1a: Chien-Pao" -> "p1: Chien-Pao"
        head, _, name = ident.partition(": ")
        return f"{head[:2]}: {name}"

    def feed(self, lines: list[str]) -> LogFacts:
        for raw in lines:
            if not isinstance(raw, str) or not raw.startswith("|"):
                continue
            parts = raw.split("|")
            tag = parts[1] if len(parts) > 1 else ""
            try:
                self._line(tag, parts)
            except Exception:  # noqa: BLE001 - one odd line must not lose the rest
                continue
        return self

    def _line(self, tag: str, parts: list[str]) -> None:
        if tag == "turn":
            self.turn = int(parts[2])
        elif tag == "poke":
            self.roster[parts[2]].append(parts[3].split(",")[0])
        elif tag in ("switch", "drag"):
            key = self._key(parts[2])
            self.moves[key] = 0
            self.last_move.pop(key, None)
            self.volatiles.pop(key, None)
        elif tag == "move":
            key = self._key(parts[2])
            self.moves[key] = self.moves.get(key, 0) + 1
            self.last_move[key] = (_norm(parts[3]), self.turn)
        elif tag == "cant":
            key = self._key(parts[2])
            self.moves[key] = self.moves.get(key, 0) + 1
        elif tag == "-enditem":
            self.consumed.add(self._key(parts[2]))
        elif (
            tag == "-start"
            and len(parts) > 3
            and parts[3].startswith(("protosynthesis", "quarkdrive"))
        ):
            self.volatiles.setdefault(self._key(parts[2]), set()).add(
                parts[3][:13] if parts[3].startswith("protosynthesis") else "quarkdrive"
            )
        elif (
            tag == "-end" and len(parts) > 3 and _norm(parts[3]) in ("protosynthesis", "quarkdrive")
        ):
            self.volatiles.get(self._key(parts[2]), set()).discard(_norm(parts[3]))
        elif tag == "-weather":
            self.weather = None if _norm(parts[2]) == "none" else WEATHERS.get(_norm(parts[2]))
        elif tag == "-fieldstart":
            what = _norm(parts[2].replace("move:", ""))
            if what == "trickroom":
                self.trick_room_since = self.turn
            elif what in TERRAINS:
                self.terrain = TERRAINS[what]
        elif tag == "-fieldend":
            what = _norm(parts[2].replace("move:", ""))
            if what == "trickroom":
                self.trick_room_since = None
            elif what in TERRAINS:
                self.terrain = None

    def protected_last(self, key: str) -> bool:
        move = self.last_move.get(key)
        return bool(move and move[0] in PROTECT_LIKE and move[1] == self.turn - 1)


def _status(value: Any) -> str | None:
    text = str(value or "").lower()
    return text if text in ("brn", "par", "psn", "tox", "slp", "frz") else None


def _boosts(mon: dict) -> dict:
    return {
        k: int(v)
        for k, v in (mon.get("boosts") or {}).items()
        if k in ("atk", "def", "spa", "spd", "spe") and v
    }


def _conditions(raw: Any) -> dict:
    out: dict[str, int] = {}
    for name, left in (raw or {}).items():
        key = _norm(name)
        if key in ("tailwind", "reflect", "lightscreen", "auroraveil"):
            out[key] = int(left) if isinstance(left, (int, float)) and left > 0 else 0
    return out


def build_state(obs: dict, cards: list[dict], *, seed: int | None = None) -> dict | None:
    """The search request for this observation, or ``None`` when the board cannot be rebuilt faithfully.

    We are always side 0 of the engine's battle; the foe's unrevealed bench is filled from the six they showed
    at Team Preview.
    """
    team = obs.get("team") or {}
    foe_team = obs.get("opponent_team") or {}
    if not team or not isinstance(team, dict):
        return None
    first_key = next(iter(team))
    mine = first_key[:2]
    theirs = "p2" if mine == "p1" else "p1"
    facts = LogFacts().feed(obs.get("protocol_log") or [])
    stable = zlib.crc32(str(obs.get("battle_tag") or "").encode())
    rng = random.Random(stable)

    active_ours = observation.active_pokemon(obs)
    active_theirs = observation.opponent_active_pokemon(obs)
    switches = observation.available_switches(obs)
    bench_species = {
        _norm(o.get("species")) for slot in switches for o in (slot or []) if isinstance(o, dict)
    }

    def entry(key: str, value: dict, side_prefix: str, is_ours: bool) -> dict | None:
        card = _card_for(value.get("species"), cards)
        if card is None:
            return None
        fainted = bool(value.get("fainted")) or (value.get("current_hp_fraction") or 0) <= 0
        spec = _set_of(card)
        consumed = key in facts.consumed or (is_ours and not value.get("item"))
        locked = None
        last = facts.last_move.get(key)
        if spec["item"].lower().startswith("choice") and last and not consumed:
            locked = last[0]
        return {
            "set": spec,
            "hpFrac": 0.0 if fainted else float(value.get("current_hp_fraction") or 0),
            "fainted": fainted,
            "status": _status(value.get("status")),
            "boosts": _boosts(value),
            "itemConsumed": consumed,
            "moveActions": facts.moves.get(key, 0),
            "choiceLock": locked,
            "protectedLast": facts.protected_last(key),
            "volatiles": sorted(facts.volatiles.get(key, set())),
            "species": _norm(value.get("species")),
            "key": key,
            "wasActive": bool(value.get("active")),
        }

    def side_mons(
        team_map: dict, prefix: str, active: list, is_ours: bool, want: int
    ) -> list[dict] | None:
        built = {
            k: entry(k, v, prefix, is_ours) for k, v in team_map.items() if isinstance(v, dict)
        }
        if any(m is None for m in built.values()):
            return None, []
        by_species = {m["species"]: m for m in built.values()}
        ordered: list[dict] = []
        forced: list[bool] = []
        for index in range(2):
            slot = active[index] if index < len(active) else None
            if isinstance(slot, dict) and _norm(slot.get("species")) in by_species:
                ordered.append(by_species.pop(_norm(slot["species"])))
                forced.append(False)
            else:
                # an empty slot: the fainted body that last stood there (it awaits a replacement)
                body = next((m for m in by_species.values() if m["fainted"] and m.get("wasActive")), None)
                if body is None:
                    return None, []
                ordered.append(by_species.pop(body["species"]))
                forced.append(True)
        rest = list(by_species.values())
        if is_ours:
            # only the four we brought: the bench we can switch to, and those already down
            rest = [m for m in rest if m["species"] in bench_species or m["fainted"]]
        elif len(ordered) + len(rest) < want:
            seen = {m["species"] for m in ordered + rest}
            pool_names = [n for n in facts.roster.get(prefix, []) if _norm(n) not in seen]
            rng.shuffle(pool_names)
            for name in pool_names[: want - len(ordered) - len(rest)]:
                card = _card_for(name, cards)
                if card is None:
                    continue  # an unknown bench member must not cost the whole search: the rest still plays out
                rest.append(
                    {
                        "set": _set_of(card),
                        "hpFrac": 1.0,
                        "fainted": False,
                        "status": None,
                        "boosts": {},
                        "itemConsumed": False,
                        "moveActions": 0,
                        "choiceLock": None,
                        "protectedLast": False,
                        "volatiles": [],
                        "species": _norm(name),
                        "key": f"{prefix}: {name}",
                    }
                )
        mons = ordered + rest
        return (mons[:want], forced) if len(mons) >= 2 else (None, [])

    ours, ours_empty = side_mons(team, mine, active_ours, True, BRING)
    foes, foes_empty = side_mons(foe_team, theirs, active_theirs, False, BRING)
    if not ours or not foes or len(ours) < 2 or len(foes) < 2:
        return None
    # an empty active slot is a pending replacement only if a living body waits on the bench
    def pending(mons: list[dict], empty: list[bool]) -> list[bool]:
        bench = any(not m["fainted"] for m in mons[2:])
        return [bool(e and bench) for e in empty] + [False] * (2 - len(empty))

    force_ours = pending(ours, ours_empty)
    force_theirs = pending(foes, foes_empty)
    for mons, flags in ((ours, force_ours), (foes, force_theirs)):
        for slot in range(2):
            if mons[slot]["fainted"] and not flags[slot]:
                # a down active with no replacement: the other slot fights on alone; keep the body in place
                pass
    if all(m["fainted"] for m in ours[:2]) or all(m["fainted"] for m in foes[:2]):
        if not (any(force_ours) or any(force_theirs)):
            return None

    trick = 0
    if facts.trick_room_since is not None:
        trick = max(1, TRICK_ROOM_TURNS - (facts.turn - facts.trick_room_since))
    return {
        "turn": facts.turn or int(obs.get("turn") or 1),
        "seed": seed if seed is not None else (stable ^ (facts.turn * 7919)) & 0xFFFFFFFF,
        "weather": facts.weather,
        "terrain": facts.terrain,
        "trickRoom": trick,
        "forceSwitch": {"ours": force_ours, "theirs": force_theirs},
        "sides": [
            {"mons": ours, "conditions": _conditions(obs.get("side_conditions"))},
            {"mons": foes, "conditions": _conditions(obs.get("opponent_side_conditions"))},
        ],
    }


# -- from the engine's answer back to the server's action ---------------------------------------


def to_action(plan: dict, template: dict) -> dict | None:
    """The engine's slots as a ``doubles_turn`` action, only if every part is one the template offers."""
    slots = {s.get("slot", i): s for i, s in enumerate(actions.slots_of(template))}
    action: dict[str, Any] = {"type": "doubles_turn"}
    taken: set[str] = set()
    for part in plan.get("slots", []):
        number = part.get("slot")
        slot = slots.get(number)
        if slot is None:
            return None
        kind = part.get("type")
        if kind == "pass":
            if not actions.has_pass(slot):
                return None
            action[f"slot_{number}"] = dict(actions.PASS)
        elif kind == "switch":
            want = _norm(part.get("species"))
            match = next(
                (s for s in actions.switches_of(slot) if _norm(s.get("species")) == want), None
            )
            choice = actions.switch_choice(match) if match else None
            if choice is None or choice["species"] in taken:
                return None
            taken.add(choice["species"])
            action[f"slot_{number}"] = choice
        elif kind == "move":
            move_id = _norm(part.get("move_id"))
            match = next((m for m in actions.moves_of(slot) if _norm(m.get("move_id")) == move_id), None)
            if match is None:
                return None
            choice = {"type": "move", "move_id": match["move_id"]}
            targets = [t for t in (match.get("targets") or []) if isinstance(t, int)]
            wanted = part.get("target", 0)
            if wanted and wanted in targets:
                choice["target"] = wanted
            elif targets and targets != [0]:
                fallback = actions.target_for(match)
                if fallback is not None:
                    choice["target"] = fallback
            action[f"slot_{number}"] = choice
        else:
            return None
    return action if "slot_0" in action and "slot_1" in action else None


def lineup_request(ours: list[list[dict]], theirs: list[tuple[list[dict], float]], cards: list[dict]) -> dict | None:
    """The engine request for candidate lineups. Each lineup is four cards, leads first.
    ``None`` when any card lacks a full set."""
    def mons(card_list: list[dict]) -> list[dict] | None:
        out = []
        for card in card_list:
            full = _card_for(card.get("species"), cards)
            if full is None:
                return None
            out.append({"set": _set_of(full)})
        return out

    our_sets = [mons(lineup) for lineup in ours]
    their_sets = [(mons(lineup), weight) for lineup, weight in theirs]
    if any(m is None for m in our_sets) or not their_sets or any(m is None for m, _ in their_sets):
        return None
    return {"ours": our_sets, "theirs": [{"mons": m, "w": w} for m, w in their_sets], "rolls": 60, "depth": 14}
