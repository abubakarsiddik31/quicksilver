"""A small Red Alert war simulator: our agent against a scripted opponent that keeps its army together.

``tests.ra_sim`` measures the build only (no enemy). This adds positions, movement, combat, towers, vision and
an enemy that behaves like the one that beat us in live matches: it masses its army behind flame towers at
its base, then pushes our base with the whole group in waves that grow, and defends when we come close.

It is a model, not the game: unit numbers are rough and the opponent is a script. Use it to compare two agent
versions on the same scenario (``scripts/ra_battle.py``), not to predict a score.
"""

from __future__ import annotations

import math
import os
import random
from types import SimpleNamespace

from altruagent import WAIT
from tests.ra_sim import CTX, ITEMS, Sim

# type: (hp, dps vs infantry, dps vs armour, range, speed, armoured)
STATS = {
    "e1": (50, 8, 2, 5, 2.6, False),
    "e3": (45, 4, 14, 5, 2.4, False),
    "1tnk": (220, 12, 14, 6, 4.0, True),
    "2tnk": (300, 16, 18, 6, 3.6, True),
    "3tnk": (380, 18, 22, 6, 3.0, True),
    "harv": (400, 0, 0, 0, 2.5, True),
    "mcv": (300, 0, 0, 0, 2.0, True),
}
BUILDING_HP = {
    "fact": 900,
    "powr": 400,
    "proc": 600,
    "barr": 500,
    "weap": 600,
    "fix": 400,
    "dome": 400,
    "pbox": 400,
    "ftur": 450,
    "tsla": 600,
}
TOWERS = {
    "pbox": (25, 8, 6.5),
    "ftur": (45, 10, 6.5),
    "tsla": (50, 40, 8.0),
}  # dps inf, dps armour, range
COST = {name: v[0] for name, v in ITEMS.items()}
COST.update({"2tnk": 800, "fact": 2000})
SIGHT = 11.0
ENGAGE = 9.0  # attack-anything stance chases this far
MY_BASE, ENEMY_BASE = (20.0, 20.0), (80.0, 80.0)


def _d(a, b) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _dmg(attacker_type: str, target_type: str) -> float:
    s = STATS.get(attacker_type)
    if s is None:
        d = TOWERS.get(attacker_type)
        return (
            0.0 if d is None else (d[1] if target_type in STATS and STATS[target_type][5] else d[0])
        )
    return s[2] if (STATS.get(target_type) or (0, 0, 0, 0, 0, False))[5] else s[1]


def _range(t: str) -> float:
    return STATS[t][3] if t in STATS else TOWERS.get(t, (0, 0, 0))[2]


class Battle(Sim):
    """Our side runs on the economy sim; this adds the war."""

    @property
    def mine(self) -> list[dict]:
        return self.units  # the economy sim owns this list (deploy rebinds it), so never cache it

    def __init__(self, faction: str = "russia", seed: int = 1, enemy_scale: float = 1.0) -> None:
        super().__init__(faction)
        ITEMS.setdefault("2tnk", (800, 28, "Vehicle", ["weap", "fix"]))
        self.rng = random.Random(seed)
        self.scale = enemy_scale
        self.units = [{"id": 1, "type": "mcv", "x": 20.0, "y": 20.0, "hp": 300, "side": 0}]
        self.bld: list[dict] = []
        self.foe: list[dict] = []
        self.foe_bld = [
            {"id": 5000, "type": "fact", "x": 80.0, "y": 80.0, "hp": 900},
            {"id": 5001, "type": "proc", "x": 84.0, "y": 78.0, "hp": 600},
            {"id": 5002, "type": "weap", "x": 78.0, "y": 85.0, "hp": 600},
            {"id": 5003, "type": "barr", "x": 85.0, "y": 84.0, "hp": 500},
        ] + [
            {"id": 5010 + i, "type": "ftur", "x": 70.0 + 3 * i, "y": 72.0 - 2 * i, "hp": 450}
            for i in range(3)
        ]
        self.foe_cash, self.foe_next, self.foe_wave, self.foe_t = 500.0, 6000, 0, 0.0
        self.kills = self.deaths = self.k_n = self.d_n = self.bk = self.bl = 0
        self.events: list[dict] = []
        self.seq = 0
        self.timeline: list[dict] = []
        self.stance: dict[int, str] = {}
        self.order: dict[int, tuple] = {}
        self.attack_at = -1e9

    # -- our orders -------------------------------------------------------------------
    def apply(self, orders: list[dict]) -> None:
        super().apply([o for o in orders if o["cmd"] in ("deploy", "build", "train", "place")])
        for o in orders:
            cmd, ids = o["cmd"], o.get("units") or []
            if cmd == "set_stance":
                for i in ids:
                    self.stance[i] = o["stance"]
            elif cmd in ("move", "attack_move") and o.get("to"):
                to = (float(o["to"][0]), float(o["to"][1]))
                for i in ids:
                    if o.get("queued") and i in self.order:
                        self.order[i] = (
                            *self.order[i][:2],
                            to,
                        )  # a queued leg: ignore (scout) beyond the first
                    else:
                        self.order[i] = (cmd, to)
            elif cmd == "cancel":
                for q, items in self.queues.items():
                    for e in items:
                        if e["item"] == str(o.get("item")).lower() and not e["ready"]:
                            self.cash += ITEMS[e["item"]][0] * e["progress"]
                            items.remove(e)
                            break

    def _sync_buildings(self) -> None:
        """Placed structures become positioned buildings (the economy sim only counted them)."""
        have: dict[str, int] = {}
        for b in self.bld:
            have[b["type"]] = have.get(b["type"], 0) + 1
        for t, n in self.owned.items():
            for _ in range(n - have.get(t, 0)):
                i = len(self.bld)
                ang = i * 1.1
                r = 0.0 if t == "fact" else (4.0 + 1.1 * i)
                if t in TOWERS:
                    r, ang = 8.5 + 1.2 * i, 0.9 + 0.5 * i  # towers face the enemy side of the base
                self.bld.append(
                    {
                        "id": 1000 + len(self.bld) + self.rng.randint(0, 0),
                        "type": t,
                        "hp": BUILDING_HP.get(t, 400),
                        "x": 20.0 + r * math.cos(ang),
                        "y": 20.0 + r * math.sin(ang),
                    }
                )
        if self.owned.get("fact") and not any(b["type"] == "fact" for b in self.bld):
            self.owned["fact"] = 0

    # -- the war ------------------------------------------------------------------------
    def _spawn_mine(self) -> None:
        for u in self.mine:
            u.setdefault("x", 24.0 + self.rng.random() * 3)
            u.setdefault("y", 24.0 + self.rng.random() * 3)
            u.setdefault("hp", STATS.get(u["type"], (100,))[0])
            u.setdefault("side", 0)

    def _targets(self, u: dict, foes: list[dict], fbld: list[dict], reach: float) -> dict | None:
        best, bd = None, reach
        for e in foes:
            d = _d(u, e)
            if d < bd:
                best, bd = e, d
        if best is None:
            for b in fbld:
                d = _d(u, b)
                if d < bd:
                    best, bd = b, d
        return best

    def _fight(self, dt: float) -> None:
        self._spawn_mine()
        hits: list[tuple[dict, float, dict]] = []
        groups = ((self.mine, self.foe, self.foe_bld, 0), (self.foe, self.mine, self.bld, 1))
        for army, foes, fbld, side in groups:
            for u in army:
                t = u["type"]
                if t in ("mcv", "harv") or t not in STATS:
                    continue
                rng = _range(t)
                order = self.order.get(u["id"]) if side == 0 else u.get("order")
                stance = self.stance.get(u["id"], "defend") if side == 0 else "attack_anything"
                if order and order[0] == "move":
                    self._go(u, order[1], dt, side)
                    continue
                reach = ENGAGE if (order or stance == "attack_anything") else rng + 3.0  # defenders answer fire
                tgt = self._targets(u, foes, fbld, max(reach, rng))
                if tgt is not None:
                    if _d(u, tgt) > rng:
                        self._step(u, tgt["x"], tgt["y"], STATS[t][4] * dt)
                    else:
                        hits.append((tgt, _dmg(t, tgt["type"]) * dt, u))
                elif order:
                    self._go(u, order[1], dt, side)
        for tw_list, foes, side in ((self.bld, self.foe, 0), (self.foe_bld, self.mine, 1)):
            for b in tw_list:
                if b["type"] in TOWERS:
                    tgt = self._targets(b, foes, [], TOWERS[b["type"]][2])
                    if tgt:
                        hits.append((tgt, _dmg(b["type"], tgt["type"]) * dt, b))
        for tgt, dmg, src in hits:
            tgt["hp"] -= dmg
            if "side" in tgt and tgt.get("side") == 0 or tgt in self.mine or tgt in self.bld:
                self.attack_at = self.t
        self._bury()

    def _step(self, u, x, y, step) -> None:
        d = math.hypot(x - u["x"], y - u["y"])
        if d > 1e-6:
            f = min(1.0, step / d)
            u["x"] += (x - u["x"]) * f
            u["y"] += (y - u["y"]) * f

    def _go(self, u, to, dt, side) -> None:
        self._step(u, to[0], to[1], STATS[u["type"]][4] * dt)
        if math.hypot(to[0] - u["x"], to[1] - u["y"]) < 1.5:
            if side == 0:
                self.order.pop(u["id"], None)
            else:
                u["order"] = None

    def _bury(self) -> None:
        def seq(kind, **kw):
            self.seq += 1
            self.events.append({"seq": self.seq, "kind": kind, **kw})

        for u in [u for u in self.mine if u["hp"] <= 0]:
            self.mine.remove(u)
            self.deaths += COST.get(u["type"], 100)
            self.d_n += 1
            self.order.pop(u["id"], None)
            seq("unit_lost", type=u["type"])
        for b in [b for b in self.bld if b["hp"] <= 0]:
            self.bld.remove(b)
            self.owned[b["type"]] = max(0, self.owned.get(b["type"], 0) - 1)
            self.deaths += COST.get(b["type"], 300)
            self.bl += 1
            seq("building_lost", type=b["type"])
        if self.attack_at >= self.t - 0.2:
            seq("under_attack")
        for u in [u for u in self.foe if u["hp"] <= 0]:
            self.foe.remove(u)
            self.kills += COST.get(u["type"], 100)
            self.k_n += 1
        for b in [b for b in self.foe_bld if b["hp"] <= 0]:
            self.foe_bld.remove(b)
            self.kills += COST.get(b["type"], 300)
            self.bk += 1
        self.events = self.events[-40:]

    # -- the opponent ---------------------------------------------------------------------
    def _enemy_ai(self, dt: float) -> None:
        t = self.t
        self.foe_cash += min(60.0, 9.0 + 0.035 * t) * self.scale * dt
        mix = ("e1", "e1", "e3", "3tnk", "e1", "3tnk", "e3", "2tnk")
        want = mix[self.foe_next % len(mix)]
        if self.foe_cash >= COST[want] and t > 20:
            self.foe_cash -= COST[want]
            self.foe_next += 1
            self.foe.append(
                {
                    "id": self.foe_next,
                    "type": want,
                    "x": 78.0 + self.rng.random() * 4,
                    "y": 74.0 + self.rng.random() * 4,
                    "hp": STATS[want][0],
                    "order": None,
                }
            )
        for u in self.foe:  # the army waits together at its rally point, behind the towers
            if u.get("order") is None and not u.get("pushing"):
                if math.hypot(u["x"] - 74, u["y"] - 72) > 4:
                    self._step(u, 74, 72, STATS[u["type"]][4] * dt)
        waiting = [u for u in self.foe if not u.get("pushing")]
        size = 9 + 2 * self.foe_wave
        if len(waiting) >= size and t > 270:
            self.foe_wave += 1
            for u in waiting:
                u["pushing"] = True
                u["order"] = ("attack_move", MY_BASE)
        for u in self.foe:
            if u.get("pushing") and u.get("order") is None and self.bld:
                u["order"] = ("attack_move", MY_BASE)
        pushing = [u for u in self.foe if u.get("pushing")]
        if 0 < len(pushing) < 3:
            for u in pushing:  # the wave is spent: the survivors fall back to the rally point
                u["pushing"], u["order"] = False, None
        for u in self.foe:  # defend the home base when we come knocking
            if not u.get("pushing") and any(
                math.hypot(m["x"] - 78, m["y"] - 78) < 22 for m in self.mine if m["type"] != "harv"
            ):
                u["order"] = ("attack_move", (70.0, 70.0))

    # -- loop -------------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        super().step(dt)
        self._sync_buildings()
        self._enemy_ai(dt)
        self._fight(dt)

    def over(self) -> str | None:
        if (
            self.t > 30
            and self.owned.get("fact") == 0
            and not any(b["type"] == "fact" for b in self.bld)
            and self.t > 60
            and not self.bld
        ):
            return "lost"
        if not any(b["type"] == "fact" for b in self.foe_bld) and not any(
            b["type"] in ("weap", "barr", "proc") for b in self.foe_bld
        ):
            return "won"
        return None

    def state(self) -> SimpleNamespace:
        s = super().state()
        raw = s.raw
        obs = raw["observation"]
        me = [u for u in self.mine]
        obs["units"] = [
            {
                "id": u["id"],
                "type": u["type"],
                "x": int(u["x"]),
                "y": int(u["y"]),
                "hp": max(1, int(100 * u["hp"] / STATS.get(u["type"], (100,))[0])),
                "idle": u["id"] not in self.order,
                "stance": self.stance.get(u["id"], "defend"),
            }
            for u in me
        ]
        obs["buildings"] = [
            {
                "id": b["id"],
                "type": b["type"],
                "x": int(b["x"]),
                "y": int(b["y"]),
                "hp": max(1, int(100 * b["hp"] / BUILDING_HP.get(b["type"], 400))),
            }
            for b in self.bld
        ]
        eyes = [*me, *self.bld]
        seen = lambda e: any(math.hypot(e["x"] - m["x"], e["y"] - m["y"]) <= SIGHT for m in eyes)  # noqa: E731
        obs["visible_enemies"] = [
            {
                "id": e["id"],
                "type": e["type"],
                "x": int(e["x"]),
                "y": int(e["y"]),
                "hp": int(100 * e["hp"] / STATS[e["type"]][0]),
            }
            for e in self.foe
            if seen(e)
        ]
        obs["visible_enemy_buildings"] = [
            {"id": b["id"], "type": b["type"], "x": int(b["x"]), "y": int(b["y"]), "hp": 100}
            for b in self.foe_bld
            if seen(b)
        ]
        obs["events"] = list(self.events)
        obs["military"] = {
            "kills_cost": self.kills,
            "deaths_cost": self.deaths,
            "units_killed": self.k_n,
            "units_lost": self.d_n,
            "buildings_killed": self.bk,
            "buildings_lost": self.bl,
            "assets_value": 0,
            "army_value": 0,
        }
        combat = [u["id"] for u in obs["units"] if u["type"] not in ("mcv", "harv") and u["idle"]]
        raw["legal_actions"]["combat_units"] = combat
        raw["legal_actions"]["idle_units"] = [u["id"] for u in obs["units"] if u["idle"]]
        return s


def play(
    make_agent,
    seconds: float = 900.0,
    faction: str = "russia",
    seed: int = 1,
    enemy_scale: float = 1.0,
    decide_every: float = 0.2,
) -> dict:
    """Run one match; the result dict carries the outcome and a minute-by-minute timeline."""
    os.environ.setdefault("RA_TRACE", "0")
    sim = Battle(faction, seed, enemy_scale)
    agent = make_agent(lambda: sim.t)
    nxt, mark = 0.0, 60.0
    outcome = None
    while sim.t < seconds and outcome is None:
        if sim.t >= nxt:
            action = agent.choose_action(sim.state(), CTX)
            if action is not WAIT:
                sim.apply(action.action["orders"])
            nxt += decide_every
        sim.step(0.1)
        outcome = sim.over()
        if sim.t >= mark:
            mark += 60.0
            mine = [u for u in sim.mine if u["type"] not in ("mcv", "harv")]
            sim.timeline.append(
                {
                    "t": round(sim.t),
                    "army": len(mine),
                    "tanks": sum(u["type"].endswith("tnk") for u in mine),
                    "harv": sum(u["type"] == "harv" for u in sim.mine),
                    "foe": len(sim.foe),
                    "cash": int(sim.cash),
                    "bld": len(sim.bld),
                    "kills": sim.kills,
                    "lost": sim.deaths,
                }
            )
    mine = [u for u in sim.mine if u["type"] not in ("mcv", "harv")]
    return {
        "outcome": outcome or "timeout",
        "t": round(sim.t),
        "kills": sim.kills,
        "deaths": sim.deaths,
        "units_killed": sim.k_n,
        "units_lost": sim.d_n,
        "buildings_lost": sim.bl,
        "enemy_buildings_killed": sim.bk,
        "army_left": len(mine),
        "foe_waves": sim.foe_wave,
        "timeline": sim.timeline,
    }
