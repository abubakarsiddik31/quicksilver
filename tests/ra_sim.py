"""A tiny Red Alert economy simulator: just enough to run the executor through an opening.

Costs and build times follow the platform guide and the lost-match trace. There is
no enemy: it measures how fast the *build* goes (war factory, harvesters, cash
starvation, tanks, defences), which is what the trace showed we lost on.
"""

from __future__ import annotations

from types import SimpleNamespace

from altruagent import WAIT, DecisionContext

# name: (cost, seconds, queue, prerequisites)
ITEMS = {
    "powr": (300, 14, "Building", ["fact"]),
    "proc": (1400, 42, "Building", ["powr"]),
    "barr": (500, 20, "Building", ["powr"]),
    "weap": (2000, 45, "Building", ["proc"]),
    "fix": (1200, 34, "Building", ["weap"]),
    "pbox": (400, 15, "Defense", ["barr"]),
    "ftur": (600, 18, "Defense", ["barr"]),
    "tsla": (1500, 40, "Defense", ["weap"]),
    "e1": (100, 3, "Infantry", ["barr"]),
    "e3": (300, 5, "Infantry", ["barr"]),
    "harv": (1100, 26, "Vehicle", ["weap", "proc"]),
    "3tnk": (950, 30, "Vehicle", ["weap", "fix"]),
    "1tnk": (700, 24, "Vehicle", ["weap"]),
    "ftrk": (600, 20, "Vehicle", ["weap"]),
}
INCOME_PER_HARV = 20.0  # measured from the lost-match trace: one harvester ~20 cr/s
# The rules table a live match hands the agent through ``context.game_config`` (prerequisites included).
GAME_CONFIG: dict = {"rules": {"buildings": {}, "units": {}}}
for _name, (_cost, _secs, _queue, _pre) in ITEMS.items():
    GAME_CONFIG["rules"]["buildings" if _queue in ("Building", "Defense") else "units"][_name] = {
        "cost": _cost, "queue": _queue, "build_time_s": _secs, "prerequisites": list(_pre),
    }
CTX = DecisionContext(session_id="sim", tournament_id=None, game_type="red_alert", agent_id="q", seat_position=0,
                      game_config=GAME_CONFIG)
# Faction-only items (Soviets: flame tower, heavy tank; Allies: pillbox, light tank).
SOVIET_ONLY, ALLIED_ONLY = {"ftur", "3tnk"}, {"pbox", "1tnk"}


class Sim:
    def __init__(self, faction: str = "russia") -> None:
        self.t, self.cash, self.faction = 0.0, 5000.0, faction
        self.owned: dict[str, int] = {}
        self.units: list[dict] = [{"id": 1, "type": "mcv"}]
        self.next_id = 10
        self.queues: dict[str, list[dict]] = {}
        self.spent = 0.0
        self.log: list[tuple[float, str]] = []
        self.history: list[dict] = []

    # -- world ---------------------------------------------------------------------
    def avail(self) -> list[str]:
        if not self.owned.get("fact"):
            return []
        out = []
        for name, (_, _, _, pre) in ITEMS.items():
            if all(self.owned.get(p) for p in pre) and name not in (ALLIED_ONLY if self.faction == "russia" else SOVIET_ONLY):
                out.append(name)
        return out

    def power(self) -> tuple[int, int]:
        provided = 100 * self.owned.get("powr", 0)
        drained = 30 * self.owned.get("proc", 0) + 30 * self.owned.get("weap", 0) + 20 * self.owned.get("barr", 0) \
            + 30 * self.owned.get("fix", 0)
        return provided, drained

    def step(self, dt: float) -> None:
        self.t += dt
        harvs = sum(1 for u in self.units if u["type"] == "harv")
        if self.owned.get("proc"):
            self.cash += harvs * INCOME_PER_HARV * dt
        provided, drained = self.power()
        speed = 1.0 if provided >= drained else 0.5
        for q, items in self.queues.items():
            for e in items:
                if e["ready"]:
                    continue
                cost, secs, _, _ = ITEMS[e["item"]]
                need = cost * dt * speed / secs
                if self.cash >= need:
                    self.cash -= need
                    self.spent += need
                    e["progress"] += dt * speed / secs
                if e["progress"] >= 1.0:
                    e["progress"], e["ready"] = 1.0, True
                    if q in ("Infantry", "Vehicle"):
                        self.units.append({"id": self.next_id, "type": e["item"], "x": 20, "y": 20})
                        if e["item"] == "harv":
                            pass
                        self.next_id += 1
                        e["done"] = True
                break  # one item per queue at a time
            self.queues[q] = [e for e in items if not e.get("done")]

    def apply(self, orders: list[dict]) -> None:
        for o in orders:
            cmd = o["cmd"]
            if cmd == "deploy" and any(u["id"] == o["units"][0] and u["type"] == "mcv" for u in self.units):
                self.units = [u for u in self.units if u["type"] != "mcv"]
                self.owned["fact"] = 1
                self.log.append((self.t, "deploy"))
            elif cmd in ("build", "train"):
                item = o["item"].lower()
                if item in self.avail():
                    for _ in range(o.get("count", 1) if cmd == "train" else 1):
                        self.queues.setdefault(ITEMS[item][2], []).append(
                            {"item": item, "progress": 0.0, "ready": False})
                    self.log.append((self.t, f"{cmd} {item}"))
            elif cmd == "place":
                for q, items in self.queues.items():
                    for e in items:
                        if e["item"] == o["item"].lower() and e["ready"] and q in ("Building", "Defense"):
                            self.owned[e["item"]] = self.owned.get(e["item"], 0) + 1
                            if e["item"] == "proc":
                                self.units.append({"id": self.next_id, "type": "harv", "x": 22, "y": 22})
                                self.next_id += 1
                            items.remove(e)
                            self.log.append((self.t, f"placed {e['item']}"))
                            break

    # -- the observation the agent sees ---------------------------------------------
    def state(self) -> SimpleNamespace:
        provided, drained = self.power()
        production = [
            {"item": e["item"], "queue": q, "progress": round(e["progress"], 3), "ready": e["ready"],
             "remaining_s": round((1 - e["progress"]) * ITEMS[e["item"]][1], 1)}
            for q, items in self.queues.items() for e in items
        ]
        units = [dict(u, hp=100, idle=True, x=u.get("x", 20), y=u.get("y", 20)) for u in self.units]
        buildings = [{"id": 1000 + i, "type": t, "hp": 100} for i, (t, n) in enumerate(self.owned.items()) for _ in range(n)]
        combat = [u["id"] for u in units if u["type"] not in ("mcv", "harv")]
        obs = {
            "tick": int(self.t * 25), "faction": self.faction, "enemy_faction": "england",
            "economy": {"cash": int(self.cash), "power_provided": provided, "power_drained": drained,
                        "harvesters": sum(1 for u in units if u["type"] == "harv")},
            "production": production, "available_production": self.avail(), "units": units, "buildings": buildings,
            "visible_enemies": [], "visible_enemy_buildings": [], "events": [], "base_center": [20, 20],
        }
        legal = {"deploy": [u["id"] for u in units if u["type"] == "mcv"], "combat_units": combat,
                 "idle_units": [u["id"] for u in units], "enemy_base": [80, 80], "attack_now": None}
        raw = {"observation": obs, "legal_actions": legal, "time": {"game_seconds": self.t, "remaining_seconds": 1200},
               "map": {"id": "sim", "bounds": [0, 0, 96, 96]}}
        return SimpleNamespace(raw=raw, legal_actions=[], is_current_actor=True)


def run(make_agent, seconds: float = 340.0, decide_every: float = 0.2, faction: str = "russia") -> dict:
    """Drive an agent through the sim; returns the same metrics the trace scorer reports.

    ``make_agent(clock)`` builds the agent with the sim's clock so cooldowns run on game time.
    """
    sim, m = Sim(faction), {"zero": 0.0, "idle": 0.0}
    agent = make_agent(lambda: sim.t)
    t_next = 0.0
    while sim.t < seconds:
        if sim.t >= t_next:
            action = agent.choose_action(sim.state(), CTX)
            if action is not WAIT:
                sim.apply(action.action["orders"])
            t_next += decide_every
        if sim.cash < 50:
            m["zero"] += 0.1
        has_weap = bool(sim.owned.get("weap"))
        if has_weap and not any(e["item"] in ("1tnk", "3tnk", "harv", "ftrk") for e in sim.queues.get("Vehicle", [])) \
                and sim.cash >= 300:
            m["idle"] += 0.1
        sim.step(0.1)
        if int(round(sim.t * 10)) % 10 == 0:
            sim.history.append({"t": round(sim.t), "owned": dict(sim.owned), "cash": int(sim.cash),
                                "harv": sum(1 for u in sim.units if u["type"] == "harv"),
                                "tanks": sum(1 for u in sim.units if u["type"] in ("1tnk", "3tnk"))})
    placed: dict[str, float] = {}
    for t, msg in sim.log:  # first placement of each item
        if msg.startswith("placed"):
            placed.setdefault(msg.split()[1], t)
    def at(sec):
        return min(sim.history, key=lambda h: abs(h["t"] - sec))
    return {
        "war_factory_s": round(placed.get("weap", -1)), "first_defence_s": round(min(
            (t for k, t in placed.items() if k in ("pbox", "ftur", "tsla")), default=-1)),
        "harvesters_at_180s": at(180)["harv"], "tanks_at_330s": at(330)["tanks"],
        "zero_cash_s": round(m["zero"]), "vehicle_idle_s": round(m["idle"]), "placed": {k: round(v) for k, v in placed.items()},
        "final": sim.history[-1] if sim.history else None,
    }
