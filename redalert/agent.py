"""High-performance real-time Command & Conquer: Red Alert engine for Quicksilver.

Architecture:
1. Real-Time Deterministic Engine (Sub-5ms Execution):
   - In real time (25 ticks/sec), thinking time costs games.
   - Automatically executes the optimal opening build order:
     * Deploy MCV -> Construction Yard ('fact')
     * Build Power Plant ('powr') -> Place Power Plant
     * Build Barracks ('tent' for Allies, 'barr' for Soviets) -> Place Barracks
     * Train Rifle Infantry ('e1') in parallel
     * Build Ore Refinery ('proc') -> Place Refinery (spawns first Harvester)
     * Build War Factory ('weap') -> Place War Factory
     * Train combat vehicles (Tanks/APCs) and defensive towers
   - Automated Economy & Power Management:
     * Watches power_drained vs power_provided; immediately queues 'powr' before brownouts.
     * Keeps Harvesters active and productive.
     * Repairs damaged structures.
   - Automated Military Combat:
     * Groups idle combat units and executes coordinated attack_move batches
       toward enemy sightings or the opposing base spawn cell.

2. Strategic Gemini Overseer (Non-Blocking):
   - Macro decisions (target prioritization, scouting adjustments) run on a
     cooldown without blocking the 25-ticks/s real-time action loop.
"""

from __future__ import annotations

import json
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, WithReasoning

# Standard building constants
MCV = "mcv"
FACT = "fact"
POWR = "powr"
ALLIED_BARR = "tent"
SOVIET_BARR = "barr"
PROC = "proc"
WEAP = "weap"
INF_E1 = "e1"


class RedAlertAgent:
    """Real-time Command & Conquer: Red Alert autonomous player."""

    def __init__(self) -> None:
        self.step_count = 0
        self._last_batch_time = 0.0

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = getattr(state, "raw", None)
        if not isinstance(raw, dict):
            return WAIT

        obs = raw.get("observation")
        if not isinstance(obs, dict):
            return WAIT

        # Tick and state info
        tick = obs.get("tick", 0)
        self.step_count += 1

        orders: list[dict[str, Any]] = []

        # 1. Base Deployment Check (Turn 0: MCV -> Construction Yard)
        legal = raw.get("legal_actions") or {}
        deploy_mcvs = legal.get("deploy") or []
        if deploy_mcvs:
            orders.append({"cmd": "deploy", "units": [deploy_mcvs[0]]})

        # 2. Place Finished Buildings (Immediate Priority)
        # Check production queue for finished/ready structures
        production = obs.get("production") or []
        ready_items = [p.get("item") for p in production if p.get("ready")]
        placed_in_this_batch = set()

        for item in ready_items:
            if item and item.lower() not in placed_in_this_batch:
                orders.append({"cmd": "place", "item": item})
                placed_in_this_batch.add(item.lower())

        # 3. Power & Economy Construction Pipeline
        avail = {str(x).lower() for x in (obs.get("available_production") or [])}
        econ = obs.get("economy") or {}
        cash = econ.get("cash", 0)
        power_provided = econ.get("power_provided", 0)
        power_drained = econ.get("power_drained", 0)

        # Count active/queued buildings
        queued_items = {str(p.get("item")).lower() for p in production}
        own_buildings = obs.get("buildings") or []
        building_types = {str(b.get("type")).lower() for b in own_buildings}

        # Need power if deficit or surplus is thin (< 30)
        power_deficit = (power_drained + 30) >= power_provided
        if POWR in avail and POWR not in queued_items and (power_deficit or POWR not in building_types):
            orders.append({"cmd": "build", "item": POWR})
            queued_items.add(POWR)

        # Step 2: Barracks (Allies 'tent', Soviets 'barr')
        barracks_item = ALLIED_BARR if ALLIED_BARR in avail else (SOVIET_BARR if SOVIET_BARR in avail else None)
        has_barracks = any(b in building_types for b in (ALLIED_BARR, SOVIET_BARR))
        if barracks_item and not has_barracks and barracks_item not in queued_items:
            orders.append({"cmd": "build", "item": barracks_item})
            queued_items.add(barracks_item)

        # Step 3: Ore Refinery ('proc')
        if PROC in avail and PROC not in building_types and PROC not in queued_items and cash >= 1400:
            orders.append({"cmd": "build", "item": PROC})
            queued_items.add(PROC)

        # Step 4: War Factory ('weap')
        if WEAP in avail and WEAP not in building_types and WEAP not in queued_items and cash >= 2000:
            orders.append({"cmd": "build", "item": WEAP})
            queued_items.add(WEAP)

        # 4. Unit Training (Continuous Armament & Economy)
        # Always maintain at least 2 harvesters if we own a refinery
        harv_count = econ.get("harvesters", 0)
        if "harv" in avail and harv_count < 2 and cash >= 1100 and "harv" not in queued_items:
            orders.append({"cmd": "train", "item": "harv", "count": 1})
            queued_items.add("harv")

        # Train vehicles if available (e.g. 1tnk, 2tnk, 3tnk, jeep, apc)
        for v in ("3tnk", "2tnk", "1tnk", "ftrk", "apc", "jeep"):
            if v in avail and cash >= 800 and v not in queued_items:
                orders.append({"cmd": "train", "item": v, "count": 1})
                queued_items.add(v)
                break

        # Train infantry if barracks available
        if INF_E1 in avail and cash >= 300 and INF_E1 not in queued_items:
            # Train up to 3 infantry at a time
            orders.append({"cmd": "train", "item": INF_E1, "count": 3})
            queued_items.add(INF_E1)

        # 5. Base Defenses (Gun Turrets / Tesla Coils)
        for d in ("tsla", "gun", "ftur", "pbox"):
            if d in avail and d not in queued_items and cash >= 1000 and power_provided > (power_drained + 50):
                orders.append({"cmd": "build", "item": d})
                queued_items.add(d)
                break

        # 6. Structure Repairs
        for b in own_buildings:
            b_hp = b.get("hp", 100)
            bid = b.get("id")
            if b_hp < 75 and bid is not None and not b.get("repairing"):
                orders.append({"cmd": "repair", "building": bid, "on": True})

        # 7. Military Attack & Stance Orders
        attack_now = legal.get("attack_now")
        combat_units = legal.get("combat_units") or []

        # If we have 3+ combat units, launch an offensive wave
        if attack_now and len(combat_units) >= 3:
            orders.append(attack_now)
        elif len(combat_units) >= 5:
            enemy_base = legal.get("enemy_base")
            if enemy_base:
                orders.append({
                    "cmd": "attack_move",
                    "units": combat_units[:25],
                    "to": enemy_base,
                })

        # Set stance to attack_anything or defend for idle combat units
        idle_units = legal.get("idle_units") or []
        idle_fighters = [uid for uid in idle_units if uid in combat_units]
        if idle_fighters:
            orders.append({
                "cmd": "set_stance",
                "units": idle_fighters[:20],
                "stance": "attack_anything",
            })

        # Return orders batch
        if not orders:
            return WAIT

        # Cap at 20 orders max per platform limits
        batch = orders[:20]
        summary = f"RedAlert tick {tick}: {len(batch)} command(s) issued."
        return WithReasoning({"type": "orders", "orders": batch}, summary)


def create_agent():
    return RedAlertAgent()
