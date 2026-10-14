"""The chain of command: the General's directive, the roster's roles, and the lessons of the lost match."""

from __future__ import annotations

import os

from altruagent import WAIT
from redalert import roles as r
from redalert.plan import Plan
from tests.test_redalert_agent import CTX, advised, make_state, orders_of

os.environ["RA_TRACE"] = "0"


def decide(**kw):
    base = dict(game_s=300, army_n=12, doctrine="mass_army", enemy_types={})
    base.update(kw)
    return r.General().decide(**base)


def test_the_enemy_is_classified_from_what_was_seen():
    assert r.profile({}) == "blind" and r.profile({"e1": 2}) == "blind"
    assert r.profile({"3tnk": 4, "e1": 2}) == "armor"
    assert r.profile({"e1": 8, "e3": 3, "2tnk": 1}) == "infantry"
    assert r.profile({"e1": 3, "2tnk": 3}) == "mixed"
    assert r.profile({"mig": 2, "e1": 4}) == "air"
    assert r.profile({"e1": 2}, enemy_towers=4) == "turtle"


def test_the_general_builds_the_counter_to_what_the_enemy_fields():
    assert decide(enemy_types={"3tnk": 5}).units[0] == "e3"  # rockets and tanks against armour
    assert decide(enemy_types={"e1": 9}).towers[0] == "ftur"  # flame towers against infantry
    assert decide(enemy_types={"e1": 9}).units[0] == "e1"
    assert decide(enemy_types={"mig": 3, "e1": 3}).towers[0] == "sam"


def test_the_home_line_scales_with_the_threat_and_the_doctrine():
    calm = decide(army_n=16)
    assert calm.garrison == 4  # a quarter of the army
    assert (
        decide(army_n=16, doctrine="pressure").garrison == 4
    )  # even attacking, nobody-home is how rushes won
    assert decide(army_n=16, raided_recently=True).garrison == 8
    assert decide(army_n=16, doctrine="defend").garrison == 8
    assert decide(army_n=6, game_s=200).garrison == 4  # never under four once the opening is over
    assert decide(army_n=3).garrison == 1  # a tiny army cannot split
    assert decide(army_n=4, game_s=100).garrison == 1


def test_raids_need_a_big_calm_army_and_scouts_double_when_blind():
    assert decide(army_n=16).raiders == 3
    assert decide(army_n=10).raiders == 0 and decide(army_n=16, game_s=200).raiders == 0
    assert (
        decide(army_n=16, enemy_near=3).raiders == 0
        and decide(army_n=16, doctrine="all_in").raiders == 0
    )
    assert decide(blind_for_s=90).scouts == 2 and decide(blind_for_s=10).scouts == 1


def test_the_vehicle_share_makes_a_combined_arms_army():
    assert decide(game_s=60).vehicle_share == 0.0  # the opening is money for the economy
    assert (
        decide(enemy_types={"3tnk": 5}).vehicle_share
        > decide(enemy_types={"e1": 9}).vehicle_share
        > 0
    )


def soldiers(*kinds):
    cost = {"e1": 100, "e3": 300, "3tnk": 950}
    return [r.Soldier(i, k, cost[k]) for i, k in enumerate(kinds)]


def test_every_soldier_has_exactly_one_role_and_quotas_hold():
    d = r.Directive(
        garrison=2,
        raiders=1,
        scouts=1,
        units=("e3",),
        towers=(),
        vehicle_share=0,
        stance="armor",
        reason="",
    )
    squad = soldiers("e1", "e1", "e3", "3tnk", "3tnk", "e1")
    got = r.assign(squad, {}, d, keep_scouts={0}, preferred=("e3",))
    assert got[0] == r.Role.SCOUT
    assert sorted(v for v in got.values()).count(r.Role.GARRISON) == 2
    assert sum(v == r.Role.RAIDER for v in got.values()) == 1
    assert set(got) == {s.id for s in squad}
    assert got[2] == r.Role.GARRISON  # the rocket soldier counters armour: he holds the line first
    assert got[3] == r.Role.STRIKE and got[4] == r.Role.STRIKE  # armour stays with the push


def test_roles_are_sticky_and_dead_soldiers_drop_out():
    d = r.Directive(
        garrison=2,
        raiders=0,
        scouts=1,
        units=(),
        towers=(),
        vehicle_share=0,
        stance="mixed",
        reason="",
    )
    squad = soldiers("e1", "e1", "e1", "e1")
    first = r.assign(squad, {}, d, keep_scouts=set())
    held = {u for u, v in first.items() if v == r.Role.GARRISON}
    again = r.assign(squad[::-1], first, d, keep_scouts=set())
    assert {u for u, v in again.items() if v == r.Role.GARRISON} == held  # no reshuffle every tick
    survivors = [s for s in squad if s.id not in held]
    after = r.assign(survivors, first, d, keep_scouts=set())
    assert (
        set(after) == {s.id for s in survivors}
        and sum(v == r.Role.GARRISON for v in after.values()) == 2
    )


# -- the lost match's lessons, at the agent level ---------------------------------------------------------------


def test_an_outnumbering_push_is_fought_at_home_not_chased():
    """ra-1791957710: 13 fighters attack-moved onto the nearest of 14 intruders, a few at a time, and died."""
    mine = [{"id": i, "type": "e1", "x": 20, "y": 20} for i in range(12)]
    push = [{"id": 90 + i, "type": "3tnk", "x": 36 + i % 3, "y": 20 + i // 3} for i in range(6)]
    state = make_state(
        tick=9000,
        units=mine,
        combat=list(range(12)),
        enemy_base=[80, 80],
        enemies=push,
        buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}],
    )
    agent, _ = advised({}, plan=Plan(buildings=(), units=(), army_mode="attack"))
    out = orders_of(agent.choose_action(state, CTX))
    chase = [o for o in out if o["cmd"] == "attack_move"]
    assert chase and all(o["to"] == [20, 20] for o in chase)  # mass at the base centre


def test_a_small_raid_is_still_chased_off():
    mine = [{"id": i, "type": "3tnk", "x": 20, "y": 20} for i in range(8)]
    raid = [{"id": 90, "type": "e1", "x": 30, "y": 22}]
    state = make_state(
        tick=9000,
        units=mine,
        combat=list(range(8)),
        enemy_base=[80, 80],
        enemies=raid,
        buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}],
    )
    agent, _ = advised({}, plan=Plan(buildings=(), units=(), army_mode="attack"))
    out = orders_of(agent.choose_action(state, CTX))
    assert any(o["cmd"] == "attack_move" and o["to"] == [30, 22] for o in out)


def test_an_infantry_army_with_a_war_factory_trains_a_tank():
    """The lost match built a 2,000 war factory and not one tank in eight minutes."""
    mine = [{"id": i, "type": "e1", "x": 20, "y": 20} for i in range(8)] + [
        {"id": 40, "type": "harv"}
    ]
    state = make_state(
        tick=5000,
        cash=3000,
        units=mine,
        combat=list(range(8)),
        enemy_base=[80, 80],
        avail=["e1", "e3", "3tnk", "2tnk", "harv", "powr"],
        buildings=[{"id": 1, "type": t} for t in ("fact", "powr", "proc", "barr", "weap")],
    )
    plan = Plan(buildings=(), units=(("e1", 30),))
    agent, _ = advised({}, plan=plan)
    out = orders_of(agent.choose_action(state, CTX))
    assert any(o["cmd"] == "train" and o["item"] in ("3tnk", "2tnk") for o in out)


def test_a_second_harvester_comes_before_the_war_factory():
    state = make_state(
        tick=1500,
        cash=1300,
        avail=["harv", "weap", "powr"],
        buildings=[{"id": 1, "type": t} for t in ("fact", "powr", "proc", "barr")],
        units=[{"id": 40, "type": "harv"}],
    )
    agent, _ = advised({}, plan=Plan())
    out = orders_of(agent.choose_action(state, CTX))
    assert any(o["cmd"] == "train" and o["item"] == "harv" for o in out)


def test_the_second_refinery_survives_the_army_floor_and_the_cancel_reflex():
    state = make_state(
        tick=6000,
        cash=5000,
        avail=["proc", "e1"],
        buildings=[{"id": 1, "type": t} for t in ("fact", "powr", "proc", "barr", "weap")],
        units=[{"id": 40, "type": "harv"}, {"id": 41, "type": "harv"}],
    )
    state.raw["time"]["game_seconds"] = 240  # the army floor is 2,800 and we have none
    agent, _ = advised({}, plan=Plan(buildings=(("proc", 2),), units=()))
    out = orders_of(agent.choose_action(state, CTX))
    assert {"cmd": "build", "item": "proc"} in out


def test_the_army_saves_for_a_tank_instead_of_feeding_the_money_to_more_infantry():
    """Cash 700 buys riflemen every tick, so a 950 tank was never affordable: the vehicle queue sat empty."""
    mine = [{"id": i, "type": "e1", "x": 20, "y": 20} for i in range(8)] + [
        {"id": 40, "type": "harv"}
    ]
    state = make_state(
        tick=5000,
        cash=700,
        units=mine,
        combat=list(range(8)),
        enemy_base=[80, 80],
        avail=["e1", "e3", "3tnk", "harv", "powr"],
        buildings=[{"id": 1, "type": t} for t in ("fact", "powr", "proc", "barr", "weap")],
    )
    agent, _ = advised({}, plan=Plan(buildings=(), units=(("e1", 30),)))
    act = agent.choose_action(state, CTX)
    out = [] if act is WAIT else orders_of(act)
    assert not any(o["cmd"] == "train" and o["item"] in ("e1", "e3") for o in out)
