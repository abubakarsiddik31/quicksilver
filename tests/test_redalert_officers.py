"""Officers and missions: roster, odds-based advance, breaking off, flanks, raids, screens, the base guard."""

from __future__ import annotations

import math

from redalert import officers as o

COST = {"e1": 100, "e3": 300, "3tnk": 950, "dog": 200, "harv": 1100, "pbox": 400}
HOME, ENEMY = (20.0, 20.0), (80.0, 80.0)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def fleet():
    clock = Clock()
    f = o.Fleet(
        lambda t: COST.get(t, 100),
        clock,
        non_combat=frozenset({"harv", "mcv"}),
        tower_types=frozenset({"pbox"}),
    )
    return f, clock


def troops(n, at, kind="e1", start=1):
    return [{"id": start + i, "type": kind, "x": at[0] + (i % 3), "y": at[1] + i // 3} for i in range(n)]


def enemies(n, at, kind="3tnk", start=900):
    return [{"id": start + i, "type": kind, "x": at[0] + i, "y": at[1]} for i in range(n)]


def step(f, missions, units, foes=(), buildings=(), towers=None):
    obs = {"visible_enemies": list(foes), "visible_enemy_buildings": list(buildings)}
    return f.step(missions, units, obs, home=HOME, enemy_base=ENEMY, own_towers=towers or [])


def cmds(res, cmd):
    return [x for x in res.orders if x["cmd"] == cmd]


# -- parsing ---------------------------------------------------------------------------------------------------


def test_missions_are_validated_and_bad_rows_dropped():
    parsed = o.parse_missions(
        [
            {"id": "main", "kind": "strike", "share": 0.7, "engage_ratio": 99},
            {"kind": "TELEPORT"},
            "garbage",
            {"kind": "flank", "side": "up", "target": [10, 12], "count": 3},
            {"kind": "raid", "target": {"x": 1}},
        ]
    )
    assert [m.kind for m in parsed] == [o.STRIKE, o.FLANK, o.RAID]
    assert parsed[0].engage_ratio == 10.0  # clamped
    assert parsed[1].side == "alternate" and parsed[1].target == [10, 12] and parsed[1].count == 3
    assert parsed[2].target == "nearest"
    assert o.parse_missions(None) == () and o.parse_missions("x") == ()


# -- roster ----------------------------------------------------------------------------------------------------


def test_the_base_always_keeps_a_guard_and_strike_takes_the_rest():
    f, _ = fleet()
    units = troops(10, (20, 20))
    f.assign((o.Mission("main", o.STRIKE),), units, HOME)
    assert len(f.members["line"]) == o.MIN_GUARD  # no line was asked for: the invariant adds one
    assert len(f.members["main"]) == 8
    assert f.members["line"].isdisjoint(f.members["main"])


def test_shares_set_the_group_sizes_and_membership_sticks():
    f, _ = fleet()
    units = troops(20, (20, 20))
    ms = (o.Mission("line", o.HOLD_LINE, share=0.25), o.Mission("raid", o.RAID, count=3), o.Mission("main", o.STRIKE))
    f.assign(ms, units, HOME)
    first = {k: set(v) for k, v in f.members.items()}
    assert (len(first["line"]), len(first["raid"]), len(first["main"])) == (5, 3, 12)
    f.assign(ms, [u for u in units if u["id"] != 7], HOME)  # one dies, nobody else changes officer
    moved = sum(len((first[k] - {7}) - f.members[k]) for k in first)
    assert moved <= 1  # the line refills one soldier from the main body; nobody else changes officer


# -- strike ----------------------------------------------------------------------------------------------------


def test_strike_musters_then_advances_together_when_the_odds_at_the_target_are_good():
    f, clock = fleet()
    m = (o.Mission("main", o.STRIKE, target="base", engage_ratio=1.0),)
    far = troops(8, (22, 22))
    res = step(f, m, far)
    assert f.state["main"] in ("mustering", "waiting")
    assert not any(x["cmd"] == "attack_move" for x in res.orders if x["units"] and x.get("to") == [80, 80])
    staged = f.members["main"] | f.members.get("line", set())
    staging = o._toward(HOME, ENEMY, min(0.2, 24.0 / math.dist(HOME, ENEMY)))
    at_stage = troops(12, (staging[0], staging[1]), kind="3tnk")  # strong enough to advance on a target it has not seen
    clock.t += 5
    res = step(f, m, at_stage)
    assert f.state["main"] == "advancing"
    assert any(x["cmd"] == "attack_move" and x["to"] == [80, 80] for x in res.orders)
    assert staged  # keep the linter honest


def test_strike_waits_when_the_enemy_at_the_target_outweighs_it():
    f, _ = fleet()
    m = (o.Mission("main", o.STRIKE, target="base", engage_ratio=1.2),)
    staging = o._toward(HOME, ENEMY, min(0.2, 24.0 / math.dist(HOME, ENEMY)))
    units = troops(8, (staging[0], staging[1]))
    res = step(f, m, units, foes=enemies(6, (78, 78)))  # 5,700 of tanks at the target against ~600
    assert f.state["main"] != "advancing"
    assert not any(x["cmd"] == "attack_move" and x["to"] == [80, 80] for x in res.orders)


def test_strike_in_contact_and_outweighed_breaks_off_fighting_and_reports_it():
    f, clock = fleet()
    m = (o.Mission("main", o.STRIKE, target="base", abort_ratio=0.7, regroup_s=30),)
    units = troops(8, (50, 50))
    res = step(f, m, units, foes=enemies(5, (54, 50)), towers=[(25.0, 25.0)])
    assert f.state["main"] == "broken"
    assert "officer_broken:main" in res.events
    back = [x for x in res.orders if x["cmd"] in ("attack_move", "move") and x["to"] == [25, 25]]
    assert back and back[0]["cmd"] == "attack_move"  # a retreat in contact keeps shooting
    # it stays back for the regroup time even once the enemy is out of sight
    clock.t += 10
    res = step(f, m, troops(8, (30, 30)), towers=[(25.0, 25.0)])
    assert f.state["main"] == "broken"
    clock.t += 30
    res = step(f, m, troops(8, (24, 24)))
    assert f.state["main"] != "broken"


def test_strike_in_contact_and_winning_keeps_attacking():
    f, _ = fleet()
    m = (o.Mission("main", o.STRIKE, target="base"),)
    units = troops(10, (50, 50), kind="3tnk")
    res = step(f, m, units, foes=enemies(2, (54, 50), kind="e1"))
    assert f.state["main"] == "engaged"
    assert not res.events or "officer_broken:main" not in res.events
    assert any(x["cmd"] == "attack_move" and x["to"] == [80, 80] for x in res.orders)


# -- the line --------------------------------------------------------------------------------------------------


def test_the_line_meets_a_threat_near_the_base_and_otherwise_holds_on_defend_stance():
    f, clock = fleet()
    m = (o.Mission("line", o.HOLD_LINE, share=1.0),)
    res = step(f, m, troops(6, (20, 20)))
    assert any(x["cmd"] == "set_stance" and x["stance"] == "defend" for x in res.orders)
    assert f.state["line"] == "holding"
    clock.t += 5
    res = step(f, m, troops(6, (20, 20)), foes=enemies(2, (30, 30), kind="e1"))
    assert f.state["line"] == "engaged"
    assert any(x["cmd"] == "attack_move" for x in res.orders)


# -- flank, raid, screen, fall back ----------------------------------------------------------------------------


def test_a_flank_swings_wide_instead_of_walking_into_the_target():
    f, _ = fleet()
    m = (o.Mission("line", o.HOLD_LINE, count=3), o.Mission("flank", o.FLANK, count=3, target=[70, 70], side="left"))
    res = step(f, m, troops(9, (20, 20)))
    wide = [x for x in res.orders if x["cmd"] == "attack_move"]
    assert wide, res.orders
    assert wide[0]["to"] != [70, 70]
    assert math.dist(wide[0]["to"], (70, 70)) > 8
    assert f.state["flank"] == "swinging"


def test_a_raid_goes_straight_for_the_harvesters_and_withdraws_from_a_bigger_force():
    f, _ = fleet()
    m = (o.Mission("line", o.HOLD_LINE, count=3), o.Mission("raid", o.RAID, count=3, target="harvesters"))
    harv = [{"id": 800, "type": "harv", "x": 60, "y": 60}]
    res = step(f, m, troops(9, (20, 20), kind="dog"), foes=harv)
    assert any(x["cmd"] == "attack_move" and x["to"] == [60, 60] for x in res.orders)
    near = troops(9, (58, 58), kind="dog")
    res = step(f, m, near, foes=harv + enemies(4, (60, 58)))
    assert f.state["raid"] == "withdrawing" and "officer_withdrew:raid" in res.events


def test_a_screen_gives_way_and_reports_when_the_enemy_comes():
    f, _ = fleet()
    m = (o.Mission("line", o.HOLD_LINE, count=3), o.Mission("eyes", o.SCREEN, count=2, leash=20))
    res = step(f, m, troops(8, (20, 20)))
    assert f.state["eyes"] in ("watching",)
    res = step(f, m, troops(8, (36, 36)), foes=enemies(3, (38, 38)))
    assert f.state["eyes"] == "falling_back" and "officer_spotted:eyes" in res.events


def test_fall_back_in_contact_fights_and_out_of_contact_walks():
    f, _ = fleet()
    m = (o.Mission("back", o.FALL_BACK, share=1.0, target="towers"),)
    res = step(f, m, troops(5, (40, 40)), foes=enemies(2, (42, 40)), towers=[(25.0, 25.0)])
    assert cmds(res, "attack_move")
    res = step(f, m, troops(5, (40, 40)), towers=[(25.0, 25.0)])
    f._last_order.clear()
    res = step(f, m, troops(5, (40, 40)), towers=[(25.0, 25.0)])
    assert cmds(res, "move") and not cmds(res, "attack_move")


def test_reports_describe_every_officer_for_the_strategist():
    f, _ = fleet()
    m = (o.Mission("line", o.HOLD_LINE, count=3), o.Mission("main", o.STRIKE))
    res = step(f, m, troops(9, (20, 20)))
    js = f.summary(res.reports)
    assert {r["id"] for r in js} == {"line", "main"}
    assert all({"soldiers", "state", "enemy_near_value", "odds"} <= set(r) for r in js)
