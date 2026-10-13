"""The battle-plan playbook (doctrines), its switching, and the battle memory Jev decides on."""

from redalert import doctrine as d
from redalert.intel import Intel
from redalert.plan import Plan


def numbers(**kw):
    base = dict(game_s=300, remaining_s=900, my_army=3000, enemy_army_seen=3000, enemy_near_base=0,
                enemy_army_near_base=0, my_losses=0, my_kills=0, income=40, harvesters=2, refineries=1, towers=1,
                army_floor=4000)
    base.update(kw)
    return d.battle_state(**base)


def test_rules_read_the_situation():
    assert d.fallback(numbers(enemy_army_near_base=3000, enemy_near_base=6)) == "defend"
    assert d.fallback(numbers(my_army=500, my_losses=4000)) == "rebuild"
    assert d.fallback(numbers(my_army=500, my_losses=0, game_s=150, army_floor=1000)) != "rebuild"  # not lost, just early
    assert d.fallback(numbers(my_army=9000, enemy_army_seen=2000)) == "all_in"
    assert d.fallback(numbers(my_army=4000, enemy_army_seen=2500)) == "pressure"
    assert d.fallback(numbers(game_s=200, my_army=3000, army_floor=2000, enemy_army_seen=0)) == "expand"
    assert d.fallback(numbers(game_s=200, my_army=1000, army_floor=2000, enemy_army_seen=0)) == "mass_army"
    assert d.fallback(numbers(remaining_s=100, my_army=2500, enemy_army_seen=3000, my_losses=500)) == "all_in"
    # leaving an emergency takes a clearly better number than entering it
    hover = numbers(my_army=2000, enemy_army_near_base=1300, enemy_near_base=3)
    assert d.fallback(hover) == "mass_army" and d.fallback(hover, current="defend") == "defend"


def test_apply_sets_army_and_money_rules_and_keeps_the_plan_lists():
    plan = Plan(buildings=(("powr", 1),), units=(("3tnk", 5),), army_mode="hold", wave_min=10)
    p = d.apply(d.DOCTRINES["pressure"], plan)
    assert (p.army_mode, p.wave_min, p.army_first) == ("attack", 4, True) and p.units == (("3tnk", 5),)
    p = d.apply(d.DOCTRINES["defend"], plan)
    assert p.army_mode == "hold" and p.units[:2] == (("e3", 6), ("e1", 6)) and ("3tnk", 5) in p.units
    assert d.apply(d.DOCTRINES["all_in"], plan).army_mode == "all_in"


def test_switching_has_hysteresis():
    s = d.DoctrineState()
    s.since = 0.0
    assert not s.update(3.0, "pressure", 0.95, "mass_army", answer_id=1)  # inside the 8 s dwell
    assert s.update(9.0, "pressure", 0.95, "mass_army", answer_id=2) and s.name == "pressure" and s.source == "jev"
    # a middling pick needs a second, different Jev answer
    assert not s.update(20.0, "expand", 0.6, "pressure", answer_id=3)
    assert not s.update(20.2, "expand", 0.6, "pressure", answer_id=3)  # the same answer read again at 5 Hz
    assert s.update(21.0, "expand", 0.6, "pressure", answer_id=4) and s.name == "expand"
    # an emergency switches at once, dwell or not
    assert s.update(22.0, "defend", 0.6, "expand", answer_id=5) and s.name == "defend"
    # an unsure Jev hands over to the rules, which must hold for 2 s
    assert not s.update(40.0, "all_in", 0.3, "mass_army")
    assert s.update(42.5, "all_in", 0.3, "mass_army") and s.name == "mass_army" and s.source == "rules"


def cost(t):
    return {"2tnk": 800, "e1": 100, "harv": 1100}.get(t, 0)


def test_intel_remembers_the_enemy_through_the_fog():
    intel = Intel()
    obs = {"visible_enemies": [{"id": 9, "type": "2tnk", "x": 60, "y": 60, "hp": 80},
                               {"id": 8, "type": "e1", "x": 25, "y": 22, "hp": 100}],
           "visible_enemy_buildings": [{"id": 50, "type": "weap", "x": 80, "y": 80}],
           "units": [{"id": 1, "type": "e3", "x": 20, "y": 20}], "buildings": []}
    intel.update(obs, 0.0)
    assert intel.enemy_army_value(cost) == 900
    assert intel.enemy_near((20, 20), cost, 1.0) == (1, 100.0)
    fog = {"visible_enemies": [], "visible_enemy_buildings": [], "units": [{"id": 1, "type": "e3", "x": 20, "y": 20}]}
    intel.update(fog, 30.0)
    assert intel.enemy_army_value(cost) == 900  # out of sight, not forgotten
    fog["units"].append({"id": 2, "type": "e3", "x": 60, "y": 61})  # we stand where the tank was: it is gone
    intel.update(fog, 31.0)
    assert intel.enemy_army_value(cost) == 100
    intel.update({"visible_enemies": [], "units": [], "buildings": []}, 200.0)
    assert intel.enemy_army_value(cost) == 0 and 50 in intel.buildings  # units fade after 90 s, buildings stay
    s = intel.summary(obs={"units": [{"id": 1, "type": "e3", "x": 20, "y": 20, "hp": 50}], "buildings": []},
                      now=200.0, home=(20, 20), enemy_base=(80, 80), cost_of=cost, idle={1})
    assert s["me"]["army_idle"] == 1 and s["me"]["army_avg_hp_pct"] == 50
    assert s["enemy"]["buildings_known_by_type"] == {"weap": 1} and s["map"]["bases_distance"] == 85
