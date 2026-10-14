"""Simulation and fuzz tests for the Pokémon policy.

``scripts/pokemon_sim.py`` plays full VGC-doubles battles on the server's own
observation/action shapes and rejects any illegal action, so these double as a
legality check over thousands of real decisions. Win-rate floors are set well
below the measured rates (≈83% vs always-first-move, ≈93% vs random) to be
robust to seed noise while still failing if the policy regresses to coin-flips.
"""

from __future__ import annotations

import copy
import importlib.util
import random
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pokemon_sim", Path(__file__).resolve().parent.parent / "scripts" / "pokemon_sim.py"
)
sim = importlib.util.module_from_spec(SPEC)
import sys  # noqa: E402

sys.modules["pokemon_sim"] = sim
SPEC.loader.exec_module(sim)


def test_it_never_plays_an_illegal_action_across_many_battles():
    # SimError would propagate: a missing target, a duplicate switch, a double pass...
    for rival in (lambda o, t: sim.first_legal(), lambda o, t: sim.random_legal(3)):
        a, b, d = sim.versus(sim.quicksilver, rival, 25, seed=11)
        assert a + b + d == 50


def test_it_beats_the_always_first_move_baseline():
    a, b, d = sim.versus(sim.quicksilver, lambda o, t: sim.first_legal(), 40, seed=21)
    assert a / (a + b + d) >= 0.70


def test_it_crushes_random_play():
    a, b, d = sim.versus(sim.quicksilver, lambda o, t: sim.random_legal(4), 40, seed=22)
    assert a / (a + b + d) >= 0.80


def test_team_preview_runs_and_beats_a_random_four():
    a, b, d = sim.preview_versus(sim.bring_quicksilver, sim.bring_random(random.Random(1)), 30, seed=5)
    assert a + b + d == 60
    assert a / (a + b + d) >= 0.45  # measured ≈60%; a floor against it getting worse than chance


def test_a_decision_is_fast_enough_for_the_clock():
    rng = random.Random(9)
    teams = sim.draw_teams(rng)
    battle = sim.Battle(teams, rng)
    policy = sim.quicksilver(teams[0], teams[1])
    obs, template = battle.observation(0), battle.template(0)
    policy(obs, template)  # warm caches
    started = time.perf_counter()
    for _ in range(5):
        policy(obs, template)
    assert (time.perf_counter() - started) / 5 < 0.5  # the decision clock is 50 s


# -- fuzz: the policy is total on garbage -------------------------------------------

JUNK = [None, "x", 7, -3, 2.5, [], {}, [None], [[]], True, {"a": 1}, "Protect"]


def _mutate(value, rng, depth=0):
    """A copy of ``value`` with a few random nodes replaced by junk or removed."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            roll = rng.random()
            if roll < 0.04:
                continue
            out[k] = rng.choice(JUNK) if roll < 0.08 else _mutate(v, rng, depth + 1)
        return out
    if isinstance(value, list):
        out = [rng.choice(JUNK) if rng.random() < 0.05 else _mutate(v, rng, depth + 1) for v in value]
        if out and rng.random() < 0.04:
            out.pop(rng.randrange(len(out)))
        return out
    return value


def test_the_battle_policy_is_total_on_corrupted_observations():
    rng = random.Random(77)
    teams = sim.draw_teams(rng)
    battle = sim.Battle(teams, rng)
    policy = sim.quicksilver(teams[0], teams[1])
    base_obs, base_template = battle.observation(0), battle.template(0)
    for i in range(400):
        obs = _mutate(copy.deepcopy(base_obs), rng)
        template = _mutate(copy.deepcopy(base_template), rng) if i % 2 else base_template
        out = policy(obs, template)  # must not raise
        assert isinstance(out, dict)


def test_team_preview_is_total_on_corrupted_rosters():
    from pokemon import lineup

    rng = random.Random(78)
    six, theirs = rng.sample(sim.POOL, 6), rng.sample(sim.POOL, 6)
    obs = {"phase": "team_preview",
           "your_roster": [{"species": c["species"], "types": ["Water"], "base_stats": {}} for c in six],
           "opponent_roster": [{"species": c["species"]} for c in theirs]}
    template = {"type": "select_lineup", "roster": [c["species"] for c in six]}
    for _ in range(300):
        out = lineup.choose_lineup(_mutate(copy.deepcopy(template), rng), _mutate(copy.deepcopy(obs), rng),
                                   pool_cards=_mutate(copy.deepcopy(sim.POOL), rng))
        assert out is None or isinstance(out, dict)
        lineup.generate_candidate_lineups(_mutate(copy.deepcopy(template), rng), _mutate(copy.deepcopy(obs), rng))


def test_choose_action_never_raises_and_falls_back_to_the_baseline():
    from types import SimpleNamespace

    from pokemon.agent import PokemonAgent

    agent = PokemonAgent(pool_path=Path("/nonexistent/pool.json"), jev_client=None)

    def boom(*args, **kwargs):
        raise RuntimeError("internal bug")

    agent._doubles_turn = boom
    template = {"type": "doubles_turn", "slots": [
        {"slot": 0, "options": [{"type": "move", "move_id": "Tackle", "targets": [1, 2]}]},
        {"slot": 1, "options": [{"type": "move", "move_id": "Growl", "targets": [0]}]}]}
    legal = [SimpleNamespace(action_id="doubles_turn", input={"action": template})]
    state = SimpleNamespace(legal_actions=legal, raw={}, observation="", phase="moving")
    out = agent.choose_action(state, SimpleNamespace(agent_id="me"))
    assert out["type"] == "doubles_turn" and out["slot_0"]["move_id"] == "Tackle"
