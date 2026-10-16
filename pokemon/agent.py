"""Quicksilver's Pokémon VGC doubles policy: the engine computes, Jev decides.

``pokemon_vgc_doubles_draft`` is played in three phases, and the policy owns
all three: the snake draft (6 picks each from a shared pool of 18 fixed,
pre-built cards), Team Preview (bring 4 of 6, lead 2) and the doubles battle
itself.

The split is Red Alert's: the deterministic engine does the exact math (team
swing scores, matchup scores, damage formulas) and builds the legal, checked
candidates; **Jev** — the TypeSafe System One tactician — reads the full game
state (rules, both boards, damage ranges, speed order, rosters) and picks
among those candidates. There is no free-text model in this game: Jev answers
a typed ``choice`` question in ~0.5 s, its confidence rides with the answer,
and on any failure or a low-confidence shrug the engine's #1 plays with zero
latency. A model call can never cost a match, and the battle bank (420 s,
forfeited when empty) is never at risk.

The battle runs on Showdown's VGC timer — 90 s at Team Preview, 55 s per
battle decision, and a **7-minute bank for the whole battle**. Running the
bank out forfeits the battle, so thinking time is a finite resource that is
spent, not free.
"""

from __future__ import annotations

import traceback

from altruagent import DecisionContext, GameState
from common import trace
from common.jev import JevClient
from pokemon import (
    actions,
    battle,
    budget,
    draft,
    learn,
    lineup,
    observation,
    pool,
    tactician,
    threat,
)


class PokemonAgent:
    """Per-match Pokémon decision logic.

    One instance per match: the runtime builds a fresh one in each match's own
    process, so state kept here never leaks between matches.
    """

    def __init__(self, pool_path=None, jev_client=None) -> None:
        # Injectable so tests never write the real pool file nor touch the network.
        self._pool_path = pool_path
        self._jev = tactician.Tactician(
            jev_client if jev_client is not None else JevClient.from_env(flag="POKEMON_JEV")
        )
        self._jev_turns = 0
        # What the observation doesn't say: Fake Out freshness, Protect last turn.
        self._memory = threat.Memory()
        # Every card this match has shown: new cards turn up between matches, so the
        # banked file is only the fallback for what this match never showed.
        self._match_cards = pool.MatchCards()
        self._agent_id: str | None = None

    def choose_action(self, state: GameState, context: DecisionContext):
        """Decide, and never raise: a raised error kills this match's worker, so any
        failure in the policy is traced and answered from the plain baseline instead."""
        try:
            return self._decide(state, context)
        except Exception as exc:  # noqa: BLE001 - one bad observation must not forfeit a match
            trace.write("error", where="choose_action", error=repr(exc), traceback=traceback.format_exc())
            return self._baseline(state)

    def _baseline(self, state: GameState):
        """The simplest legal answer for whatever was asked."""
        legal = getattr(state, "legal_actions", None) or []
        if not legal:
            return actions.PASS
        try:
            first = legal[0]
            kind = actions.template_type(first)
            if kind == "select_lineup":
                roster = actions.roster_of(actions.template_of(first))
                bring = roster[:4]
                return {"type": "select_lineup", "bring": bring, "leads": bring[:2]} if len(bring) == 4 else first
            if kind == "doubles_turn":
                return self._baseline_turn(actions.template_of(first))
        except Exception:  # noqa: BLE001
            pass
        return legal[0]

    def _decide(self, state: GameState, context: DecisionContext):
        legal = state.legal_actions
        if not legal:
            # The runner only asks on a real decision and the server always
            # lists something legal for one, so this is defensive only.
            return actions.PASS

        # The card pool is published nowhere else, so bank it the first time we
        # see a draft. The pool is fixed, so one capture makes every later
        # matchup computation offline work.
        self._capture_pool(state)
        self._agent_id = getattr(context, "agent_id", None) or self._agent_id
        self._learn(state)

        first = legal[0]
        kind = actions.template_type(first)
        if kind == "select_lineup":
            return self._lineup(actions.template_of(first), state)
        if kind == "doubles_turn":
            return self._doubles_turn(actions.template_of(first), state)
        if str(first.action_id).startswith("draft_pick:"):
            return self._draft(legal, state, context)
        # Any other enumerated action: the server already filtered this list,
        # so any entry is legal.
        return first

    def _draft(self, legal, state: GameState, context: DecisionContext):
        """Choose from the offered cards by team value and denial, mapped back
        onto the server's own action list, with Jev reading the full draft
        (rules, both rosters, the top cards' swing math) when time allows."""
        obs = observation.observation_of(state)
        top = draft.top_cards(obs, agent_id=context.agent_id)

        if len(top) >= 2 and self._jev.enabled and budget.affords(obs, tactician.JEV_TIMEOUT):
            request = tactician.draft_request(obs, context.agent_id, top)
            label = self._jev.choose(
                state=request.state, instructions=request.instructions, criteria=request.criteria
            )
            if label is not None:
                card_id = top[tactician.INDEX[label]][0]
                for action in legal or []:
                    action_id = getattr(action, "action_id", None)
                    if isinstance(action_id, str) and action_id == f"draft_pick:{card_id}":
                        return action

        chosen = draft.choose_from_actions(obs, agent_id=context.agent_id, legal_actions=legal)
        return chosen if chosen is not None else legal[0]

    def _learn(self, state: GameState, template: dict | None = None) -> None:
        """Take in this state's cards and any species or moves the dex lacks."""
        obs = observation.observation_of(state)
        self._match_cards.observe(obs, self._agent_id)
        learn.from_observation(obs, template)

    def _cards(self) -> list[dict] | None:
        return self._match_cards.with_fallback(pool.load_pool(self._pool_path))

    def _capture_pool(self, state: GameState) -> None:
        """Merge this state's cards into ``pokemon/pool.json``.

        Deliberately silent and total: a failure here must never cost a match,
        and the pool only matters for offline work afterwards.
        """
        try:
            obs = observation.observation_of(state)
            if obs.get("phase") != "draft" and not obs.get("available_cards"):
                return
            pool.capture(state, path=self._pool_path)
        except Exception:  # noqa: BLE001 - never let bookkeeping kill a worker
            return

    # -- Team Preview ----------------------------------------------------

    def _lineup(self, template: dict, state: GameState) -> dict:
        """Bring the four with the best matchups against their six, and lead
        the two best equipped to open — with Jev reading both full sixes and
        the top lineups' matchup math during the 90-second preview window."""
        obs = observation.observation_of(state)
        loaded_pool = self._cards()

        candidates = lineup.generate_candidate_lineups(
            template, obs, pool_cards=loaded_pool, max_candidates=3
        )
        if candidates:
            if (
                len(candidates) >= 2
                and self._jev.enabled
                and budget.affords(obs, tactician.JEV_TIMEOUT)
            ):
                request = tactician.preview_request(obs, candidates)
                label = self._jev.choose(
                    state=request.state,
                    instructions=request.instructions,
                    criteria=request.criteria,
                )
                if label is not None:
                    return candidates[tactician.INDEX[label]][0]
            return candidates[0][0]

        chosen = lineup.choose_lineup(template, obs, pool_cards=loaded_pool)
        if chosen is not None:
            return chosen

        roster = actions.roster_of(template)
        if len(roster) < 4:
            return actions.PASS
        bring = roster[:4]
        return {"type": "select_lineup", "bring": bring, "leads": bring[:2]}

    # -- Doubles turn ----------------------------------------------------

    def _doubles_turn(self, template: dict, state: GameState) -> dict:
        """Pick both slots' actions from the board. The engine ranks the joint
        plans by exact damage; when the ranking is close enough to matter,
        Jev reads the full board (HP, boosts, items, abilities, speed order,
        per-plan damage ranges) and picks. On any failure the engine's #1
        plays; when the engine declines entirely, Jev picks per slot from the
        template's own options.

        Two slots may not switch to the same Pokémon, so a species already
        taken by slot 0 is skipped when choosing slot 1.
        """
        obs = observation.observation_of(state)
        self._learn(state, template)
        loaded_pool = self._cards()
        forced = bool(observation.force_switch(obs))
        self._memory.observe(observation.active_pokemon(obs), observation.turn(obs), forced=forced)
        self._memory.observe_foes(observation.opponent_active_pokemon(obs), observation.turn(obs), forced=forced)

        plan = self._plan_turn(template, obs, loaded_pool)
        self._memory.record(observation.active_pokemon(obs), plan)
        return plan

    def _plan_turn(self, template: dict, obs: dict, loaded_pool) -> dict:
        candidates = battle.generate_candidate_plans(
            template,
            obs,
            pool_cards=loaded_pool,
            max_candidates=len(tactician.LABELS),
            memory=self._memory,
            diverse=True,
        )

        if candidates:
            if (
                len(candidates) >= 2
                and self._jev.enabled
                and self._jev_turns < tactician.MAX_CALLS_PER_MATCH
                and budget.affords(obs, tactician.JEV_TIMEOUT)
            ):
                ctx = battle.make_ctx(obs, pool_cards=loaded_pool, memory=self._memory)
                request = tactician.battle_request(obs, candidates, ctx.ours, ctx.theirs, ctx=ctx)
                label = self._jev.choose(
                    state=request.state,
                    instructions=request.instructions,
                    criteria=request.criteria,
                )
                if label is not None:
                    self._jev_turns += 1
                    return candidates[tactician.INDEX[label]][0]

            # No Jev (or no time for it): execute the top evaluated plan.
            return candidates[0][0]

        chosen = battle.choose_turn(template, obs, pool_cards=loaded_pool, memory=self._memory)
        if chosen is not None:
            return chosen

        # The deterministic evaluator declined (unreadable board). Jev still
        # picks from the template's own options — one question per slot —
        # where a free-text model used to be asked.
        if self._jev.enabled and budget.affords(obs, tactician.JEV_TIMEOUT):
            fallback = tactician.slots_fallback(template, obs, self._jev)
            if fallback is not None:
                return fallback

        return self._baseline_turn(template)

    def _baseline_turn(self, template: dict) -> dict:
        """First usable move per slot, first switch slot 0 didn't take, else pass."""
        slots = actions.slots_of(template)
        if len(slots) != 2:
            # Not a shape we can fill. Passing is refused by the server, but an
            # empty dict would raise a DecisionError and kill the worker, so
            # send the template's own shape back and let the server judge it.
            return {"type": "doubles_turn"}

        taken: set[str] = set()
        action: dict = {"type": "doubles_turn"}
        for index, slot in enumerate(slots):
            number = slot.get("slot", index)
            if number not in (0, 1):
                return {"type": "doubles_turn"}
            pick = self._slot_choice(slot, taken)
            if pick.get("type") == "switch":
                taken.add(pick["species"])
            action[f"slot_{number}"] = pick
        return action

    def _slot_choice(self, slot: dict, taken: set[str]) -> dict:
        """Baseline: first move (preferring a pass if that is all there is),
        else the first switch slot 0 didn't already take, else pass."""
        options = actions.options_of(slot)
        if not options:
            return dict(actions.PASS)
        if all(o.get("type") == "pass" for o in options):
            return dict(actions.PASS)

        for move in actions.moves_of(slot):
            choice = actions.move_choice(move)
            if choice is not None:
                return choice

        for switch in actions.switches_of(slot):
            choice = actions.switch_choice(switch)
            if choice is not None and choice["species"] not in taken:
                return choice

        if actions.has_pass(slot):
            return dict(actions.PASS)
        # Nothing usable in this slot: a pass is the least-bad guess.
        return dict(actions.PASS)


def create_agent():
    return PokemonAgent()
