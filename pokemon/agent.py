"""Quicksilver's Pokémon VGC doubles draft policy.

``pokemon_vgc_doubles_draft`` is played in three phases, and the policy owns
all three: the snake draft (6 picks each from a shared pool of 18 fixed,
pre-built cards), Team Preview (bring 4 of 6, lead 2) and the doubles battle
itself.

The battle runs on Showdown's VGC timer — 90 s at Team Preview, 55 s per
battle decision, and a **7-minute bank for the whole battle**. Running the
bank out forfeits the battle, so thinking time is a finite resource that is
spent, not free.

.. warning::

    **This is currently a BASELINE, not the competitive policy.** It always
    makes a *legal* choice and will play a whole match to the end, which makes
    it a correct safety net and a working end-to-end target for test matches —
    but it has no strategy. It will lose to any opponent that has one.

    The competitive policy replaces ``choose_action`` here: precomputed
    draft/matchup tables over the fixed 18-card pool, and a search over the
    doubles state. See ``SUMMARY.md``.
"""

from __future__ import annotations

from altruagent import DecisionContext, GameState
from common.llm import GeminiClient
from pokemon import actions, advisor, battle, budget, draft, lineup, observation, pool


class PokemonAgent:
    """Per-match Pokémon decision logic.

    One instance per match: the runtime builds a fresh one in each match's own
    process, so state kept here never leaks between matches.
    """

    def __init__(self, pool_path=None) -> None:
        # Injectable so tests never write the real pool file.
        self._pool_path = pool_path
        self._llm: GeminiClient | None = None
        self._llm_checked: bool = False
        self._advisor_calls_count: int = 0

    def _client(self) -> GeminiClient | None:
        if not self._llm_checked:
            self._llm = GeminiClient.from_env()
            self._llm_checked = True
        return self._llm

    def choose_action(self, state: GameState, context: DecisionContext):
        legal = state.legal_actions
        if not legal:
            # The runner only asks on a real decision and the server always
            # lists something legal for one, so this is defensive only.
            return actions.PASS

        # The card pool is published nowhere else, so bank it the first time we
        # see a draft. The pool is fixed, so one capture makes every later
        # matchup computation offline work.
        self._capture_pool(state)

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
        onto the server's own action list (a constructed pick is refused)."""
        obs = observation.observation_of(state)
        chosen = draft.choose_from_actions(
            obs, agent_id=context.agent_id, legal_actions=legal
        )
        return chosen if chosen is not None else legal[0]

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
        the two best equipped to open.

        Team Preview is the one decision made with full information about the
        opposing team, so it is worth choosing rather than taking the first
        four. Falls back to the simple baseline if the rosters can't be read.
        """
        obs = observation.observation_of(state)
        chosen = lineup.choose_lineup(
            template, obs, pool_cards=pool.load_pool(self._pool_path)
        )
        if chosen is not None:
            return chosen

        roster = actions.roster_of(template)
        if len(roster) < 4:
            return actions.PASS
        bring = roster[:4]
        return {"type": "select_lineup", "bring": bring, "leads": bring[:2]}

    # -- Doubles turn ----------------------------------------------------

    def _doubles_turn(self, template: dict, state: GameState) -> dict:
        """Pick both slots' actions from the board, optionally consulting the
        tactical advisor on critical 50/50 dilemmas, then fall back to baseline.

        Two slots may not switch to the same Pokémon, so a species already
        taken by slot 0 is skipped when choosing slot 1.
        """
        obs = observation.observation_of(state)
        loaded_pool = pool.load_pool(self._pool_path)

        # 1. Generate top candidate joint turn plans
        candidates = battle.generate_candidate_plans(
            template, obs, pool_cards=loaded_pool, max_candidates=3
        )

        if candidates:
            # Check if this turn is a tactical dilemma warranting advisor review
            client = self._client()
            if (
                client is not None
                and self._advisor_calls_count < advisor.MAX_ADVISOR_CALLS_PER_MATCH
                and budget.affords(obs, advisor.ADVISOR_TIMEOUT + 1.0)
                and battle.is_tactical_dilemma(candidates, obs)
            ):
                self._advisor_calls_count += 1
                adv_pick = advisor.advise(
                    client,
                    obs,
                    candidates,
                    timeout=advisor.ADVISOR_TIMEOUT,
                )
                if adv_pick is not None:
                    return adv_pick

            # Otherwise execute the top mathematically evaluated plan
            return candidates[0][0]

        # 2. If candidates generator found no board, try single turn solver
        chosen = battle.choose_turn(
            template, obs, pool_cards=loaded_pool
        )
        if chosen is not None:
            return chosen

        # 3. When the deterministic evaluator declines, consult the LLM fallback
        # if configured and permitted by the clock budget.
        client = self._client()
        if client is not None and budget.affords(obs, budget.LLM_SECONDS_DECISION):
            model_turn = battle.choose_turn_with_model(
                template, obs, client, timeout=min(8.0, budget.LLM_SECONDS_DECISION)
            )
            if model_turn is not None:
                return model_turn

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
