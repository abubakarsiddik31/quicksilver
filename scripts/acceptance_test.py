"""Milestone 5 FINAL ACCEPTANCE TEST — developer/manual tool only. NOT RUN
as part of `pytest` or CI; invoke it by hand, deliberately, against the real
deployed platform.

RETIRED: like scripts/smoke_game.py, this drives the platform-agent APIs
(signup, claim, /competitions), which the platform turned off (HTTP 410). It
no longer runs against the deployed platform; it is kept for reference.

Proves the messaging-enabled contestant workflow works through the actual
contestant-facing entry point, `python -m agent` — never by calling
`run_match`/`run_game`/`run_once_concurrent` directly (those are proven at
the unit level in tests/test_runner.py; this script's whole point is
end-to-end confidence in the real command a contestant actually runs).

Two scenarios, each played out with two *real, separate* `python -m agent`
OS subprocesses (one per player) against one shared `repeated_pd`
competition (messaging enabled by the platform's own default preset — see
Agent_ACP/backend/src/services/competitionPresets.ts):

  A. Both subprocesses run `examples/basic_agent.py`'s content — no
     `choose_message` defined anywhere. Proves the default
     auto-`TERMINATE_MESSAGING` behavior alone is enough to finish a
     messaging-enabled match, with zero contestant messaging code.

  B. Both subprocesses run `examples/messaging_agent.py`'s content — a
     stateful agent that actually calls `SendMessage`. Proves the real
     `send_message` wire path (not just `terminate_messaging`) works
     end-to-end through the real runtime.

Running two independent `python -m agent` processes at once (one per
player) also exercises them concurrently — a weaker, more externally
realistic form of the concurrency Milestone 4C's own smoke test already
proved at the single-agent/multi-match level (`--concurrent` in
scripts/smoke_game.py); this script does not re-prove that internal
mechanism, to keep this test focused on messaging correctness.

Setup mirrors scripts/smoke_game.py: your existing primary agent
(ALTRUAGENT_CONTROL_URL/ALTRUAGENT_API_KEY) plus one temporary signed-up-
and-claimed opponent agent (claiming needs a human/admin bearer token,
prompted via getpass — never echoed, stored, or logged). The temporary
agent is left claimed afterward; there is no safe deletion endpoint.

Contestant file handling: exactly like smoke_game.py's `--concurrent` mode,
this temporarily replaces `agent/agent.py`'s *contents* with one of the two
example files' contents via `scripts/_agent_file_swap.py` (backed up first,
always restored, regardless of outcome or interruption) — this is the one
proven-reliable way to make a real `python -m agent` subprocess run specific,
deterministic contestant logic without touching
altruagent.worker/supervisor/runner. If a leftover backup from a previous,
abnormally-terminated run is found, this refuses to guess which file holds
your real code and asks you to resolve it by hand — see that module's
docstring.

Bounded by design: explicit per-scenario match-completion timeout, explicit
subprocess-startup timeout, and subprocess termination + `.wait()` in a
`finally` block regardless of outcome. Uses `subprocess.Popen`/`.terminate()`/
`.wait(timeout=...)` throughout — Windows-safe (no POSIX signal handling,
no shell=True). Never calls the unauthenticated `/games/{id}/cancel` — on a
timeout, this only stops the subprocesses; the match itself may be left
incomplete on the server (a known platform gap: there is no safe "abandon
this match" endpoint today — see the Milestone 5 completion report's
remaining-platform-work list).

Run (after copying .env.example -> .env and filling in your own agent's
credentials, exactly like scripts/smoke_game.py):

    python scripts/acceptance_test.py

--------------------------------------------------------------------------
Milestone 6 (`--mcp`): MCP-first gameplay migration acceptance
--------------------------------------------------------------------------

    python scripts/acceptance_test.py --mcp

Proves the generic MCP gameplay contract (altruagent.mcp_game.MCPGameSession
+ altruagent.runner) actually works against every currently-registered
adapter in Agent_ACP (confirmed via the earlier read-only audit: exactly two
— "openspiel", covering tic_tac_toe/repeated_pd/avalon, and "pokemon",
covering four game-type variants including pokemon_gen9ou_draft — no other
adapter/game exists in the workspace, so none are invented here):

  A. tic_tac_toe — get_game_state / get_legal_actions / LegalAction /
     play_action through MCPGameSession directly, confirms state_version
     strictly increases across two calls, then resigns (bounded, no need to
     complete a full match to prove the wire mechanics).
  B. repeated_pd — reuses the existing `default`/`messaging` scenarios
     above verbatim: they already exercise MCP transparently (python -m
     agent's production path is MCP unconditionally after this migration,
     with zero scenario-code changes needed), including a real messaging
     phase (send_message/terminate through MCP, then real moves).
  C. avalon — 5 real participants (the primary agent + 4 temporary ones).
     Confirms real MCP state/legal actions for whichever seat's turn it
     actually is, submits one legal action, confirms state_version
     advanced. Does not play to completion (a full 5-player match is out
     of scope for this proof).
  D. pokemon_gen9ou_draft — 2 participants. Confirms real MCP state/
     structured legal actions (`draft_pick:<card_id>`-shaped, not bare
     ints), submits one structured action, confirms state_version advanced.
  E. A full Pokémon battle flow (draft -> teambuild -> moves/switches to
     completion) is NOT implemented here — it needs a reachable Showdown
     websocket (`GAMEAPI_POKEMON_SHOWDOWN_WS` on the gameapi deployment;
     see Agent_ACP/gameapi/docs/pokemon-runtime.md), which this script has
     no way to detect or provision from the contestant side. `run_pokemon()`
     below stops after D and prints exactly this as the environment
     blocker, rather than silently skipping it.

Every check above proves "no REST gameplay fallback" the same structural
way tests/test_no_rest_fallback.py does at the unit level (MCPGameSession
and GameSession share no method names except a zero-arg `resign()`, and
Match.game() only ever returns MCPGameSession) — this script additionally
demonstrates it operationally by only ever calling MCPGameSession methods
for A/C/D, and by python -m agent for B (whose only gameplay path is MCP
after this migration, per altruagent/runner.py).

Setup mirrors the rest of this script: real signup/claim of temporary
agents, human/admin bearer token(s) via getpass (never echoed/logged/
stored), cleanup via resign or natural completion only — never
`/games/{id}/cancel`.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

# Allow running this script directly without having pip-installed the project.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from altruagent.client import AltruAgentClient  # noqa: E402

from _agent_file_swap import AgentFileSwapError, restore_real_agent, temporary_agent  # noqa: E402

NUM_ROUNDS = 3  # small on purpose — bounds how many messaging<->moving cycles this test waits through
HTTP_TIMEOUT_SECONDS = 10.0
COMPETITION_POLL_TIMEOUT_SECONDS = 60.0
COMPETITION_POLL_INTERVAL_SECONDS = 3.0
SUBPROCESS_STARTUP_TIMEOUT_SECONDS = 30.0
MATCH_COMPLETION_TIMEOUT_SECONDS = 180.0
MATCH_POLL_INTERVAL_SECONDS = 3.0
SUBPROCESS_TERMINATE_TIMEOUT_SECONDS = 10.0

EXAMPLES_DIR = REPO_ROOT / "examples"

SCENARIOS = {
    "default": EXAMPLES_DIR / "basic_agent.py",       # no choose_message -> auto-terminate
    "messaging": EXAMPLES_DIR / "messaging_agent.py",  # real SendMessage flow
}


class AcceptanceTestError(RuntimeError):
    """Raised for any acceptance-test failure. Message is always safe to
    print — never include secrets when raising this."""


def _checkpoint(label: str) -> None:
    print(f"[OK] {label}")


def _fail(message: str) -> None:
    raise AcceptanceTestError(message)


def _json_or_fail(response: httpx.Response, what: str) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        detail = None
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("error")
        detail = detail or (response.text[:300] if response.text else None)
        _fail(f"{what} failed (HTTP {response.status_code}): {detail}")
    if not isinstance(body, dict):
        _fail(f"{what} returned a non-JSON or unexpected response body.")
    return body


# -- agent signup/claim/competition setup (same shape as scripts/smoke_game.py) --


def signup_temporary_agent(http: httpx.Client, control_url: str) -> tuple[str, str, str]:
    name = f"acceptance-test-{uuid.uuid4().hex[:10]}"
    response = http.post(
        f"{control_url}/auth/agent/signup",
        json={"name": name, "description": "Milestone 5 acceptance-test opponent (safe to ignore/delete)"},
    )
    body = _json_or_fail(response, "temporary agent signup")
    api_key = body.get("api_key")
    claim_token = body.get("claim_token")
    if not api_key or not claim_token:
        _fail("Signup response did not include both api_key and claim_token.")
    return name, api_key, claim_token


def claim_temporary_agent(http: httpx.Client, control_url: str, claim_token: str, human_token: str) -> None:
    response = http.post(
        f"{control_url}/auth/human/claim",
        json={"claim_token": claim_token},
        headers={"Authorization": f"Bearer {human_token}"},
    )
    _json_or_fail(response, "claiming the temporary agent")


def create_repeated_pd_competition(http: httpx.Client, control_url: str, human_token: str) -> str:
    """POST /admin/competitions/create with preset=repeated_pd.

    The preset (Agent_ACP backend/src/services/competitionPresets.ts) forces
    messaging_enabled=True and max_participants=2 — exactly what this test
    needs, with no messaging_config assembled by hand.
    """
    response = http.post(
        f"{control_url}/admin/competitions/create",
        json={"preset": "repeated_pd", "num_rounds": NUM_ROUNDS},
        headers={"Authorization": f"Bearer {human_token}"},
    )
    body = _json_or_fail(response, "repeated_pd competition creation")
    session_id = body.get("session_id")
    if not session_id:
        _fail("Competition creation response did not include session_id.")
    return session_id


def join_competition(client: AltruAgentClient, session_id: str) -> None:
    client.request("POST", f"/competitions/{session_id}/join")


def create_plain_competition(
    http: httpx.Client, control_url: str, human_token: str, *, game_type: str, max_participants: int
) -> str:
    """POST /admin/competitions/create with a bare game_type — no preset.

    Presets only exist for repeated_pd/avalon's messaging_config (see
    Agent_ACP backend/src/services/competitionPresets.ts); every other
    game_type (tic_tac_toe, the pokemon_* variants) is created this way,
    same as scripts/smoke_game.py's create_competition already does for
    tic_tac_toe.
    """
    response = http.post(
        f"{control_url}/admin/competitions/create",
        json={"game_type": game_type, "max_participants": max_participants},
        headers={"Authorization": f"Bearer {human_token}"},
    )
    body = _json_or_fail(response, f"{game_type} competition creation")
    session_id = body.get("session_id")
    if not session_id:
        _fail(f"{game_type} competition creation response did not include session_id.")
    return session_id


def _poll_until(
    poll: Callable[[], dict],
    is_ready: Callable[[dict], bool],
    *,
    timeout_seconds: float,
    interval_seconds: float,
    timeout_message: Callable[[dict], str],
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = poll()
    while True:
        if is_ready(last):
            return last
        if time.monotonic() >= deadline:
            _fail(timeout_message(last))
        time.sleep(interval_seconds)
        last = poll()


def wait_for_in_progress(poll: Callable[[], dict]) -> dict:
    return _poll_until(
        poll,
        lambda last: last.get("status") == "in_progress",
        timeout_seconds=COMPETITION_POLL_TIMEOUT_SECONDS,
        interval_seconds=COMPETITION_POLL_INTERVAL_SECONDS,
        timeout_message=lambda last: (
            f"Competition did not reach in_progress within "
            f"{COMPETITION_POLL_TIMEOUT_SECONDS:.0f}s (last status: {last.get('status')!r})."
        ),
    )


def wait_for_completed(poll: Callable[[], dict]) -> dict:
    return _poll_until(
        poll,
        lambda last: last.get("status") == "completed",
        timeout_seconds=MATCH_COMPLETION_TIMEOUT_SECONDS,
        interval_seconds=MATCH_POLL_INTERVAL_SECONDS,
        timeout_message=lambda last: (
            f"Match did not reach completed within {MATCH_COMPLETION_TIMEOUT_SECONDS:.0f}s "
            f"(last status: {last.get('status')!r}) — this is a real hard failure, not "
            "cleaned up further: this test never calls /games/{id}/cancel."
        ),
    )


# -- contestant file swap (shared with smoke_game.py --concurrent — see
# scripts/_agent_file_swap.py for the install/restore mechanics and the
# safety rules around a leftover backup from a crashed prior run) -----------


# -- one scenario ------------------------------------------------------------


def run_scenario(
    scenario: str,
    primary: AltruAgentClient,
    primary_api_key: str,
    control_url: str,
    opponent_api_key: str,
) -> None:
    print(f"\n=== Scenario: {scenario} ({SCENARIOS[scenario].name}) ===")

    human_token = getpass.getpass(
        f"[{scenario}] Paste a human/admin bearer token (used only in-memory, never logged): "
    )
    if not human_token:
        _fail("A human/admin bearer token is required to create a competition.")

    http = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    try:
        session_id = create_repeated_pd_competition(http, control_url, human_token)
        _checkpoint(f"created repeated_pd competition {session_id}")

        join_competition(primary, session_id)
        opponent_client = AltruAgentClient(
            control_url=control_url, api_key=opponent_api_key, load_env_file=False
        )
        try:
            join_competition(opponent_client, session_id)
        finally:
            opponent_client.close()
        _checkpoint("both agents joined")

        wait_for_in_progress(lambda: http.get(f"{control_url}/competitions/{session_id}").json())
        _checkpoint("competition in_progress")
    finally:
        http.close()

    try:
        with temporary_agent(SCENARIOS[scenario].read_text(encoding="utf-8")):
            processes: list[subprocess.Popen] = []
            try:
                for label, api_key in (("primary", primary_api_key), ("opponent", opponent_api_key)):
                    env = {
                        **os.environ,
                        "ALTRUAGENT_CONTROL_URL": control_url,
                        "ALTRUAGENT_API_KEY": api_key,
                    }
                    # Inherit this process's stdout/stderr rather than capturing
                    # to a pipe nobody reads: a captured-but-undrained pipe can
                    # fill up and block a chatty child indefinitely, and the
                    # failure message below promises the child's output is
                    # visible "above/below" — which is only true if it was
                    # never captured in the first place.
                    process = subprocess.Popen(
                        [sys.executable, "-m", "agent"],
                        cwd=str(REPO_ROOT),
                        env=env,
                    )
                    processes.append(process)
                    print(f"[{scenario}] started `python -m agent` for {label}, pid={process.pid}")

                # Startup sanity check: both processes must still be alive shortly
                # after launch (an immediate exit means a config/import error).
                time.sleep(min(SUBPROCESS_STARTUP_TIMEOUT_SECONDS, 5.0))
                for process in processes:
                    if process.poll() is not None:
                        _fail(
                            f"`python -m agent` exited immediately (code {process.returncode}) "
                            f"during scenario {scenario!r} — see its output above/below."
                        )

                control_url_client = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
                try:
                    wait_for_completed(
                        lambda: control_url_client.get(f"{control_url}/competitions/{session_id}").json()
                    )
                finally:
                    control_url_client.close()
                _checkpoint(f"scenario {scenario!r} match completed via real python -m agent subprocesses")
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=SUBPROCESS_TERMINATE_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=SUBPROCESS_TERMINATE_TIMEOUT_SECONDS)
    except AgentFileSwapError as exc:
        _fail(str(exc))


# -- Milestone 6: direct MCPGameSession checks (--mcp) -----------------------

CURRENT_ACTOR_POLL_TIMEOUT_SECONDS = 30.0
CURRENT_ACTOR_POLL_INTERVAL_SECONDS = 2.0


def _find_current_actor(clients: list[AltruAgentClient], *, session_id: str, game_server_url: str):
    """Poll every given client's MCPGameSession.get_state() until exactly one
    reports is_current_actor=True, and return (that client's game session,
    its state). Bounded — never loops forever.
    """
    from altruagent.mcp_game import MCPGameSession  # local import: only needed here

    sessions: list[MCPGameSession] = [
        client.mcp_game(session_id=session_id, game_server_url=game_server_url) for client in clients
    ]
    deadline = time.monotonic() + CURRENT_ACTOR_POLL_TIMEOUT_SECONDS
    while True:
        for session in sessions:
            state = session.get_state()
            if state.is_current_actor:
                return session, state
        if time.monotonic() >= deadline:
            _fail(
                f"No participant reported is_current_actor=True for session_id={session_id!r} "
                f"within {CURRENT_ACTOR_POLL_TIMEOUT_SECONDS:.0f}s."
            )
        time.sleep(CURRENT_ACTOR_POLL_INTERVAL_SECONDS)


def run_tic_tac_toe_mcp_check(primary: AltruAgentClient, control_url: str, human_token: str) -> None:
    print("\n=== MCP check A: tic_tac_toe ===")
    http = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    opponent_client = None
    try:
        _, opponent_api_key, claim_token = signup_temporary_agent(http, control_url)
        claim_temporary_agent(http, control_url, claim_token, human_token)
        opponent_client = AltruAgentClient(control_url=control_url, api_key=opponent_api_key, load_env_file=False)
        _checkpoint("temporary tic_tac_toe opponent signed up and claimed")

        session_id = create_plain_competition(
            http, control_url, human_token, game_type="tic_tac_toe", max_participants=2
        )
        join_competition(primary, session_id)
        join_competition(opponent_client, session_id)
        competition = wait_for_in_progress(lambda: http.get(f"{control_url}/competitions/{session_id}").json())
        game_server_url = competition["game_server_url"]
        _checkpoint(f"tic_tac_toe competition in_progress (session_id={session_id})")

        mover_session, state = _find_current_actor(
            [primary, opponent_client], session_id=session_id, game_server_url=game_server_url
        )
        _checkpoint(f"MCP get_game_state: is_current_actor found, phase={state.phase!r}")

        legal = mover_session.get_legal_actions()
        if not legal.get("actions"):
            _fail("get_legal_actions returned no actions for the current actor.")
        from altruagent.models import LegalAction

        actions = [LegalAction.from_dict(a) for a in legal["actions"]]
        _checkpoint(f"MCP get_legal_actions: {[a.action_id for a in actions]}")

        state_version_before = legal["state_version"]
        play_result = mover_session.play_action(action_id=actions[0].action_id, state_version=state_version_before)
        state_version_after = play_result["state_version"]
        if state_version_after <= state_version_before:
            _fail(
                f"state_version did not advance after play_action "
                f"({state_version_before} -> {state_version_after})."
            )
        _checkpoint(f"MCP play_action: state_version {state_version_before} -> {state_version_after}")

        # Bounded cleanup: resign rather than play out a full match — this
        # check's job is proving the MCP wire mechanics, not a real game.
        mover_session.resign()
        wait_for_completed(lambda: http.get(f"{control_url}/competitions/{session_id}").json())
        _checkpoint("tic_tac_toe MCP check complete (resigned to end cleanly)")
    finally:
        if opponent_client is not None:
            opponent_client.close()
        http.close()


def run_avalon_mcp_check(primary: AltruAgentClient, control_url: str, human_token: str) -> None:
    print("\n=== MCP check C: avalon (5 players) ===")
    http = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    opponent_clients: list[AltruAgentClient] = []
    try:
        for _ in range(4):
            _, api_key, claim_token = signup_temporary_agent(http, control_url)
            claim_temporary_agent(http, control_url, claim_token, human_token)
            opponent_clients.append(AltruAgentClient(control_url=control_url, api_key=api_key, load_env_file=False))
        _checkpoint("4 temporary avalon participants signed up and claimed")

        session_id = create_plain_competition(
            http, control_url, human_token, game_type="avalon", max_participants=5
        )
        all_clients = [primary, *opponent_clients]
        for client in all_clients:
            join_competition(client, session_id)
        competition = wait_for_in_progress(lambda: http.get(f"{control_url}/competitions/{session_id}").json())
        game_server_url = competition["game_server_url"]
        _checkpoint(f"avalon competition in_progress (session_id={session_id})")

        mover_session, state = _find_current_actor(
            all_clients, session_id=session_id, game_server_url=game_server_url
        )
        _checkpoint(f"MCP get_game_state: is_current_actor found, phase={state.phase!r}")

        legal = mover_session.get_legal_actions()
        if not legal.get("actions"):
            _fail("get_legal_actions returned no actions for the current actor.")
        state_version_before = legal["state_version"]
        action_id = legal["actions"][0]["action_id"]
        play_result = mover_session.play_action(action_id=action_id, state_version=state_version_before)
        state_version_after = play_result["state_version"]
        if state_version_after <= state_version_before:
            _fail(
                f"state_version did not advance after play_action "
                f"({state_version_before} -> {state_version_after})."
            )
        _checkpoint(
            f"MCP play_action on avalon: submitted action_id={action_id!r}, "
            f"state_version {state_version_before} -> {state_version_after}"
        )
        print(
            "[INFO] Not playing avalon to completion (5-player match — out of "
            "scope for this proof). Match is left in_progress; no /cancel is "
            "called. A human/admin may need to clean this up manually."
        )
    finally:
        for client in opponent_clients:
            client.close()
        http.close()


def run_pokemon_mcp_check(primary: AltruAgentClient, control_url: str, human_token: str) -> None:
    print("\n=== MCP check D: pokemon_gen9ou_draft ===")
    http = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    opponent_client = None
    try:
        _, opponent_api_key, claim_token = signup_temporary_agent(http, control_url)
        claim_temporary_agent(http, control_url, claim_token, human_token)
        opponent_client = AltruAgentClient(control_url=control_url, api_key=opponent_api_key, load_env_file=False)
        _checkpoint("temporary pokemon opponent signed up and claimed")

        session_id = create_plain_competition(
            http, control_url, human_token, game_type="pokemon_gen9ou_draft", max_participants=2
        )
        join_competition(primary, session_id)
        join_competition(opponent_client, session_id)
        competition = wait_for_in_progress(lambda: http.get(f"{control_url}/competitions/{session_id}").json())
        game_server_url = competition["game_server_url"]
        _checkpoint(f"pokemon_gen9ou_draft competition in_progress (session_id={session_id})")

        mover_session, state = _find_current_actor(
            [primary, opponent_client], session_id=session_id, game_server_url=game_server_url
        )
        _checkpoint(f"MCP get_game_state: is_current_actor found, phase={state.phase!r} (expected 'draft')")

        legal = mover_session.get_legal_actions()
        if not legal.get("actions"):
            _fail("get_legal_actions returned no actions for the current draft picker.")
        action_id = legal["actions"][0]["action_id"]
        if not action_id.startswith("draft_pick:"):
            _fail(f"Expected a structured 'draft_pick:<card_id>' action_id, got {action_id!r}.")
        _checkpoint(f"MCP get_legal_actions: structured action_id confirmed ({action_id!r})")

        state_version_before = legal["state_version"]
        play_result = mover_session.play_action(action_id=action_id, state_version=state_version_before)
        state_version_after = play_result["state_version"]
        if state_version_after <= state_version_before:
            _fail(
                f"state_version did not advance after play_action "
                f"({state_version_before} -> {state_version_after})."
            )
        _checkpoint(f"MCP play_action: state_version {state_version_before} -> {state_version_after}")

        print(
            "\n[ENVIRONMENT BLOCKER] Not attempting a full Pokemon battle "
            "(draft -> teambuild -> moves/switches to completion): this "
            "requires a reachable Showdown websocket on the gameapi "
            "deployment (GAMEAPI_POKEMON_SHOWDOWN_WS — see "
            "Agent_ACP/gameapi/docs/pokemon-runtime.md), which this script "
            "has no way to detect or provision from the contestant side. "
            "This match is left in_progress (draft incomplete); no /cancel "
            "is called — a human/admin may need to clean it up manually."
        )
    finally:
        if opponent_client is not None:
            opponent_client.close()
        http.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run the Milestone 6 MCP-first acceptance checks (tic_tac_toe/avalon/pokemon) "
        "instead of the Milestone 5 messaging scenarios.",
    )
    args = parser.parse_args()

    try:
        primary = AltruAgentClient()
    except Exception as exc:  # ConfigurationError
        print(f"Configuration error: {exc}")
        return 1

    control_url = primary.control_url
    # AltruAgentClient() already called load_dotenv(), which populates
    # os.environ from .env if it wasn't already set — read it back the same
    # way rather than reaching into the client's private _api_key attribute.
    primary_api_key = os.environ.get("ALTRUAGENT_API_KEY")
    if not primary_api_key:
        print("Configuration error: ALTRUAGENT_API_KEY is not set.")
        return 1

    if args.mcp:
        human_token = getpass.getpass(
            "Paste a human/admin bearer token (used only in-memory, never logged, "
            "reused for every temporary agent/competition this check creates): "
        )
        if not human_token:
            print("Configuration error: a human/admin bearer token is required.")
            return 1
        try:
            run_tic_tac_toe_mcp_check(primary, control_url, human_token)
            run_avalon_mcp_check(primary, control_url, human_token)
            run_pokemon_mcp_check(primary, control_url, human_token)
            print("\nMILESTONE 6 MCP ACCEPTANCE CHECKS PASSED (A, C, D — see B above for repeated_pd)")
            print(
                "NOTE: check B (repeated_pd, incl. messaging) is proven by running this "
                "script WITHOUT --mcp — those scenarios already exercise MCP unconditionally "
                "after this migration. Full Pokemon battle completion (E) was not attempted; "
                "see the environment-blocker message printed above."
            )
            return 0
        except AcceptanceTestError as exc:
            print(f"\nMCP ACCEPTANCE CHECK FAILED: {exc}")
            return 1
        finally:
            primary.close()

    http = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    try:
        name, opponent_api_key, claim_token = signup_temporary_agent(http, control_url)
        _checkpoint(f"signed up temporary opponent agent '{name}'")
        human_token = getpass.getpass(
            "Paste a human/admin bearer token to claim the temporary opponent "
            "(used only in-memory, never logged): "
        )
        if not human_token:
            _fail("A human/admin bearer token is required to claim the temporary agent.")
        claim_temporary_agent(http, control_url, claim_token, human_token)
        _checkpoint("temporary opponent claimed")
    finally:
        http.close()

    try:
        for scenario in ("default", "messaging"):
            run_scenario(scenario, primary, primary_api_key, control_url, opponent_api_key)
        print("\nMILESTONE 5 ACCEPTANCE TEST PASSED (both scenarios)")
        return 0
    except AcceptanceTestError as exc:
        print(f"\nACCEPTANCE TEST FAILED: {exc}")
        return 1
    finally:
        # Idempotent no-op in the normal case — run_scenario's own
        # temporary_agent(...) context manager already restores agent.py
        # unconditionally. Kept as a harmless extra safety net.
        restore_real_agent()
        primary.close()


if __name__ == "__main__":
    raise SystemExit(main())
