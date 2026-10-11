"""Entry point for `python -m agent`.

One way to run your agent: set `ALTRUAGENT_OFFICIAL_AGENT_KEY` to your
Official Agent Key (from the tournament dashboard) and run

    python -m agent --tournament          # your tournament games
    python -m agent --match               # your test matches (Testing page)
    python -m agent --tournament --match  # both, in one process

It authenticates as your registered self-hosted agent and keeps one worker
process per assigned game of the chosen kind(s) until Ctrl+C
(`altruagent.supervisor.run_tournament_forever`, which filters assignments
on their `context`). Nobody copies an id and nobody claims anything. A game
of the other kind is not played; the runtime says once per game that it is
waiting. `--check-tournament` verifies the connection and the agent without
playing. `--agent MODULE[:FACTORY]` picks a different factory than
`agent.agent:create_agent`.

Retired modes only print a notice pointing to the command above (see
`altruagent.notices`):

- `python -m agent` with no mode: the old platform API-key mode
  (`ALTRUAGENT_API_KEY`), turned off on the platform.
- `--claim seatclaim_...` (and `ALTRUAGENT_CLAIM_TOKEN`): Testing claim
  codes, replaced by the Official Agent Key.

This file is deliberately thin — the assignment loop and worker lifecycle
live in `altruagent.supervisor`, one seat's setup in `altruagent.worker`,
and one game's play loop in `altruagent.runner`.

IMPORTANT (Windows multiprocessing): the `if __name__ == "__main__":` guard
at the bottom of this file is not just style — `multiprocessing`'s `spawn`
start method (required on Windows, used here unconditionally so behavior is
identical everywhere) re-imports this exact module in every worker process.
Without the guard, each freshly-spawned worker would re-run `main()` itself
and spawn further workers recursively.
"""

from __future__ import annotations

import argparse
import os
import sys

from altruagent.agent_loader import DEFAULT_AGENT_SPEC
from altruagent.agent_loader import load_agent_factory as _load_agent_factory
from altruagent.errors import AltruAgentError, AuthenticationError, ConfigurationError
from altruagent.notices import AGENT_GUIDE_URL, CLAIM_CODES_RETIRED_NOTICE, PLATFORM_KEY_RETIRED_NOTICE
from altruagent.official import OfficialAgentClient, OfficialAgentError, is_fatal_auth_error, is_registration_incomplete
from altruagent.runner import _resolve_decision_fn
from altruagent.supervisor import ALL_KINDS, TESTING, TOURNAMENT, WAITING_MESSAGE, run_tournament_forever


CLAIM_TOKEN_ENV = "ALTRUAGENT_CLAIM_TOKEN"


def _retired(notice: str) -> int:
    """Print a retired mode's notice and fail: nothing was run."""
    print(notice)
    return 1


# What the runtime says it plays, by the kinds the flags chose.
PLAYING_MESSAGES = {
    ALL_KINDS: "Playing your test matches and tournament games.",
    frozenset({TOURNAMENT}): "Playing your tournament games only. Add --match to also play your test matches.",
    frozenset({TESTING}): "Playing your test matches only. Add --tournament to also play your tournament games.",
}


def _run_tournament(agent_spec: str, kinds: frozenset[str] = ALL_KINDS) -> int:
    """The runtime: authenticate with the Official Agent Key, then keep one
    worker process per assigned game of the chosen ``kinds`` (test matches,
    tournament games or both) until Ctrl+C.
    """
    try:
        _load_agent_factory(agent_spec)  # validated eagerly; built once per match, in its worker
    except ValueError as exc:
        print(f"Startup error: {exc}")
        return 1

    try:
        official = OfficialAgentClient()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    try:
        try:
            official.authenticate()
        except AltruAgentError as exc:
            if not is_registration_incomplete(exc):
                print(f"Could not connect with your Official Agent Key: {exc}")
                return 1
            # The key is fine; the registration needs the contestant (for
            # example updated Official Rules to accept). The loop below says
            # what to do and keeps trying, so the runtime plays as soon as
            # they have.
        else:
            print("Connected with your Official Agent Key.")
        print(PLAYING_MESSAGES[kinds])
        print(f"{WAITING_MESSAGE} (Press Ctrl+C to stop.)", flush=True)
        run_tournament_forever(official, agent_spec=agent_spec, kinds=kinds)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except AltruAgentError as exc:
        print(f"\nStopped due to an unrecoverable error: {exc}")
        return 1
    finally:
        official.close()


def _mark(symbol: str, fallback: str) -> str:
    """``symbol`` if the console can print it (a redirected Windows console may not)."""
    try:
        symbol.encode(sys.stdout.encoding or "ascii")
        return symbol
    except (UnicodeEncodeError, LookupError):
        return fallback


def _check_tournament(agent_spec: str) -> int:
    """Pre-tournament connection check. Needs no assigned match; never prints
    the key or any token. Returns 0 only if every step passes.
    """
    ok_mark, fail_mark = _mark("✓", "[ok]"), _mark("✗", "[FAIL]")

    def ok(message: str) -> None:
        print(f"{ok_mark} {message}")

    def fail(message: str) -> int:
        print(f"{fail_mark} {message}")
        return 1

    try:
        official = OfficialAgentClient()
    except ConfigurationError as exc:
        return fail(str(exc))

    try:
        try:
            official.authenticate()
        except OfficialAgentError as exc:
            ok("Control plane reachable")
            if exc.status_code in (400, 401) and is_fatal_auth_error(exc):
                return fail(f"Official Agent Key rejected: {exc}")
            return fail(f"Official agent authentication failed: {exc}")
        except AltruAgentError as exc:
            return fail(f"Control plane not reachable at {official.control_url}: {exc}")
        ok("Control plane reachable")
        ok("Official Agent Key accepted")

        try:
            assignments = official.assignments()
        except AuthenticationError as exc:
            return fail(f"Tournament agent session was not accepted: {exc}")
        except AltruAgentError as exc:
            ok("Tournament agent authenticated")
            return fail(f"Assignment discovery failed: {exc}")
        ok("Tournament agent authenticated")
        ok(f"Assignment discovery available ({len(assignments)} active assignment(s))")

        try:
            _resolve_decision_fn(_load_agent_factory(agent_spec)())
        except Exception as exc:  # noqa: BLE001 - contestant code
            return fail(f"Agent {agent_spec} could not be created: {exc}")
        ok(f"Agent ready ({agent_spec})")
        ok("Ready to play Testing and tournament games")
        return 0
    finally:
        official.close()


_DESCRIPTION = """\
Run your AltruAgent agent.

  1. Set ALTRUAGENT_OFFICIAL_AGENT_KEY (in .env) to your Official Agent Key,
     generated on the tournament dashboard's Agent Configuration page.
  2. Run it with --tournament (your tournament games), --match (your test
     matches) or both, and leave it running.

It plays each game with agent/agent.py's create_agent(). While it waits for a
game it uses no AI tokens (it checks every ~10 s). Your agent must be
Self-hosted and your event registration complete.

The agent/agent.py you start with is a placeholder (the first legal move): it
finishes a Werewolf game but can't finish a Pokémon or Red Alert match.
Replace it, or run the LLM example (needs OPENAI_API_KEY in .env)."""

_EPILOG = f"""examples:
  python -m agent --check-tournament                   check your key, connection and agent
  python -m agent --tournament                         play your tournament games (Ctrl+C to stop)
  python -m agent --match                              play your test matches
  python -m agent --tournament --match                 play both in one process
  python -m agent --check-tournament --agent examples.llm_agent
                                                       check the LLM example agent
  python -m agent --match --agent examples.llm_agent   play test matches with it

guide: {AGENT_GUIDE_URL}"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --tournament and --match may be combined (both kinds in one process);
    # each excludes --check-tournament and --claim. --match's exclusions are
    # checked in main(): argparse puts an option in one exclusive group only.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tournament",
        action="store_true",
        help=(
            'play your tournament games (Swiss/bracket games, after you press "Register my agent"); '
            "runs until Ctrl+C"
        ),
    )
    parser.add_argument(
        "--match",
        action="store_true",
        help=(
            'play your test matches (Testing page: matches you create with "Mine (self-hosted)" '
            "or join from Open matches); runs until Ctrl+C"
        ),
    )
    mode.add_argument(
        "--check-tournament",
        action="store_true",
        help="check your Official Agent Key, connection and agent without playing anything",
    )
    # Retired: Testing claim codes. Still parsed so an old command prints the
    # notice instead of an argparse error; hidden from --help.
    mode.add_argument("--claim", nargs="?", const="", metavar="TOKEN", help=argparse.SUPPRESS)
    parser.add_argument(
        "--agent",
        metavar="MODULE[:FACTORY]",
        help=f"agent factory to use (default: {DEFAULT_AGENT_SPEC})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args([] if argv is None else argv)

    if args.match and args.check_tournament:
        parser.error("argument --match: not allowed with argument --check-tournament")
    if args.match and args.claim is not None:
        parser.error("argument --match: not allowed with argument --claim")

    if args.check_tournament:
        return _check_tournament(args.agent or DEFAULT_AGENT_SPEC)
    if args.tournament or args.match:
        kinds = frozenset(kind for kind, chosen in ((TOURNAMENT, args.tournament), (TESTING, args.match)) if chosen)
        return _run_tournament(args.agent or DEFAULT_AGENT_SPEC, kinds)

    if args.claim is not None:
        return _retired(CLAIM_CODES_RETIRED_NOTICE)
    if args.agent is not None:
        parser.error("--agent is only supported together with --tournament, --match or --check-tournament")
    if os.environ.get(CLAIM_TOKEN_ENV):
        return _retired(CLAIM_CODES_RETIRED_NOTICE)
    return _retired(PLATFORM_KEY_RETIRED_NOTICE)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
