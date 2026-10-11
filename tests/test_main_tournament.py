"""CLI tests for ``python -m agent --tournament`` / ``--match`` /
``--check-tournament``. The official client and the supervisor loop are
replaced by fakes; no network, no processes. The retired modes (no mode,
``--claim``) are covered by tests/test_main.py.
"""

from __future__ import annotations

import pytest

import agent.__main__ as agent_main
from altruagent.errors import AuthenticationError, ConfigurationError, PlatformError
from altruagent.official import OfficialAgentError
from altruagent.supervisor import TESTING, TOURNAMENT

KEY = "eak_live_" + "cd" * 32


class FakeOfficial:
    def __init__(self, *, auth_error=None, assignments_result=()):
        self.control_url = "https://control.example.test"
        self.auth_error = auth_error
        self.assignments_result = assignments_result
        self.authenticated = False
        self.closed = False

    def authenticate(self):
        if self.auth_error:
            raise self.auth_error
        self.authenticated = True

    def assignments(self):
        if isinstance(self.assignments_result, Exception):
            raise self.assignments_result
        return list(self.assignments_result)

    def close(self):
        self.closed = True


@pytest.fixture
def env(monkeypatch):
    for name in ("ALTRUAGENT_CLAIM_TOKEN", "ALTRUAGENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALTRUAGENT_OFFICIAL_AGENT_KEY", KEY)
    monkeypatch.setenv("ALTRUAGENT_CONTROL_URL", "https://control.example.test")


def _install(monkeypatch, official=None, *, supervisor=None):
    official = official or FakeOfficial()
    calls = []
    monkeypatch.setattr(agent_main, "OfficialAgentClient", lambda *a, **k: official)

    def fake_supervisor(client, *, agent_spec, kinds):
        calls.append((client, agent_spec, kinds))
        if supervisor:
            supervisor()

    monkeypatch.setattr(agent_main, "run_tournament_forever", fake_supervisor)
    return official, calls


# -- --tournament -----------------------------------------------------------------------


def test_tournament_mode_authenticates_and_runs_the_tournament_supervisor(env, monkeypatch, capsys):
    official, calls = _install(monkeypatch)

    assert agent_main.main(["--tournament"]) == 0

    assert official.authenticated and official.closed
    assert calls == [(official, "agent.agent:create_agent", frozenset({TOURNAMENT}))]
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["Connected with your Official Agent Key.",
                       "Playing your tournament games only. Add --match to also play your test matches.",
                       "Waiting for your next game... (Press Ctrl+C to stop.)"]


@pytest.mark.parametrize(
    "argv, kinds, playing",
    [
        (["--tournament"], {TOURNAMENT},
         "Playing your tournament games only. Add --match to also play your test matches."),
        (["--match"], {TESTING},
         "Playing your test matches only. Add --tournament to also play your tournament games."),
        (["--tournament", "--match"], {TESTING, TOURNAMENT}, "Playing your test matches and tournament games."),
        (["--match", "--tournament"], {TESTING, TOURNAMENT}, "Playing your test matches and tournament games."),
    ],
)
def test_each_flag_chooses_which_kinds_of_game_one_process_plays(env, monkeypatch, capsys, argv, kinds, playing):
    official, calls = _install(monkeypatch)

    assert agent_main.main(argv) == 0

    assert calls == [(official, "agent.agent:create_agent", frozenset(kinds))]
    assert capsys.readouterr().out.splitlines()[1] == playing


def _run_with_the_real_supervisor(monkeypatch, argv):
    """``main(argv)`` with the real supervisor loop (3 polls, fake processes)
    over one test match and one tournament game without a ``context``."""
    from functools import partial

    from altruagent import supervisor
    from altruagent.models import OfficialAssignment
    from test_supervisor import FakeProcessFactory

    rows = [
        OfficialAssignment(match_id="m-1", seat_id="s-1", game_type="werewolf", context="testing"),
        OfficialAssignment(match_id="m-2", seat_id="s-2", game_type="werewolf",
                           tournament_name="Fall Cup", round_label="Swiss round 1 of 3"),
    ]
    factory = FakeProcessFactory()
    monkeypatch.setattr(agent_main, "OfficialAgentClient", lambda *a, **k: FakeOfficial(assignments_result=rows))
    monkeypatch.setattr(agent_main, "run_tournament_forever", partial(
        supervisor.run_tournament_forever, sleep=lambda s: None, process_factory=factory, max_iterations=3))
    assert agent_main.main(argv) == 0
    return [p.args[0].seat_id for p in factory.processes]


TOURNAMENT_WARNING = ("You have a tournament game waiting (Fall Cup, Swiss round 1 of 3): run with --tournament "
                      "to play it — it counts as a loss if your agent doesn't connect within the window.")


def test_match_mode_end_to_end_plays_the_test_match_and_warns_once_about_the_tournament_game(env, monkeypatch, capsys):
    assert _run_with_the_real_supervisor(monkeypatch, ["--match"]) == ["s-1"]

    out = capsys.readouterr().out.splitlines()
    assert out.count(TOURNAMENT_WARNING) == 1
    assert "Test match waiting: run with --match to play it" not in out


def test_tournament_mode_end_to_end_plays_the_tournament_game_and_notes_the_test_match(env, monkeypatch, capsys):
    assert _run_with_the_real_supervisor(monkeypatch, ["--tournament"]) == ["s-2"]

    out = capsys.readouterr().out.splitlines()
    assert out.count("Test match waiting: run with --match to play it") == 1
    assert TOURNAMENT_WARNING not in out


def test_both_flags_end_to_end_play_both_games_in_one_process(env, monkeypatch, capsys):
    assert _run_with_the_real_supervisor(monkeypatch, ["--tournament", "--match"]) == ["s-1", "s-2"]

    out = capsys.readouterr().out
    assert "You have a tournament game waiting" not in out and "Test match waiting" not in out


def test_tournament_mode_passes_agent_override(env, monkeypatch):
    _, calls = _install(monkeypatch)

    assert agent_main.main(["--tournament", "--agent", "examples.llm_agent"]) == 0
    assert calls[0][1] == "examples.llm_agent"


def test_match_mode_passes_agent_override(env, monkeypatch):
    _, calls = _install(monkeypatch)

    assert agent_main.main(["--match", "--agent", "examples.llm_agent"]) == 0
    assert calls == [(calls[0][0], "examples.llm_agent", frozenset({TESTING}))]


def test_match_mode_rejects_bad_agent_spec_before_connecting(env, monkeypatch, capsys):
    official, calls = _install(monkeypatch)

    assert agent_main.main(["--match", "--agent", "no_such_module_xyz"]) == 1
    assert not official.authenticated and calls == []


def test_tournament_mode_rejects_bad_agent_spec_before_connecting(env, monkeypatch, capsys):
    official, calls = _install(monkeypatch)

    assert agent_main.main(["--tournament", "--agent", "no_such_module_xyz"]) == 1
    assert "Could not find agent module" in capsys.readouterr().out
    assert not official.authenticated and calls == []


def test_tournament_mode_configuration_error(env, monkeypatch, capsys):
    def raise_config(*a, **k):
        raise ConfigurationError("ALTRUAGENT_OFFICIAL_AGENT_KEY is not set.")

    monkeypatch.setattr(agent_main, "OfficialAgentClient", raise_config)

    assert agent_main.main(["--tournament"]) == 1
    assert "ALTRUAGENT_OFFICIAL_AGENT_KEY is not set" in capsys.readouterr().out


def test_tournament_mode_authentication_failure(env, monkeypatch, capsys):
    official, calls = _install(monkeypatch, FakeOfficial(auth_error=OfficialAgentError("The Official Agent Key was not accepted.")))

    assert agent_main.main(["--tournament"]) == 1
    assert "Could not connect with your Official Agent Key" in capsys.readouterr().out
    assert calls == [] and official.closed


def test_tournament_mode_with_updated_rules_keeps_running_and_says_what_to_do(env, monkeypatch, capsys):
    # Was: "Could not connect with your Official Agent Key: ..." and exit 1.
    # The key is fine; the runtime waits for the contestant to accept the
    # updated Official Rules, then plays.
    from functools import partial

    from altruagent import supervisor

    rules = OfficialAgentError("Accept the updated Official Rules on your dashboard (x).", status_code=403,
                               error_code="registration_incomplete")
    rules.next_step = "rules"
    official = FakeOfficial(auth_error=rules, assignments_result=rules)
    monkeypatch.setattr(agent_main, "OfficialAgentClient", lambda *a, **k: official)
    monkeypatch.setattr(agent_main, "run_tournament_forever", partial(
        supervisor.run_tournament_forever, sleep=lambda s: None, max_iterations=3))

    assert agent_main.main(["--tournament"]) == 0

    out = capsys.readouterr().out.splitlines()
    assert "Could not connect with your Official Agent Key" not in "\n".join(out)
    assert "Connected with your Official Agent Key." not in out
    assert out[:3] == ["Playing your tournament games only. Add --match to also play your test matches.",
                       "Waiting for your next game... (Press Ctrl+C to stop.)",
                       "Accept the updated Official Rules on your dashboard "
                       "(https://platform.altruagent-game.com/tournament/dashboard); I'll keep trying."]


def test_tournament_mode_ctrl_c(env, monkeypatch, capsys):
    def interrupt():
        raise KeyboardInterrupt

    official, _ = _install(monkeypatch, supervisor=interrupt)

    assert agent_main.main(["--tournament"]) == 0
    assert "Stopped." in capsys.readouterr().out and official.closed


def test_tournament_mode_fatal_auth_error_during_run(env, monkeypatch, capsys):
    def revoked():
        raise AuthenticationError("The Official Agent Key was not accepted.")

    _install(monkeypatch, supervisor=revoked)

    assert agent_main.main(["--tournament"]) == 1
    assert "unrecoverable error" in capsys.readouterr().out


def test_tournament_mode_ignores_a_stray_claim_token_env(env, monkeypatch):
    monkeypatch.setenv("ALTRUAGENT_CLAIM_TOKEN", "seatclaim_should_be_ignored")
    _, calls = _install(monkeypatch)

    assert agent_main.main(["--tournament"]) == 0 and len(calls) == 1


def test_tournament_mode_ignores_a_stray_platform_api_key(env, monkeypatch):
    monkeypatch.setenv("ALTRUAGENT_API_KEY", "sk_agent_old_platform_key")
    _, calls = _install(monkeypatch)

    assert agent_main.main(["--tournament"]) == 0 and len(calls) == 1


@pytest.mark.parametrize("argv", [["--tournament", "--claim", "seatclaim_x"],
                                  ["--tournament", "--check-tournament"],
                                  ["--check-tournament", "--claim", "seatclaim_x"],
                                  ["--match", "--check-tournament"],
                                  ["--check-tournament", "--match"],
                                  ["--match", "--claim", "seatclaim_x"],
                                  ["--tournament", "--match", "--check-tournament"]])
def test_modes_are_mutually_exclusive(env, monkeypatch, argv):
    _, calls = _install(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        agent_main.main(argv)
    assert exc_info.value.code == 2 and calls == []


# -- --check-tournament ---------------------------------------------------------------------------


def _check(monkeypatch, capsys, official, argv=("--check-tournament",)):
    _install(monkeypatch, official)
    monkeypatch.setattr(agent_main, "_mark", lambda symbol, fallback: symbol)
    code = agent_main.main(list(argv))
    return code, capsys.readouterr().out


def test_check_tournament_success(env, monkeypatch, capsys):
    code, out = _check(monkeypatch, capsys, FakeOfficial(assignments_result=[]))

    assert code == 0
    assert out.splitlines() == [
        "✓ Control plane reachable",
        "✓ Official Agent Key accepted",
        "✓ Tournament agent authenticated",
        "✓ Assignment discovery available (0 active assignment(s))",
        "✓ Agent ready (agent.agent:create_agent)",
        "✓ Ready to play Testing and tournament games",
    ]
    assert KEY not in out


def test_check_tournament_never_acquires_or_renews_a_seat_lease(env, monkeypatch, capsys):
    import httpx

    from altruagent.official import OfficialAgentClient

    paths = []

    def backend(request):
        paths.append(request.url.path)
        if request.url.path == "/tournament/agent/authenticate":
            return httpx.Response(200, json={"access_token": "sess-1", "expires_at": "x"})
        return httpx.Response(200, json={"assignments": [
            {"match_id": "m", "seat_id": "s", "game_type": "werewolf", "seat_position": 3, "seat_count": 7,
             "match_status": "in_progress", "seat_status": "running"}]})

    monkeypatch.setattr(agent_main, "OfficialAgentClient",
                        lambda: OfficialAgentClient(load_env_file=False, transport=httpx.MockTransport(backend)))
    monkeypatch.setattr(agent_main, "_mark", lambda symbol, fallback: symbol)

    assert agent_main.main(["--check-tournament"]) == 0
    assert paths == ["/tournament/agent/authenticate", "/tournament/agent/assignments"]
    assert "(1 active assignment(s))" in capsys.readouterr().out


def test_check_tournament_uses_agent_override(env, monkeypatch, capsys):
    code, out = _check(monkeypatch, capsys, FakeOfficial(), argv=("--check-tournament", "--agent", "examples.smoke_agent"))

    assert code == 0 and "✓ Agent ready (examples.smoke_agent)" in out


@pytest.mark.parametrize(
    "official, last_line",
    [
        (FakeOfficial(auth_error=PlatformError("Could not reach", status_code=None)), "✗ Control plane not reachable"),
        (FakeOfficial(auth_error=OfficialAgentError("not accepted", status_code=401)), "✗ Official Agent Key rejected"),
        (FakeOfficial(auth_error=OfficialAgentError("boom", status_code=500)), "✗ Official agent authentication failed"),
        (FakeOfficial(auth_error=OfficialAgentError(
            "temporary", status_code=401, error_code="invalid_official_agent_key",
            detail="Failed to mint agent session: Request rate limit reached")),
         "✗ Official agent authentication failed"),
        (FakeOfficial(auth_error=OfficialAgentError("slow down", status_code=429, error_code="rate_limited")),
         "✗ Official agent authentication failed"),
        (FakeOfficial(assignments_result=AuthenticationError("session rejected")), "✗ Tournament agent session was not accepted"),
        (FakeOfficial(assignments_result=PlatformError("down", status_code=500)), "✗ Assignment discovery failed"),
    ],
)
def test_check_tournament_failures_are_nonzero_and_specific(env, monkeypatch, capsys, official, last_line):
    code, out = _check(monkeypatch, capsys, official)

    assert code == 1
    assert out.splitlines()[-1].startswith(last_line)
    assert "Ready to play" not in out


def test_check_tournament_agent_factory_failure(env, monkeypatch, capsys):
    code, out = _check(monkeypatch, capsys, FakeOfficial(),
                       argv=("--check-tournament", "--agent", "examples.basic_agent:missing"))

    assert code == 1 and out.splitlines()[-1].startswith("✗ Agent examples.basic_agent:missing could not be created")


def test_check_tournament_configuration_failure(env, monkeypatch, capsys):
    monkeypatch.setattr(agent_main, "_mark", lambda symbol, fallback: symbol)
    monkeypatch.setenv("ALTRUAGENT_OFFICIAL_AGENT_KEY", "sk_agent_wrong_kind")

    assert agent_main.main(["--check-tournament"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("✗ ALTRUAGENT_OFFICIAL_AGENT_KEY doesn't look like an Official Agent Key")
    assert "sk_agent_wrong_kind" not in out


def test_mark_falls_back_when_console_cannot_encode(monkeypatch):
    import io
    import sys

    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    assert agent_main._mark("✓", "[ok]") == "[ok]"
