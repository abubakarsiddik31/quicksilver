"""Log lines reach the terminal in one write each.

Several worker processes share the runtime's terminal (or a log file). With
unbuffered output (PYTHONUNBUFFERED=1, ``python -u``), ``print`` writes the
text and the newline separately, so two processes could glue their lines
together. Each line, newline included, must be a single write.
"""

from __future__ import annotations

import sys
import types

from altruagent.console import print_line
from altruagent.supervisor import TournamentState, run_tournament_once
from altruagent.worker import TournamentWorkerInput, run_tournament_worker
from test_supervisor import FakeProcessFactory
from test_tournament import EXEC, SPEC, FakeOfficial, FakeWorkerOfficial, a


class RecordingStream:
    """A stdout that records every write call separately."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushes = 0
        self.encoding = "utf-8"

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        self.flushes += 1


def _every_write_is_one_whole_line(writes: list[str]) -> bool:
    return bool(writes) and all(w.endswith("\n") and w.count("\n") == 1 and w != "\n" for w in writes)


def test_print_line_writes_the_text_and_its_newline_at_once_and_flushes(monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr(sys, "stdout", stream)

    print_line("Match finished.")

    assert stream.writes == ["Match finished.\n"]
    assert stream.flushes >= 1


def test_the_worker_writes_each_line_at_once(monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr(sys, "stdout", stream)

    def fake_run_game(game, context, contestant):
        return types.SimpleNamespace(termination_reason="normal", returns={"synthetic-1": 1.0})

    run_tournament_worker(TournamentWorkerInput("seat-1", "match-1", "werewolf", SPEC, EXEC),
                          official_factory=lambda: FakeWorkerOfficial(None), run_game_fn=fake_run_game)

    assert len(stream.writes) >= 2  # starting, finished
    assert _every_write_is_one_whole_line(stream.writes)


def test_the_supervisor_writes_each_line_at_once_by_default(monkeypatch):
    stream = RecordingStream()
    monkeypatch.setattr(sys, "stdout", stream)
    factory = FakeProcessFactory()
    state = TournamentState()

    run_tournament_once(FakeOfficial([a("seat-1")]), state, agent_spec=SPEC, now=lambda: 1.0,
                        process_factory=factory)
    factory.processes[0].finish(0)
    run_tournament_once(FakeOfficial([]), state, agent_spec=SPEC, now=lambda: 2.0, process_factory=factory)

    assert len(stream.writes) >= 3  # Match assigned, Starting match..., Match finished.
    assert _every_write_is_one_whole_line(stream.writes)
