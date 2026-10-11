"""Shared, safety-hardened helper for temporarily replacing the contestant's
``agent/agent.py`` with fixed test content, and restoring it afterward.

Used by both ``smoke_game.py --concurrent`` and ``acceptance_test.py`` to run
real ``python -m agent`` / spawned-worker processes against deterministic
decision logic, without ever touching ``altruagent.worker``/``supervisor``/
``runner`` (see ``smoke_game.py``'s own docstring for why the filesystem is
the only channel that reliably reaches a spawned worker on this project).

This module exists because the original per-script versions of this logic
could destroy a contestant's real ``agent/agent.py``:

- On startup, if a backup already existed (from a previous run that didn't
  restore cleanly), the old code silently assumed the backup was "the real
  file" and overwrote ``agent.py`` with it — including if the contestant had
  since noticed the stub and written fresh work into ``agent.py``.
- The install step wrote the backup and the stub as two separate,
  non-atomic file writes, so a crash mid-write could leave either file
  truncated.
- ``smoke_game.py``'s own restore was gated on a flag set *after* the swap,
  so a Ctrl-C in the (tiny but real) gap between the swap and the flag
  update would skip the restore entirely, leaving ``agent.py`` as a
  permanent always-``RESIGN`` stub.

Fix: never guess. If a backup is already present, refuse to touch either
file and tell the user to resolve it by hand. Otherwise, install and restore
are both single atomic file replacements (temp file + ``os.replace``), and
restoring is idempotent and safe to call unconditionally — see
``temporary_agent`` below, which ties install/restore to a ``with`` block so
there is no separate flag to get out of sync with reality.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

AGENT_FILE = Path(__file__).resolve().parent.parent / "agent" / "agent.py"
AGENT_FILE_BACKUP = AGENT_FILE.with_suffix(".py.smoke_test_backup")


class AgentFileSwapError(RuntimeError):
    """Raised when it is not safe to install or restore automatically.
    Message is always safe to print — never include secrets when raising
    this."""


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as a single atomic replace, so a crash
    mid-write can never leave ``path`` truncated or half-written."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def install_temporary_agent(source_text: str) -> None:
    """Back up the real ``agent/agent.py`` and replace its contents with
    ``source_text``.

    Raises ``AgentFileSwapError`` — touching neither file — if a backup is
    already present. That means a previous run of one of these test scripts
    did not restore cleanly (or another run is using it right now); this
    deliberately does not guess which of ``agent.py``/the backup holds the
    contestant's real code, since guessing wrong silently destroys it.
    """
    if AGENT_FILE_BACKUP.exists():
        raise AgentFileSwapError(
            f"{AGENT_FILE_BACKUP} already exists, which means a previous run "
            "of a test script (smoke_game.py --concurrent or "
            "acceptance_test.py) did not restore agent/agent.py cleanly, or "
            "another run is using it right now. Refusing to guess which of "
            f"{AGENT_FILE.name} or {AGENT_FILE_BACKUP.name} holds your real "
            f"agent code — compare the two files yourself, make sure "
            f"{AGENT_FILE.name} has the version you want to keep, then "
            f"delete {AGENT_FILE_BACKUP.name} before re-running."
        )

    real_text = AGENT_FILE.read_text(encoding="utf-8")
    _atomic_write(AGENT_FILE_BACKUP, real_text)
    _atomic_write(AGENT_FILE, source_text)


def restore_real_agent() -> None:
    """Undo ``install_temporary_agent``. Idempotent and safe to call even if
    installation never happened (no backup present -> no-op) — always call
    this unconditionally from a ``finally`` block or via ``temporary_agent``
    below, never gated behind a separate "did the swap happen" flag.
    """
    if AGENT_FILE_BACKUP.exists():
        os.replace(AGENT_FILE_BACKUP, AGENT_FILE)


@contextmanager
def temporary_agent(source_text: str) -> Iterator[None]:
    """Install ``source_text`` as ``agent/agent.py`` for the duration of the
    ``with`` block, then restore the real file — including if the block
    raises, is interrupted (``KeyboardInterrupt``), or exits normally.

    If ``install_temporary_agent`` itself raises (pre-existing backup), the
    ``with`` block's body never runs and nothing is restored — correct,
    since nothing was swapped in that case.
    """
    install_temporary_agent(source_text)
    try:
        yield
    finally:
        restore_real_agent()
