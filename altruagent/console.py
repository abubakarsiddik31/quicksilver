"""The runtime's terminal output: one log line per write.

The supervisor and every game's worker process share one terminal (or one
log file). ``print`` writes a line's text and its newline separately, so with
unbuffered output (``PYTHONUNBUFFERED=1``, ``python -u``) two processes could
glue their lines together. ``print_line`` writes the text and the newline at
once and flushes, so each line arrives whole and without delay, also when the
output goes to a pipe or a file.
"""

from __future__ import annotations

import sys


def print_line(message: str) -> None:
    """Write ``message`` and a newline to stdout in one write, then flush."""
    stream = sys.stdout
    if stream is None:  # no console at all (e.g. pythonw on Windows)
        return
    stream.write(f"{message}\n")
    stream.flush()
