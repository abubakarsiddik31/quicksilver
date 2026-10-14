"""One JSON-lines trace file per match, for every game, so a finished game can be analysed.

Every match runs in its own process, so each gets its own file:
``traces/<game>-<unix time>-<pid>.jsonl`` (folder: ``TRACE_DIR``, off: ``TRACE=0``).
Lines are ``{"t": <unix time>, "kind": ..., ...}``:

- ``decision``  each ``choose_action``: the full unmodified server state (``raw`` — both
                sides' visible info, battle log, rosters, chat), the legal actions, what
                we played and how long it took
- ``message``   each ``choose_message`` (Werewolf): the window we saw and what we said
- ``result``    the server's answer to each move, or the refusal
- ``jev``       each Jev consultation (``kind="jev"``): the question, candidates, answer

Red Alert keeps its own richer trace (``redalert/trace.py``). Writing never raises.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any

_path: str | None = None
_failed = False
_game = "game"


def set_game(game_type: str | None) -> None:
    global _game
    _game = "".join(c if c.isalnum() or c in "-_" else "_" for c in (game_type or "game"))


def _target() -> str | None:
    global _path, _failed
    if _failed or os.environ.get("TRACE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if _path is None:
        folder = os.environ.get("TRACE_DIR") or "traces"
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            _failed = True
            return None
        _path = os.path.join(folder, f"{_game}-{int(time.time())}-{os.getpid()}.jsonl")
    return _path


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def write(kind: str, **data: Any) -> None:
    path = _target()
    if path is None:
        return
    try:
        line = json.dumps(
            {"t": round(time.time(), 2), "kind": kind, **data},
            default=_plain,
            separators=(",", ":"),
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 - tracing must never cost a match
        pass


def state_fields(state: Any) -> dict:
    """What a state tells us, for a ``decision`` line."""
    return {
        "phase": getattr(state, "phase", None),
        "move_count": getattr(state, "move_count", None),
        "legal": list(getattr(state, "legal_actions", None) or []),
        "new_messages": list(getattr(state, "new_messages", None) or []),
        "raw": getattr(state, "raw", None),
    }
