"""One JSON-lines trace file per match, so a finished game can be analysed.

Every match runs in its own process, so each gets its own file:
``traces/ra-<unix time>-<pid>.jsonl`` (override the folder with ``RA_TRACE_DIR``,
switch off with ``RA_TRACE=0``). Lines are ``{"t": <unix time>, "kind": ..., ...}``:

- ``plan``    each LLM plan: why it was consulted, latency, the full plan
- ``jev``     each Jev answer: values, confidences, latency
- ``orders``  each batch sent: tick and a compact summary of its orders
- ``refused`` orders the server refused, with the reason
- ``state``   a sample every ~5 s: cash, power, units, army, enemies seen (by type), queue, doctrine, roles
- ``directive`` the General's orders whenever they change: enemy stance, garrison / raider / scout quotas

Writing never raises and never blocks the decision loop for more than a short append.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_path: str | None = None
_failed = False


def _target() -> str | None:
    global _path, _failed
    if _failed or os.environ.get("RA_TRACE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if _path is None:
        folder = os.environ.get("RA_TRACE_DIR") or "traces"
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            _failed = True
            return None
        _path = os.path.join(folder, f"ra-{int(time.time())}-{os.getpid()}.jsonl")
    return _path


def write(kind: str, **data: Any) -> None:
    path = _target()
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"t": round(time.time(), 2), "kind": kind, **data},
                    default=str,
                    separators=(",", ":"),
                )
                + "\n"
            )
    except OSError:
        pass


def current_path() -> str | None:
    return _path
