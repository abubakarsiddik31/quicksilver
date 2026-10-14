"""A minimal TypeSafe Jev client: fast, typed, calibrated decisions.

Jev is a "System 1" model. It is given a ``state`` and named typed ``questions``
and answers each with probabilities, never free text:

- ``choice``: one option key out of ``criteria`` (a dict of key -> description),
- ``score``: a level on an ordered ``criteria`` list,
- ``noul``: a single probability between 0 and 1.

Every question in one request is answered against the same state, in parallel.
Verified live: ``POST {base}/v1/systemone`` with ``{"model": "jev-latest",
"state": ..., "questions": {...}}`` and ``Authorization: Bearer <key>``.

Like :mod:`common.llm`, this is *total*: :meth:`JevClient.ask` returns ``None`` on
any failure (no key, timeout, 429, a malformed body) and callers fall back to
their deterministic path. A model call must never be able to cost a match.
The key is only ever sent to the configured TypeSafe base URL.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
# Jev answers in ~0.4-0.5 s; a decision that takes longer is already stale.
DEFAULT_TIMEOUT = 2.5


@dataclass(frozen=True)
class Answer:
    type: str
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)


def parse_answers(body: Any) -> dict[str, Answer] | None:
    """The ``answers`` of a response body as :class:`Answer` objects, or ``None``."""
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        return None
    out: dict[str, Answer] = {}
    for name, a in answers.items():
        if not isinstance(a, dict):
            continue
        probs = a.get("probabilities")
        out[str(name)] = Answer(
            type=str(a.get("type")),
            choice=a.get("choice") if isinstance(a.get("choice"), str) else None,
            score=float(a["score"]) if isinstance(a.get("score"), (int, float)) else None,
            noul=float(a["noul"]) if isinstance(a.get("noul"), (int, float)) else None,
            confidence=float(a["confidence"])
            if isinstance(a.get("confidence"), (int, float))
            else None,
            probabilities={
                str(k): float(v) for k, v in probs.items() if isinstance(v, (int, float))
            }
            if isinstance(probs, dict)
            else {},
        )
    return out


class JevClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @classmethod
    def from_env(cls, env: Any = None, *, flag: str = "RA_JEV") -> JevClient | None:
        """A client if ``TYPESAFE_API_KEY`` is set and ``flag`` is not switched off.

        Each game owns its kill switch: Red Alert reads ``RA_JEV``, Pokémon
        passes ``flag="POKEMON_JEV"`` — one game losing Jev must not take the
        other's with it.
        """
        source = os.environ if env is None else env
        if str(source.get(flag, "1")).strip().lower() in ("0", "false", "no", "off"):
            return None
        key = source.get("TYPESAFE_API_KEY")
        if not isinstance(key, str) or not key.strip():
            return None
        return cls(
            key.strip(),
            model=source.get("JEV_MODEL") or DEFAULT_MODEL,
            base_url=source.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL,
        )

    def _http(self) -> httpx.Client:
        # One keep-alive connection: a fresh TLS handshake per call would cost
        # more than Jev's own latency.
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout, transport=self._transport)
            return self._client

    def ask(
        self, state: Any, questions: dict[str, dict[str, Any]], *, timeout: float | None = None
    ) -> dict[str, Answer] | None:
        """One request, all questions answered against ``state``; ``None`` on any failure."""
        if not self.api_key or not questions:
            return None
        try:
            response = self._http().post(
                f"{self.base_url}/v1/systemone",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "state": state, "questions": questions},
                timeout=timeout or self.timeout,
            )
            if response.status_code != 200:
                self.last_error = f"HTTP {response.status_code}"
                return None
            answers = parse_answers(response.json())
            self.last_error = None if answers else "no answers in response"
            return answers
        except Exception as exc:  # noqa: BLE001 - a model must never break a decision
            self.last_error = type(exc).__name__
            return None
