"""A minimal Gemini client, for the few places a deterministic policy is not enough.

**The design rule, and the reason this is safe.** The policy decides *what* to
say or do; the model only decides *how to phrase it*. A phrased result that
contradicts the intent is discarded, so a confused — or deliberately manipulated
— model cannot change the claim itself. That property matters because Werewolf
discussion is the one place a model in this agent reads text written by
adversaries, and the platform's own documentation warns that opponents will try
prompt injection.

Everything here is total: :meth:`GeminiClient.generate` returns ``None`` on any
failure — no key, a timeout, a refusal, a malformed body — and callers fall back
to their deterministic path. A model call must never be able to cost a match.

Auth, per the Gemini API reference: the key goes either in an ``x-goog-api-key``
header or as a ``?key=`` query parameter, to
``https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.8-flash"

# A short leash. These calls sit inside a game decision, and a slow answer is
# worse than the deterministic fallback that is always available.
DEFAULT_TIMEOUT = 12.0


def _clean(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip().strip('"').strip()
    return stripped or None


class GeminiClient:
    """One model, one endpoint, no retries beyond the transport's own."""

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
        # Injectable so tests exercise the parsing without a network call.
        self._transport = transport

    # -- construction ---------------------------------------------------

    @classmethod
    def from_env(cls, env: Any = None) -> "GeminiClient | None":
        """A client if a key is configured, otherwise ``None``.

        ``GEMINI_API_KEY`` is the documented name; ``GOOGLE_API_KEY`` is a
        common alias, and both are accepted so a working environment does not
        silently do nothing.
        """
        source = os.environ if env is None else env
        key = source.get("GEMINI_API_KEY") or source.get("GOOGLE_API_KEY")
        if not isinstance(key, str) or not key.strip():
            return None
        return cls(
            key.strip(),
            model=source.get("GEMINI_MODEL") or DEFAULT_MODEL,
            base_url=source.get("GEMINI_BASE_URL") or DEFAULT_BASE_URL,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # -- generation -----------------------------------------------------

    def _payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
    ) -> dict:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    @staticmethod
    def parse(body: Any) -> str | None:
        """The text of the first candidate, or ``None``.

        A blocked prompt carries ``promptFeedback.blockReason`` and no usable
        candidate, which is a normal outcome rather than an error.
        """
        if not isinstance(body, dict):
            return None
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        first = candidates[0]
        if not isinstance(first, dict):
            return None
        content = first.get("content")
        if not isinstance(content, dict):
            return None
        parts = content.get("parts")
        if not isinstance(parts, list):
            return None
        texts = [
            part.get("text")
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return _clean("".join(texts))

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.9,
        max_output_tokens: int = 256,
        timeout: float | None = None,
    ) -> str | None:
        """One generation, or ``None``. Never raises."""
        if not self.configured or not isinstance(prompt, str) or not prompt.strip():
            return None
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            with httpx.Client(
                timeout=timeout or self.timeout, transport=self._transport
            ) as client:
                response = client.post(
                    url,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json=self._payload(
                        prompt, system, temperature, max_output_tokens
                    ),
                )
            if response.status_code != 200:
                return None
            return self.parse(response.json())
        except Exception:  # noqa: BLE001 - a model must never break a decision
            return None
