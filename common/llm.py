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
import re
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
    def from_env(cls, env: Any = None) -> GeminiClient | None:
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
        json_output: bool = False,
        thinking_budget: int = 0,
    ) -> dict:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max(max_output_tokens, 1024),
                # Set thinkingBudget: 0 so internal reasoning tokens never consume
                # the output quota or truncate the real message in Gemini 3.8 Flash
                "thinkingConfig": {"thinkingBudget": thinking_budget},
            },
            # Relax safety thresholds to BLOCK_NONE so competitive game terms
            # ("eliminate", "kill", "accuse", "attack", "faint") in Werewolf and
            # Pokemon are never falsely flagged by standard safety filters.
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }
        if json_output:
            payload["generationConfig"]["responseMimeType"] = "application/json"
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
        json_output: bool = False,
        thinking_budget: int = 0,
    ) -> str | None:
        """One generation, or ``None``. Never raises."""
        if not self.configured or not isinstance(prompt, str) or not prompt.strip():
            return None
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            with httpx.Client(timeout=timeout or self.timeout, transport=self._transport) as client:
                response = client.post(
                    url,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json=self._payload(
                        prompt,
                        system,
                        temperature,
                        max_output_tokens,
                        json_output,
                        thinking_budget,
                    ),
                )
            if response.status_code != 200:
                return None
            return self.parse(response.json())
        except Exception:  # noqa: BLE001 - a model must never break a decision
            return None


# -- choosing among enumerated options --------------------------------------

CHOICE_SYSTEM = (
    "You are choosing actions in a turn-based game. You will be given the "
    "situation and a list of legal option ids. Choose exactly one option id for "
    "each slot you are asked about.\n"
    "Reply with only lines of the form 'slot_0=<id>', one per slot. Use ids "
    "exactly as given. Never invent an id. Do not explain."
)

_SLOT_CHOICE = re.compile(r"slot\s*_?\s*(\d+)\s*[=:]\s*([A-Za-z0-9_:.\-]+)", re.IGNORECASE)


def parse_choices(text: Any, allowed: list[str], slots: int) -> list[str] | None:
    """``slots`` option ids parsed out of a reply, or ``None`` if unusable.

    Tolerant on purpose: models wrap ids in prose, quote them, or number slots
    differently. Every accepted id must appear in ``allowed`` — a choice the
    caller did not offer is never guessed at, because acting on an id the
    server never listed is exactly the failure mode that forfeits a turn.
    """
    if not isinstance(text, str) or not text or not allowed or slots <= 0:
        return None
    permitted = set(allowed)

    found: dict[int, str] = {}
    for index, candidate in _SLOT_CHOICE.findall(text):
        slot = int(index)
        if slot < slots and candidate in permitted and slot not in found:
            found[slot] = candidate

    # Fill any slot the labelled form missed, in order of first mention.
    if len(found) < slots:
        mentioned: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_:.\-]+", text):
            if token in permitted and token not in mentioned:
                mentioned.append(token)
        spare = [token for token in mentioned if token not in found.values()]
        for slot in range(slots):
            if slot not in found and spare:
                found[slot] = spare.pop(0)

    if len(found) != slots:
        return None
    return [found[slot] for slot in range(slots)]


def choose_ids(
    client: Any,
    *,
    instruction: str,
    allowed: list[str],
    slots: int,
    context: str = "",
    timeout: float | None = None,
) -> list[str] | None:
    """Ask the model for one option id per slot, or ``None``. Never raises."""
    if client is None or not allowed or slots <= 0:
        return None
    prompt = f"{instruction}\n\n{context}\n\nLegal option ids:\n" + "\n".join(
        f"- {option}" for option in allowed
    )
    try:
        reply = client.generate(
            prompt,
            system=CHOICE_SYSTEM,
            temperature=0.2,
            max_output_tokens=120,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - a model must never break a decision
        return None
    return parse_choices(reply, allowed, slots)
