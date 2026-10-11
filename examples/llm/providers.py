"""The model-calling layer for the example LLM agent.

An ``LLMProvider`` has one job: send chat messages and return a JSON object
that matches a given schema. Game logic never talks to a vendor API
directly, so adding another provider (Claude, Gemini, a local model) means
writing one class with ``complete_structured`` and choosing it in
``provider_from_env`` — nothing else changes.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 60.0


class ProviderError(RuntimeError):
    """The model request failed or its reply wasn't a JSON object."""


class LLMProvider(Protocol):
    model: str

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        """Return the model's answer as a dict matching ``schema``, or raise
        ``ProviderError``. ``timeout`` (seconds), when given, bounds this one
        request: a game with a clock (Pokémon) passes what's left of the
        decision's time. A provider without the parameter still works; the
        agent then only limits how many requests it makes.
        """
        ...


class OpenAIProvider:
    """OpenAI Chat Completions with strict structured output, over the
    starter's existing ``httpx`` dependency (no SDK).

    The API key only ever goes into the Authorization header — never into
    ``repr``, logs, or error messages (provider error bodies can echo parts
    of the key, so only the HTTP status is reported).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        # Only a game with a clock passes a timeout; otherwise the client's own applies.
        extra = {"timeout": timeout} if timeout is not None else {}
        try:
            response = self._http.post("/chat/completions", json=body, **extra)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise ProviderError(f"OpenAI request failed (HTTP {response.status_code})")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("OpenAI response had no message") from None
        if message.get("refusal"):
            raise ProviderError("the model refused to answer")
        try:
            data = json.loads(message.get("content") or "")
        except ValueError:
            raise ProviderError("the model's reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("the model's reply was not a JSON object")
        return data


def provider_from_env() -> LLMProvider:
    """``OPENAI_API_KEY`` (required), ``OPENAI_MODEL``, ``OPENAI_BASE_URL``."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your environment or the starter's .env.")
    return OpenAIProvider(
        api_key,
        os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
    )
