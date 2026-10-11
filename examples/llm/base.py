"""Shared types for the example LLM agent and its structured-action adapters.

A ``Choice`` is one decision described in model-agnostic terms: what the
model is shown, the JSON schema it must answer in (built from the server's
current options), how an answer becomes a real GameAPI action (raising
``InvalidChoice`` when it doesn't fit the server's constraints), and the
deterministic action to play if the model never gives a valid answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from altruagent import GameState, LegalAction


class InvalidChoice(ValueError):
    """The model's answer doesn't match the server's legal options."""


class UnsupportedStructuredAction(ValueError):
    """GameAPI offered a structured action template no adapter knows how to
    fill in. Raised instead of guessing a payload.
    """


@dataclass
class Choice:
    kind: str
    prompt: dict
    schema: dict
    build: Callable[[dict], Any]
    fallback: Callable[[], Any]


# A structured-action adapter: given the template's legal action and the
# state it came from, describe the decision as a Choice.
AdapterFactory = Callable[[LegalAction, GameState], Choice]


def object_schema(properties: dict, *, reasoning: bool = True) -> dict:
    """A strict JSON-schema object: every property required, nothing extra."""
    if reasoning:
        properties = {**properties, "reasoning_summary": {"type": "string"}}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
