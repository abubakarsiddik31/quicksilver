"""Shared, defensive parsing of platform HTTP responses — used by both the
control-plane client (``client.py``) and the auth strategies (``auth.py``),
which can't import each other's module-level helpers without a cycle.
"""

from __future__ import annotations

from typing import Any

import httpx


def _parse_json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {}


def _parse_error_body(response: httpx.Response) -> dict:
    """Best-effort parse of an error response body.

    Tolerates the platform's inconsistent error envelopes: a `next_actions`
    array, a `recovery_action` object instead, a plain `{"error": "..."}`
    with no `detail`, or a non-JSON body (e.g. an upstream gateway error
    page), which is captured as text rather than raised.
    """
    try:
        body = response.json()
    except ValueError:
        return {"error": None, "detail": response.text[:500] or None, "next_action": None}

    if not isinstance(body, dict):
        return {"error": None, "detail": str(body), "next_action": None}

    next_action = body.get("recovery_action")
    if next_action is None:
        next_actions = body.get("next_actions")
        if isinstance(next_actions, list) and next_actions:
            next_action = next_actions[0]

    return {
        "error": body.get("error"),
        "detail": body.get("detail"),
        "next_action": next_action,
    }
