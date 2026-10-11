"""Import a contestant agent factory from a ``MODULE[:FACTORY]`` spec.

Shared by ``python -m agent``'s ``--agent`` option and the tournament worker
processes (which receive only the spec string and import it themselves, since
a spawned process can't be handed a function object from its parent).
"""

from __future__ import annotations

import importlib
from typing import Callable

DEFAULT_AGENT_SPEC = "agent.agent:create_agent"


def load_agent_factory(spec: str) -> Callable:
    """Import ``MODULE[:FACTORY]`` (``FACTORY`` defaults to ``create_agent``)
    and return the factory without calling it. Raises ``ValueError`` with a
    human-readable message for every way that can go wrong.
    """
    module_name, _, factory_name = spec.partition(":")
    factory_name = factory_name or "create_agent"
    if (
        not module_name
        or not all(part.isidentifier() for part in module_name.split("."))
        or not factory_name.isidentifier()
    ):
        raise ValueError(
            "--agent expects MODULE[:FACTORY] — a dotted module path such as "
            f"examples.messaging_agent or my_agents.v2:build_agent — got {spec!r}."
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            raise ValueError(
                f"Could not find agent module {module_name!r}. Run from the starter's "
                "root directory and pass a dotted module path, not a file path."
            ) from exc
        raise ValueError(f"Importing agent module {module_name!r} failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any import-time failure in contestant code
        raise ValueError(f"Importing agent module {module_name!r} raised {exc!r}.") from exc

    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(
            f"{module_name}:{factory_name} is missing or not callable. It must be a "
            "function returning your decision logic (a function, or an object "
            "exposing choose_action(state, context))."
        )
    return factory
