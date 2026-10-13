"""The strategy "plan" the LLM writes and the fast harness executes.

A plan is *standing orders*, not a single move: what structures and units to
want, how the army should behave, optional one-off raw orders, and when the
model wants to be consulted next. The harness re-applies it to every fresh
state (about 5 a second), so the model's latency never stalls the base.

Everything here is validation. The model's text is untrusted data: every field
is checked against a closed set, a regex or a range, and anything unusable falls
back to the previous plan's value. Raw orders are sanitised again against the
*live* state at the moment they are sent (:func:`sanitize_order`), because the
ids and cells the model saw are seconds old.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

ARMY_MODES = ("hold", "attack", "all_in")
NAMED_TARGETS = ("nearest", "base", "building")
ITEM_RE = re.compile(r"^[a-z0-9_]{2,8}$")

MAX_WANTS = 24
MAX_RAW_ORDERS = 8
MAX_COUNT = 40
REVIEW_MIN_S, REVIEW_MAX_S = 4, 60

# cmd -> (required fields, optional fields). Mirrors the platform's order schema.
ORDER_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack_move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack": (frozenset({"units"}), frozenset({"target", "to", "queued"})),
    "stop": (frozenset({"units"}), frozenset()),
    "deploy": (frozenset({"units"}), frozenset()),
    "unload": (frozenset({"units"}), frozenset()),
    "guard": (frozenset({"units", "target"}), frozenset({"queued"})),
    "harvest": (frozenset({"units"}), frozenset({"to", "queued"})),
    "set_stance": (frozenset({"units", "stance"}), frozenset()),
    "build": (frozenset({"item"}), frozenset()),
    "train": (frozenset({"item"}), frozenset({"count"})),
    "place": (frozenset({"item"}), frozenset({"at"})),
    "cancel": (frozenset({"item"}), frozenset()),
    "sell": (frozenset({"building"}), frozenset()),
    "repair": (frozenset({"building"}), frozenset({"on"})),
    "power_toggle": (frozenset({"building"}), frozenset({"on"})),
    "set_primary": (frozenset({"building"}), frozenset()),
    "rally": (frozenset({"building", "to"}), frozenset()),
}
STANCES = ("hold_fire", "return_fire", "defend", "attack_anything")
UNIT_ORDERS = {"move", "attack_move", "attack", "stop", "deploy", "unload", "guard", "harvest", "set_stance"}
BUILDING_ORDERS = {"sell", "repair", "power_toggle", "set_primary", "rally"}
ITEM_ORDERS = {"build", "train", "place", "cancel"}


@dataclass(frozen=True)
class Plan:
    assessment: str = ""
    # Priority-ordered desired TOTALS (owned + queued). The first entry still
    # below its count and currently buildable is built next.
    buildings: tuple[tuple[str, int], ...] = ()
    units: tuple[tuple[str, int], ...] = ()
    army_mode: str = "attack"
    wave_min: int = 8
    target: Any = "nearest"  # "nearest" | "base" | "building" | (x, y)
    gather: tuple[int, int] | None = None
    army_first: bool = False  # True: don't hold back cash for wanted structures
    orders: tuple[dict, ...] = ()  # one-off raw orders, run once per plan
    review_in_s: int = 15  # when the model wants to be consulted again

    def as_json(self) -> dict[str, Any]:
        return {
            "assessment": self.assessment,
            "buildings": [{"item": i, "count": c} for i, c in self.buildings],
            "units": [{"item": i, "count": c} for i, c in self.units],
            "army": {
                "mode": self.army_mode,
                "wave_min": self.wave_min,
                "target": list(self.target) if isinstance(self.target, tuple) else self.target,
                "gather": list(self.gather) if self.gather else None,
            },
            "army_first": self.army_first,
            "orders": list(self.orders),
            "review_in_s": self.review_in_s,
        }


# The opening every plan starts from until the model's first answer lands, and
# the fallback when no model is configured. "barracks" resolves to this
# faction's tent (Allies) or barr (Soviets).
DEFAULT_PLAN = Plan(
    assessment="default opening: power, refinery, barracks, war factory, then army",
    buildings=(
        ("powr", 1), ("proc", 1), ("barracks", 1), ("weap", 1),
        ("powr", 2), ("proc", 2),
    ),
    units=(
        ("e1", 4), ("harv", 2), ("3tnk", 8), ("2tnk", 8), ("1tnk", 8),
        ("ftrk", 6), ("apc", 6), ("jeep", 3), ("e1", 12),
    ),
)


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(low, min(high, int(value)))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _cell(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_int(v) for v in value):
        return int(value[0]), int(value[1])
    return None


def _wants(value: Any, default: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        return default
    out: list[tuple[str, int]] = []
    for row in value[:MAX_WANTS]:
        if not isinstance(row, dict):
            continue
        item = str(row.get("item", "")).lower().strip()
        count = row.get("count")
        if ITEM_RE.match(item) and _is_int(count):
            out.append((item, max(0, min(MAX_COUNT, count))))
    return tuple(out)


def _shape_ok(order: Any) -> bool:
    """Static validity of a raw order: known cmd, required fields, no unknown ones."""
    if not isinstance(order, dict):
        return False
    cmd = order.get("cmd")
    if cmd not in ORDER_FIELDS:
        return False
    required, optional = ORDER_FIELDS[cmd]
    fields = set(order) - {"cmd"}
    if not required <= fields or not fields <= required | optional:
        return False
    if "units" in order and not (
        isinstance(order["units"], list) and order["units"] and all(_is_int(u) for u in order["units"])
    ):
        return False
    if "to" in order and _cell(order["to"]) is None:
        return False
    if "at" in order and _cell(order["at"]) is None:
        return False
    if "stance" in order and order["stance"] not in STANCES:
        return False
    if cmd in ITEM_ORDERS and not ITEM_RE.match(str(order.get("item", "")).lower()):
        return False
    if cmd == "attack" and ("target" in order) == ("to" in order):
        return False  # exactly one of target / to
    return True


def parse_plan(text: Any, base: Plan = DEFAULT_PLAN) -> Plan | None:
    """A validated plan from a model reply, or ``None`` if there is no usable JSON object."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    army = data.get("army") if isinstance(data.get("army"), dict) else {}
    mode = str(army.get("mode", base.army_mode)).lower()
    target: Any = base.target
    raw_target = army.get("target", target)
    if isinstance(raw_target, str) and raw_target.lower() in NAMED_TARGETS:
        target = raw_target.lower()
    elif _cell(raw_target) is not None:
        target = _cell(raw_target)
    gather = _cell(army["gather"]) if "gather" in army else base.gather
    note = data.get("assessment")
    orders = tuple(o for o in (data.get("orders") or [])[:MAX_RAW_ORDERS] if _shape_ok(o)) if isinstance(
        data.get("orders"), list
    ) else ()
    army_first = data.get("army_first")
    return Plan(
        assessment=" ".join(note.split())[:400] if isinstance(note, str) else "",
        buildings=_wants(data.get("buildings"), base.buildings),
        units=_wants(data.get("units"), base.units),
        army_mode=mode if mode in ARMY_MODES else base.army_mode,
        wave_min=_clamp(army.get("wave_min"), 3, 30, base.wave_min),
        target=target,
        gather=gather,
        army_first=army_first if isinstance(army_first, bool) else base.army_first,
        orders=orders,
        review_in_s=_clamp(data.get("review_in_s"), REVIEW_MIN_S, REVIEW_MAX_S, base.review_in_s),
    )


def sanitize_order(
    order: dict[str, Any],
    unit_ids: set[int],
    building_ids: set[int],
    enemy_ids: set[int],
    bounds: tuple[int, int, int, int] | None,
    avail: set[str],
) -> dict[str, Any] | None:
    """A raw order checked against the live state, with dead ids dropped; ``None`` if nothing is left."""
    if not _shape_ok(order):
        return None
    out = dict(order)
    cmd = out["cmd"]
    for key in ("to", "at"):
        if key in out and bounds is not None:
            x, y = out[key]
            bx, by, bw, bh = bounds
            if not (bx <= x < bx + bw and by <= y < by + bh):
                return None
    if "units" in out:
        # attack/stop/set_stance/guard may also name own buildings (defenses).
        pool = unit_ids | (building_ids if cmd in {"attack", "stop", "set_stance", "guard"} else set())
        out["units"] = [u for u in out["units"] if u in pool][:50]
        if not out["units"]:
            return None
    if "building" in out and out["building"] not in building_ids:
        return None
    if cmd == "attack" and "target" in out and out["target"] not in enemy_ids:
        return None
    if cmd == "guard" and out["target"] not in (unit_ids | building_ids):
        return None
    if cmd in {"build", "train"}:
        if str(out["item"]).lower() not in avail:
            return None
        if cmd == "train":
            out["count"] = _clamp(out.get("count"), 1, 10, 1)
    return out


def with_orders(plan: Plan, orders: tuple[dict, ...]) -> Plan:
    return replace(plan, orders=orders)
