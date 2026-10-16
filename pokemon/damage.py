"""Gen 9 damage calculation for VGC doubles.

The policy trusts these numbers to choose moves, so the design rule here is:
**compute what we can compute exactly, and say plainly what we could not.**
Every result carries an ``unmodelled`` tuple naming anything recognised as
potentially damage-relevant that was not applied. A silently wrong number is
far worse than an admitted gap, because nothing downstream can detect it.

Format: ``gen9vgc2025regi`` — level 50, doubles, **no Terastallization**. Tera is
deliberately not implemented; there is no Tera action in this format.

The formula (Gen 5+ shape, as used by Showdown)::

    base = floor(floor(floor(2·level/5 + 2) · power · A / D) / 50) + 2
    damage = base · modifiers · roll,  roll ∈ [0.85, 1.00] over 16 values

Precision note: the real game runs the modifier chain in integer arithmetic
with ``pokeRound`` (round-half-down). This implementation uses floats and floors
at the end, so a damage figure can differ from Showdown's by about one point.
That is immaterial for move choice (which only compares options) but it means
these numbers are not a byte-exact replay of the sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pokemon import dex
from pokemon import stats as stats_mod

# Type-boosting items: 1.2x to moves of one type. Showdown's item data records
# the hook (``onBasePowerPriority``) but not which type it applies to, so the
# mapping has to live here. Plates are 1.2x as well.
TYPE_BOOST_ITEMS: dict[str, str] = {
    "Mystic Water": "Water",
    "Charcoal": "Fire",
    "Miracle Seed": "Grass",
    "Magnet": "Electric",
    "Never-Melt Ice": "Ice",
    "Black Belt": "Fighting",
    "Black Glasses": "Dark",
    "Sharp Beak": "Flying",
    "Poison Barb": "Poison",
    "Soft Sand": "Ground",
    "Hard Stone": "Rock",
    "Silver Powder": "Bug",
    "Spell Tag": "Ghost",
    "Metal Coat": "Steel",
    "Twisted Spoon": "Psychic",
    "Dragon Fang": "Dragon",
    "Silk Scarf": "Normal",
    "Fairy Feather": "Fairy",
    "Draco Plate": "Dragon",
    "Dread Plate": "Dark",
    "Earth Plate": "Ground",
    "Fist Plate": "Fighting",
    "Flame Plate": "Fire",
    "Icicle Plate": "Ice",
    "Insect Plate": "Bug",
    "Iron Plate": "Steel",
    "Meadow Plate": "Grass",
    "Mind Plate": "Psychic",
    "Pixie Plate": "Fairy",
    "Sky Plate": "Flying",
    "Splash Plate": "Water",
    "Spooky Plate": "Ghost",
    "Stone Plate": "Rock",
    "Toxic Plate": "Poison",
    "Zap Plate": "Electric",
}

# Abilities that double the attacking stat outright.
_HUGE_POWER = {"Huge Power", "Pure Power"}

# Abilities that halve incoming damage of specific types, or in general.
_THICK_FAT_TYPES = {"Fire", "Ice"}
_FILTER = {"Filter", "Solid Rock", "Prism Armor"}
_MULTISCALE = {"Multiscale", "Shadow Shield"}
# Abilities that make the holder immune to a type regardless of typing.
_TYPE_IMMUNITY_ABILITIES: dict[str, str] = {
    "Levitate": "Ground",
    "Flash Fire": "Fire",
    "Volt Absorb": "Electric",
    "Lightning Rod": "Electric",
    "Motor Drive": "Electric",
    "Water Absorb": "Water",
    "Storm Drain": "Water",
    "Dry Skin": "Water",
    "Sap Sipper": "Grass",
}

# Weather names as the sim reports them. Keys are lowercased with spaces and
# hyphens removed, so "Harsh Sunshine" and "Desolate Land" both fold to Sun.
_WEATHER_ALIASES = {
    "rain": "Rain",
    "primordialsea": "Rain",
    "heavyrain": "Rain",
    "sun": "Sun",
    "sunny": "Sun",
    "sunnyday": "Sun",
    "desolateland": "Sun",
    "harshsunshine": "Sun",
    "extremelyharshsunlight": "Sun",
    "sand": "Sand",
    "sandstorm": "Sand",
    "snow": "Snow",
    "hail": "Snow",
    "snowscape": "Snow",
}

_SCREEN_BREAKING_ABILITIES = {"Infiltrator"}


# -- cards ------------------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """One pre-built draft card, with its stats already computed."""

    card_id: str
    species: str
    item: str | None = None
    ability: str | None = None
    nature: str | None = None
    level: int = 50
    evs: dict = field(default_factory=dict)
    ivs: dict = field(default_factory=dict)
    moves: tuple[str, ...] = ()
    stats: dict = field(default_factory=dict)

    @property
    def types(self) -> tuple[str, ...]:
        return dex.types_of(self.species)

    @property
    def max_hp(self) -> int:
        return stats_mod.stat_of(self.stats, "hp")


def parse_card(raw: Any) -> Card:
    """Build a :class:`Card` from the dict a draft state publishes.

    Total: an unreadable card yields an empty-but-valid Card rather than an
    exception, because this runs inside a live match's decision path.
    """
    if not isinstance(raw, dict):
        return Card(card_id="", species="")
    species = raw.get("species") if isinstance(raw.get("species"), str) else ""
    evs = raw.get("evs") if isinstance(raw.get("evs"), dict) else {}
    ivs = raw.get("ivs") if isinstance(raw.get("ivs"), dict) else {}
    nature = raw.get("nature") if isinstance(raw.get("nature"), str) else None
    level = raw.get("level")
    level = int(level) if isinstance(level, int) and not isinstance(level, bool) else 50
    raw_moves = raw.get("moves")
    moves = tuple(m for m in raw_moves if isinstance(m, str)) if isinstance(raw_moves, list) else ()
    return Card(
        card_id=raw.get("card_id") if isinstance(raw.get("card_id"), str) else "",
        species=species,
        item=raw.get("item") if isinstance(raw.get("item"), str) else None,
        ability=raw.get("ability") if isinstance(raw.get("ability"), str) else None,
        nature=nature,
        level=level,
        evs=dict(evs),
        ivs=dict(ivs),
        moves=moves,
        stats=stats_mod.compute_stats(species, evs, ivs, nature, level),
    )


# -- results ----------------------------------------------------------------


@dataclass(frozen=True)
class DamageResult:
    """A damage roll range, plus what could not be modelled."""

    move: str
    move_type: str | None
    category: str | None
    power: int
    effectiveness: float
    min_damage: int
    max_damage: int
    defender_max_hp: int
    unmodelled: tuple[str, ...] = ()

    @property
    def immune(self) -> bool:
        return self.effectiveness == 0.0

    @property
    def min_percent(self) -> float:
        if not self.defender_max_hp:
            return 0.0
        return 100.0 * self.min_damage / self.defender_max_hp

    @property
    def max_percent(self) -> float:
        if not self.defender_max_hp:
            return 0.0
        return 100.0 * self.max_damage / self.defender_max_hp

    @property
    def possible_ko_hits(self) -> int | None:
        """Fewest hits to KO — the best-case roll. ``None`` if it cannot damage."""
        if self.max_damage <= 0:
            return None
        return math.ceil(self.defender_max_hp / self.max_damage)

    @property
    def guaranteed_ko_hits(self) -> int | None:
        """Hits to KO even on the worst roll — what you can rely on."""
        if self.min_damage <= 0:
            return None
        return math.ceil(self.defender_max_hp / self.min_damage)

    def is_ko(self) -> bool:
        return self.min_damage >= self.defender_max_hp


def _canonical_weather(weather: Any) -> str | None:
    if not isinstance(weather, str) or not weather:
        return None
    return _WEATHER_ALIASES.get(weather.replace(" ", "").replace("-", "").lower())


_RUIN = {  # ability id -> (stat it lowers on every OTHER Pokémon, multiplier)
    "swordofruin": "def",
    "beadsofruin": "spd",
    "tabletsofruin": "atk",
    "vesselofruin": "spa",
}
RUIN_FACTOR = 0.75


def _ruined(stat: str, holder: Card, ruins: set[str]) -> float:
    """The Ruin multiplier on ``holder``'s ``stat``: a Ruin ability hits everyone but its own holder."""
    own = dex.to_id(holder.ability)
    return RUIN_FACTOR if any(_RUIN[r] == stat and r != own for r in ruins) else 1.0


def _attack_stat(
    card: Card,
    category: str,
    boosts: dict,
    crit: bool,
    burned: bool = False,
    *,
    stat_card: Card | None = None,
    stat_key: str | None = None,
    stat_boosts: dict | None = None,
    ruins: set[str] | None = None,
) -> float:
    """The attacker's offensive stat. Body Press reads the user's Defense and Foul Play the TARGET's Attack
    (``stat_card``/``stat_key``/``stat_boosts``); the item and ability modifiers stay the user's."""
    key = stat_key or ("atk" if category == "Physical" else "spa")
    source, source_boosts = (stat_card or card), (boosts if stat_boosts is None else stat_boosts)
    stage = source_boosts.get(key, 0)
    if crit:
        # A critical hit ignores the attacker's negative boosts.
        stage = max(0, stage if isinstance(stage, int) else 0)
    value = stats_mod.stat_of(source.stats, key) * stats_mod.stage_multiplier(stage)
    if ruins and stat_card is None:
        value *= _ruined(key, card, ruins)  # Tablets / Vessel of Ruin on the other side

    if category == "Physical":
        if card.item == "Choice Band":
            value *= 1.5
        if card.ability in _HUGE_POWER:
            value *= 2.0
        # Protosynthesis / Quark Drive with Booster Energy (Atk highest)
        if card.item == "Booster Energy" and card.ability in ("Protosynthesis", "Quark Drive"):
            value *= 1.3
        # Burn halves physical damage (Gen 3+), except for Guts.
        if burned and card.ability != "Guts":
            value *= 0.5
    else:
        if card.item == "Choice Specs":
            value *= 1.5
        if card.item == "Booster Energy" and card.ability in ("Protosynthesis", "Quark Drive"):
            value *= 1.3
    return value


def _defense_stat(
    card: Card, category: str, boosts: dict, crit: bool, ruins: set[str] | None = None
) -> float:
    key = "def" if category == "Physical" else "spd"
    stage = boosts.get(key, 0)
    if crit:
        # A critical hit ignores the defender's positive boosts.
        stage = min(0, stage if isinstance(stage, int) else 0)
    value = stats_mod.stat_of(card.stats, key) * stats_mod.stage_multiplier(stage)
    if ruins:
        value *= _ruined(key, card, ruins)  # Sword / Beads of Ruin on the other side
    if category == "Special" and card.item == "Assault Vest":
        value *= 1.5
    return max(1.0, value)


def expected_hits(move: str, attacker: Card) -> tuple[int, float]:
    """``(guaranteed hits, expected hits)`` of a move: 1 for a single strike, 2-5 for Scale Shot and kin."""
    hits = (dex.move(move) or {}).get("multihit")
    if isinstance(hits, int) and hits > 1:
        return hits, float(hits)
    if isinstance(hits, list) and len(hits) == 2 and all(isinstance(h, int) for h in hits):
        low, high = hits
        if dex.to_id(attacker.ability) == "skilllink":
            return high, float(high)
        # The 2-5 hit table (35/35/15/15 %) averages 3.1; a 2-3 hit move averages 2.5.
        return low, 3.1 if (low, high) == (2, 5) else (low + high) / 2
    return 1, 1.0


def damage(
    attacker: Card,
    defender: Card,
    move: str,
    *,
    attacker_boosts: dict | None = None,
    defender_boosts: dict | None = None,
    weather: Any = None,
    spread: bool = False,
    crit: bool = False,
    reflect: bool = False,
    light_screen: bool = False,
    helping_hand: bool = False,
    friend_guard: bool = False,
    defender_full_hp: bool = False,
    burned: bool = False,
    ruins: Any = (),
) -> DamageResult:
    """Damage from ``attacker``'s ``move`` into ``defender``.

    ``spread=True`` for a move hitting more than one target in doubles (0.75x).
    Boosts are stat stages (±1..±6) keyed ``atk``/``def``/``spa``/``spd``.

    Field state the cards cannot tell us — an Intimidate that already fired, a
    Tailwind, an opponent's item — is the caller's job: pass it in through
    ``attacker_boosts``/``defender_boosts``/``weather``/screens. Anything
    recognised but unapplied is named in ``DamageResult.unmodelled``.
    """
    attacker_boosts = attacker_boosts or {}
    defender_boosts = defender_boosts or {}
    unmodelled: list[str] = []

    move_type = dex.move_type(move)
    category = dex.move_category(move)
    power = dex.move_power(move)

    result = DamageResult(
        move=move,
        move_type=move_type,
        category=category,
        power=power,
        effectiveness=1.0,
        min_damage=0,
        max_damage=0,
        defender_max_hp=defender.max_hp,
    )

    if not dex.is_damaging(move) or not move_type or category not in ("Physical", "Special"):
        return result  # a status move deals no direct damage

    # Immunity from typing, plus abilities that grant one.
    effectiveness = dex.effectiveness(move_type, defender.types)
    if _TYPE_IMMUNITY_ABILITIES.get(defender.ability or "") == move_type:
        effectiveness = 0.0
    if effectiveness == 0.0:
        return DamageResult(
            move=move,
            move_type=move_type,
            category=category,
            power=power,
            effectiveness=0.0,
            min_damage=0,
            max_damage=0,
            defender_max_hp=defender.max_hp,
            unmodelled=tuple(unmodelled),
        )

    # Ruin abilities on the field (plus either side's own, whose holder they spare).
    ruin_set = {dex.to_id(r) for r in ruins or ()} | {
        dex.to_id(c.ability) for c in (attacker, defender) if dex.to_id(c.ability) in _RUIN
    }
    ruin_set &= set(_RUIN)

    entry = dex.move(move) or {}
    stat_card = stat_key = stat_boosts = None
    if entry.get("overrideOffensivePokemon") == "target":  # Foul Play: the target's own Attack
        stat_card, stat_key, stat_boosts = defender, entry.get("overrideOffensiveStat") or "atk", defender_boosts
    elif entry.get("overrideOffensiveStat") in ("def", "spd", "atk", "spa"):  # Body Press: the user's Defense
        stat_key = entry["overrideOffensiveStat"]
    attack = _attack_stat(
        attacker,
        category,
        attacker_boosts,
        crit,
        burned,
        stat_card=stat_card,
        stat_key=stat_key,
        stat_boosts=stat_boosts,
        ruins=ruin_set,
    )
    defence = _defense_stat(defender, category, defender_boosts, crit, ruin_set)

    base = (
        math.floor(
            math.floor(math.floor(2 * attacker.level / 5 + 2) * power * attack / defence) / 50
        )
        + 2
    )

    modifiers = 1.0

    # Same-type attack bonus.
    stab = 1.5 if move_type in attacker.types else 1.0
    if attacker.ability == "Adaptability" and stab > 1.0:
        stab = 2.0
    modifiers *= stab
    modifiers *= effectiveness

    if spread:
        modifiers *= 0.75

    canonical = _canonical_weather(weather)
    if canonical == "Rain":
        if move_type == "Water":
            modifiers *= 1.5
        elif move_type == "Fire":
            modifiers *= 0.5
    elif canonical == "Sun":
        if move_type == "Fire":
            modifiers *= 1.5
        elif move_type == "Water":
            modifiers *= 0.5

    # Screens halve the relevant side, and a critical hit ignores them.
    if not crit and "Infiltrator" not in (attacker.ability or ""):
        if category == "Physical" and reflect:
            modifiers *= 0.5
        if category == "Special" and light_screen:
            modifiers *= 0.5

    if helping_hand:
        modifiers *= 1.5
    if friend_guard:
        modifiers *= 0.75

    # Item modifiers.
    item = attacker.item or ""
    if TYPE_BOOST_ITEMS.get(item) == move_type:
        modifiers *= 1.2
    if item == "Life Orb":
        modifiers *= 1.3

    # Ability modifiers on the defending side.
    ability = defender.ability or ""
    if ability == "Thick Fat" and move_type in _THICK_FAT_TYPES:
        modifiers *= 0.5
    if ability in _MULTISCALE and defender_full_hp:
        modifiers *= 0.5
    if ability in _FILTER and effectiveness > 1.0:
        modifiers *= 0.75
    if ability == "Fluffy" and category == "Physical":
        modifiers *= 0.5

    if crit:
        modifiers *= 1.5

    # Report the gaps honestly rather than guessing at them.
    if (
        item
        and item not in TYPE_BOOST_ITEMS
        and item
        not in {
            "Life Orb",
            "Choice Band",
            "Choice Specs",
            "Assault Vest",
        }
    ):
        unmodelled.append(f"item:{item}")
    if (
        ability
        and ability
        not in (
            _HUGE_POWER
            | _FILTER
            | _MULTISCALE
            | _SCREEN_BREAKING_ABILITIES
            | set(_TYPE_IMMUNITY_ABILITIES)
        )
        and ability not in ("Thick Fat", "Adaptability", "Fluffy")
    ):
        unmodelled.append(f"ability:{ability}")

    raw = base * modifiers
    low_hits, mean_hits = expected_hits(move, attacker)
    min_damage = max(1, math.floor(raw * 0.85)) * low_hits
    max_damage = max(1, math.floor(raw * mean_hits))

    return DamageResult(
        move=move,
        move_type=move_type,
        category=category,
        power=power,
        effectiveness=effectiveness,
        min_damage=min_damage,
        max_damage=max_damage,
        defender_max_hp=defender.max_hp,
        unmodelled=tuple(unmodelled),
    )


def best_move(
    attacker: Card,
    defender: Card,
    *,
    moves: Any = None,
    **kwargs: Any,
) -> DamageResult | None:
    """The attacker's hardest-hitting move into this defender.

    A convenience for the policy: ``None`` when nothing in the move list deals
    damage at all.
    """
    candidates = moves if moves is not None else attacker.moves
    best: DamageResult | None = None
    for move in candidates or ():
        if not isinstance(move, str):
            continue
        result = damage(attacker, defender, move, **kwargs)
        if result.max_damage <= 0:
            continue
        if best is None or result.max_damage > best.max_damage:
            best = result
    return best
