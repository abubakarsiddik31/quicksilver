"""Re-download the vendored Pokémon Showdown data in this directory.

Run it from anywhere::

    python pokemon/data/fetch_data.py

It is **not** needed at runtime — :mod:`pokemon.stats` and :mod:`pokemon.damage`
read the committed JSON files. This script exists so the vendored data can be
refreshed and audited: every file it writes comes from one of the URLs in
``SOURCES`` below, with no hand editing.

Sources
-------
``play.pokemonshowdown.com/data/`` is the data directory that the live Showdown
client serves, i.e. the same tables the tournament's Showdown server plays by.
It publishes two different formats:

* ``pokedex.json`` and ``moves.json`` are real JSON. They are stored here
  byte-for-byte as downloaded.
* ``abilities.js``, ``items.js`` and ``typechart.js`` are JavaScript modules
  (``exports.BattleAbilities = {...};``). They are pure data — no functions,
  no comments — so :func:`_parse_js_module` converts them to JSON. This data
  directory publishes no ``abilities.json`` / ``items.json`` /
  ``typechart.json`` (those URLs 404), and the server-side TypeScript sources
  do contain callback functions, so the JS modules are the cleanest machine
  readable form available. The conversion is verified by
  ``tests/test_pokemon_damage.py`` which re-parses the committed JSON and
  checks the entries damage maths depends on.
* ``natures.json`` does not exist anywhere in that directory, so natures come
  from Showdown's own source of truth, ``data/natures.ts`` in the
  ``smogon/pokemon-showdown`` repository, parsed by :func:`_parse_natures_ts`.

Only Python's standard library is used. Downloads go through :mod:`urllib`;
``curl`` was used by hand for the first fetch and produces identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent

SHOWDOWN_DATA = "https://play.pokemonshowdown.com/data"
SHOWDOWN_SOURCE = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data"

# (output file name, source URL, how to turn the response into JSON)
SOURCES: list[tuple[str, str, str]] = [
    ("pokedex.json", f"{SHOWDOWN_DATA}/pokedex.json", "json"),
    ("moves.json", f"{SHOWDOWN_DATA}/moves.json", "json"),
    ("abilities.json", f"{SHOWDOWN_DATA}/abilities.js", "js"),
    ("items.json", f"{SHOWDOWN_DATA}/items.js", "js"),
    ("typechart.json", f"{SHOWDOWN_DATA}/typechart.js", "js"),
    ("natures.json", f"{SHOWDOWN_SOURCE}/natures.ts", "natures"),
]


# --------------------------------------------------------------------------
# JavaScript object literal -> Python data
# --------------------------------------------------------------------------
#
# Showdown's published ``.js`` data files are ``exports.BattleX = {..};`` where
# the object literal differs from JSON in exactly one way: bare identifier
# keys (``{adamant:{name:"Adamant"}}``). Values are double-quoted strings,
# numbers, ``true``/``false``/``null``, arrays and nested objects. There are no
# functions, comments, template literals or single-quoted strings in the files
# this script reads; the parser nevertheless accepts single-quoted strings and
# reports a clear error on anything it does not understand rather than
# guessing.

_WHITESPACE = " \t\r\n"
_DIGITS = set(string.digits)
_IDENT_START = set(string.ascii_letters) | {"_", "$"}
_IDENT_CHARS = _IDENT_START | _DIGITS
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
    "\\": "\\",
    "/": "/",
    "'": "'",
    '"': '"',
}


class JsLiteralError(ValueError):
    """The text is not a JavaScript object literal this parser understands."""


class _JsParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    # -- helpers ----------------------------------------------------------
    def _error(self, message: str) -> JsLiteralError:
        line = self.text.count("\n", 0, self.pos) + 1
        column = self.pos - self.text.rfind("\n", 0, self.pos)
        return JsLiteralError(f"{message} at line {line} column {column}")

    def _skip_whitespace(self) -> None:
        text, pos = self.text, self.pos
        while pos < len(text) and text[pos] in _WHITESPACE:
            pos += 1
        self.pos = pos

    def _peek(self) -> str:
        self._skip_whitespace()
        if self.pos >= len(self.text):
            raise self._error("unexpected end of input")
        return self.text[self.pos]

    # -- grammar ----------------------------------------------------------
    def parse_value(self) -> Any:
        char = self._peek()
        if char == "{":
            return self.parse_object()
        if char == "[":
            return self.parse_array()
        if char in "\"'":
            return self.parse_string()
        if char == "-" or char in _DIGITS:
            return self.parse_number()
        return self.parse_keyword()

    def parse_object(self) -> dict[str, Any]:
        assert self.text[self.pos] == "{"
        self.pos += 1
        result: dict[str, Any] = {}
        while True:
            if self._peek() == "}":
                self.pos += 1
                return result
            key = self.parse_key()
            if self._peek() != ":":
                raise self._error(f"expected ':' after object key {key!r}")
            self.pos += 1
            result[key] = self.parse_value()
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "}":
                self.pos += 1
                return result
            raise self._error(f"expected ',' or '}}' after object key {key!r}")

    def parse_array(self) -> list[Any]:
        assert self.text[self.pos] == "["
        self.pos += 1
        result: list[Any] = []
        while True:
            if self._peek() == "]":
                self.pos += 1
                return result
            result.append(self.parse_value())
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == "]":
                self.pos += 1
                return result
            raise self._error("expected ',' or ']' in array")

    def parse_key(self) -> str:
        char = self._peek()
        if char in "\"'":
            return self.parse_string()
        start = self.pos
        if char not in _IDENT_START:
            raise self._error(f"expected an object key, found {char!r}")
        while self.pos < len(self.text) and self.text[self.pos] in _IDENT_CHARS:
            self.pos += 1
        return self.text[start : self.pos]

    def parse_string(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        chunks: list[str] = []
        while True:
            if self.pos >= len(self.text):
                raise self._error("unterminated string")
            char = self.text[self.pos]
            if char == quote:
                self.pos += 1
                return "".join(chunks)
            if char == "\\":
                self.pos += 1
                chunks.append(self._unescape())
                continue
            chunks.append(char)
            self.pos += 1

    def _unescape(self) -> str:
        if self.pos >= len(self.text):
            raise self._error("unterminated escape sequence")
        char = self.text[self.pos]
        self.pos += 1
        if char == "u":
            return self._hex_escape(4)
        if char == "x":
            return self._hex_escape(2)
        if char in _ESCAPES:
            return _ESCAPES[char]
        return char

    def _hex_escape(self, length: int) -> str:
        digits = self.text[self.pos : self.pos + length]
        if len(digits) != length:
            raise self._error("truncated unicode escape")
        self.pos += length
        return chr(int(digits, 16))

    def parse_number(self) -> int | float:
        match = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?").match(self.text, self.pos)
        if match is None:
            raise self._error("malformed number")
        self.pos = match.end()
        token = match.group(0)
        if any(char in token for char in ".eE"):
            return float(token)
        return int(token)

    def parse_keyword(self) -> Any:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] in _IDENT_CHARS:
            self.pos += 1
        word = self.text[start : self.pos]
        if word == "true":
            return True
        if word == "false":
            return False
        if word == "null":
            return None
        raise self._error(f"unsupported JavaScript value {word!r}")


def _parse_js_module(text: str, variable: str) -> Any:
    """Return the data assigned to ``exports.<variable>`` in a Showdown module."""
    marker = f"exports.{variable}"
    start = text.find(marker)
    if start < 0:
        raise JsLiteralError(f"{marker!r} not found in module")
    start = text.find("{", start)
    if start < 0:
        raise JsLiteralError(f"no object literal after {marker!r}")
    parser = _JsParser(text)
    parser.pos = start
    value = parser.parse_value()
    parser._skip_whitespace()
    if parser.text[parser.pos :].strip(" \t\r\n;") != "":
        raise JsLiteralError(f"trailing content after {marker!r} object")
    return value


_JS_VARIABLES = {
    "abilities.json": "BattleAbilities",
    "items.json": "BattleItems",
    "typechart.json": "BattleTypeChart",
}


# --------------------------------------------------------------------------
# natures.ts -> JSON
# --------------------------------------------------------------------------

_TS_NATURE = re.compile(
    r"^\t(?P<id>[a-z]+):\s*\{(?P<body>.*?)^\t\},",
    re.MULTILINE | re.DOTALL,
)


def _parse_natures_ts(text: str) -> dict[str, dict[str, str]]:
    """Parse Showdown's ``data/natures.ts``.

    Each entry is a small literal block; only ``name``, ``plus`` and ``minus``
    are read. A nature with neither stat is neutral, matching Showdown's
    ``NatureData`` (``plus``/``minus`` omitted).
    """
    natures: dict[str, dict[str, str]] = {}
    for match in _TS_NATURE.finditer(text):
        body = match.group("body")
        entry: dict[str, str] = {}
        name = re.search(r"name:\s*[\"']([^\"']+)[\"']", body)
        if name is None:
            raise JsLiteralError(f"nature {match.group('id')!r} has no name")
        entry["name"] = name.group(1)
        for field in ("plus", "minus"):
            found = re.search(rf"{field}:\s*[\"']([a-z]+)[\"']", body)
            if found is not None:
                entry[field] = found.group(1)
        natures[match.group("id")] = entry
    if len(natures) != 25:
        raise JsLiteralError(f"expected 25 natures, parsed {len(natures)}")
    return natures


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "quicksilver-fetch-data"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main(argv: list[str]) -> int:
    only = set(argv[1:])
    fetched_utc = time.strftime("%Y-%m-%d", time.gmtime())
    print(f"fetching into {DATA_DIR} (date {fetched_utc} UTC)")
    for name, url, kind in SOURCES:
        if only and name not in only:
            continue
        raw = fetch(url)
        if kind == "json":
            payload = raw
            # Validate before writing: a truncated download must not land.
            json.loads(raw.decode("utf-8"))
        else:
            text = raw.decode("utf-8")
            if kind == "js":
                data = _parse_js_module(text, _JS_VARIABLES[name])
            else:
                data = _parse_natures_ts(text)
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        (DATA_DIR / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()[:16]
        print(f"  {name:<16} {len(payload):>8} bytes  sha256:{digest}  <- {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
