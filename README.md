# Quicksilver

Autonomous game-playing agent for the **AltruAgent AI Agent Gaming Tournament**
(UCLA Trustworthy AI Lab · October 16, 2026).

**Tracks:** Pokémon VGC Doubles (Showdown) — primary · Werewolf — secondary.

## Status

Skeleton during development — the competitive implementation is built privately
and will be pushed and pinned to `main` before the submission lock
(**October 13, 2026, 11:59 PM PT**). Components intentionally withheld from
this public view will be identified with a description of their function per
the tournament's disclosure rules.

## Repository layout

```
common/      shared agent infrastructure (implementation private)
pokemon/     Showdown battle agent (implementation private)
werewolf/    Werewolf game agent (implementation private)
```

## Configuration

All secrets (model API keys, game-server credentials) live in a local `.env`
file, which is gitignored from the first commit — never committed. See
`.env.example` for the shape.

## Tournament compliance

- **Autonomous play** — no human selects, edits, or approves any action during
  a match; all gameplay decisions are made by the submitted code.
- **Player-view only** — the agent uses only information delivered through its
  own seat/role interface, plus permitted reference material.
- **Fresh match memory** — per-match state is isolated; no cross-match learning
  unless game rules permit it.

## License

MIT — see [LICENSE](LICENSE).
