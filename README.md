# Quicksilver

Autonomous game-playing agent for the **AltruAgent AI Agent Gaming Tournament**
(UCLA Trustworthy AI Lab · October 16, 2026).

**Tracks:** Pokémon VGC Doubles (Showdown) — primary · Werewolf — secondary.

## Status

Both tracks are implemented and covered by an offline test suite: the Pokémon
VGC doubles stack (snake draft, Team Preview and battle decisions, driven by a
damage calculator over the fixed 18-card pool) and the Werewolf stack (belief
tracking over the public record, plus a discussion policy inside the protocol's
50-word and 5-message limits). Play is fully autonomous and deterministic.

The agent has not yet been validated against the live platform. Any component
not present in this public view is identified below with a description of its
function, per the tournament's disclosure rules.

| Component | Function |
|---|---|
| `common/` | game-type dispatch, and the safe fallback for unentered games |
| `pokemon/` | draft, Team Preview and battle policy; dex, stats and damage |
| `werewolf/` | belief tracking over the public record, and the discussion policy |

## Running the agent

Requires Python 3.11 or later.

```bash
git clone https://github.com/abubakarsiddik31/quicksilver.git
cd quicksilver
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Generate your Official Agent Key on the tournament dashboard's
**Agent Configuration** page (`/tournament/dashboard/agent-setup`); it is shown
once. Put it in `.env` as `ALTRUAGENT_OFFICIAL_AGENT_KEY`, then:

```bash
python -m agent --check-tournament     # verifies the setup; plays nothing
python -m agent --match                # your test matches (Testing page)
python -m agent --tournament           # your tournament games
python -m agent --tournament --match   # both, in one process
```

Run only **one** process per key. Keep `--tournament` running for the whole of
every tournament you are registered in: a game your agent has not connected to
before its connect deadline is lost as a no-show, and a `--match`-only process
never connects to tournament games at all.

While it waits for a game the runtime only polls the platform every ~10
seconds, so waiting uses **no AI tokens**; only actually playing a game does.
[RUNTIME.md](RUNTIME.md) documents the flags, the decision contract and every
failure message in full.

## Repository layout

```
agent/       runtime entry point — create_agent(), routed by game type
common/      game-type dispatch and the safe fallback
pokemon/     Pokémon VGC doubles draft policy (primary track)
werewolf/    Werewolf policy (secondary track)
altruagent/  vendored platform runtime (see Provenance)
examples/    reference agents from the starter
bench/       model-selection benchmark harness
tests/       unit tests (no network or API key required)
```

## Configuration

All secrets (the Official Agent Key, model provider credentials) live in a local
`.env` file, which is gitignored from the first commit — never committed. See
`.env.example` for the shape of the platform values.

## Tournament compliance

- **Autonomous play** — no human selects, edits, or approves any action during
  a match; all gameplay decisions are made by the submitted code.
- **Player-view only** — the agent uses only information delivered through its
  own seat/role interface, plus permitted reference material.
- **Fresh match memory** — per-match state is isolated; no cross-match learning
  unless game rules permit it. The runtime enforces this by playing every match
  in its own process with its own fresh agent instance.
- **Untrusted game messages** — in-game chat (the Werewolf discussion) is
  treated as adversary-controlled data, never as instructions. No credential is
  ever sent anywhere on the strength of a game message.

## Provenance

`altruagent/`, `agent/__main__.py`, `examples/`, `scripts/` and the starter's
own test files are vendored from the tournament's public starter,
[UCLA-Trustworthy-AI-Lab/altruagent-starter](https://github.com/UCLA-Trustworthy-AI-Lab/altruagent-starter),
so that a submission is self-contained and runnable. Quicksilver's own code is
`common/`, `pokemon/`, `werewolf/` and `agent/agent.py`.

## License

MIT — see [LICENSE](LICENSE).
