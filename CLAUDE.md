# Hard rules for this repository

These come straight from the repository owner. They are not suggestions. Do not ignore them, do not trade them
off against speed, and do not wait to be reminded.

## 1. Decisions come from a full battle picture, not from patch rules

- Every game agent (Red Alert, Pokémon, Werewolf, ...) decides from a real model of the situation, the way a
  real commander or player would: what the enemy likely has (seen, remembered, inferred), how the matchup plays out
  (unit/Pokémon types, defences, range, terrain, tempo), and what is actually happening in the fight right now
  (the real exchange rate against what was predicted).
- The agent has memory and keeps the whole game in mind. It retains what it has learned for the entire match, not
  just the current tick: where and when the enemy was seen and with what, what their economy and tech look like,
  every fight so far (what we sent, what it cost each side, why it went that way), what each earlier plan
  achieved, and how the match stands overall (economy, army, trades, the clock and the tiebreak). Decisions are
  strategy built on that retained picture, not reactions to the latest observation.
- The General / strategist decides. Doctrine and "rules" are a guide for it, not a replacement for it.
- Do NOT add hard-coded thresholds, minimum counts, magic numbers or special-case overrides to the executors to
  fix one observed loss ("advance only with 5 units", "wait until value 9000", "push at 7 minutes"). Fixing one
  symptom this way breaks another; this has happened repeatedly and is the main reason the bots keep losing.
- The only hard limits allowed are safety invariants that are always true (the base always keeps a guard; never
  sell the last production building; never act on an illegal move). Anything that is a judgement call belongs to
  the decision model.

## 2. Fixes must be general and reasoned

- Before changing behaviour, state the root cause and why the change generalises beyond the one game that exposed
  it. If the honest answer is "this patches one case", do not make the change; improve the model instead.
- Change one thing at a time and judge it on many recorded games (trade rate, army/economy at fixed times,
  result), not on a single match.
- When a change interacts with another rule, fix the shared model, not the second rule.

## 3. Do not burn the owner's API credits

- Live games cost real money. Never use live or self-play games to experiment, to "see if it works", or on a loop.
- Validate offline first: replay recorded traces in `traces/`, the Pokémon engine in `pokemon/sim`, and recorded
  fights. Only play live games after a change has passed offline checks and the owner agrees.
- Never schedule repeating jobs (cron, loops, auto-join, auto-create matches) unless the owner asks for exactly
  that, and always say what it costs. Never run routine self-play.
- Joining matches that other contestants opened is fine when asked; creating matches only when asked.

## 4. Never cut a running real match

- Do not restart or stop the runner while any match against another contestant is running. Python changes need a
  runner restart, so make them, then wait for the matches to finish before restarting (JS engine changes in
  `pokemon/sim` apply to new matches without one).
- After a restart-requiring change, restart as soon as no real match is running, so the change actually takes
  effect.

## 5. Patience is correct; rushing is not

- There is no need to win early. Do not tune Red Alert (or any game) to attack earlier, to rush, or to take risks
  for tempo. A match that is decided on the time-limit tiebreak is decided on trades and economy, so protect both.
- Do not push an army forward because it is large; push when the picture says the fight is favourable and keep
  re-checking while it happens.

## 6. Pokémon: use the rules engine

- Pokémon turns and Team Preview are decided by the search on `@pkmn/sim` in `pokemon/sim` and
  `pokemon/simsearch.py`. Do not add heuristic patches in front of it. If the search is wrong, fix the engine
  inputs, the foe model or the evaluation.
- The engine silently falls back to heuristics when it cannot rebuild the board (missing card sets, wildcard
  formes, ...). Treat any fallback as a bug to find, not a normal path.

## 7. Git and workflow

- Never commit or push unless the owner says to. Never push to GitHub in this repository without being told.
- Do not spend time or tokens on local unit tests; they are not worth it here. Use trace replay and the engine
  for validation instead. Keep existing tests passing when a change touches them, but do not build new test
  suites as a substitute for checking behaviour on real games.
- Keep replies short and plain. Say what is wrong, what you changed and why, and what is not verified.

## 8. Running the agent

- Always start the runner with the Red Alert strategist on, or Red Alert silently plays the old built-in playbook
  with no General and no match memory:

  ```bash
  RA_LLM=1 uv run python -m agent --match
  ```

- Check the first lines of the log for "playing the built-in playbook"; if it appears, `RA_LLM=1` was missing.
- Only one runtime may hold a seat: before starting, check that no other `python -m agent` process is running, and
  never kill a runner you did not start.
