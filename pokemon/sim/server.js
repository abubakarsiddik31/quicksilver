'use strict';
// JSON-lines server: one request per line on stdin, one response per line on stdout.
const readline = require('readline');
const engine = require('./engine');

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let req;
  try {
    req = JSON.parse(line);
  } catch (e) {
    return;
  }
  const out = { id: req.id };
  try {
    if (req.cmd === 'ping') out.ok = true;
    else if (req.cmd === 'search') out.result = engine.search(req.state, req.budget_ms || 3000, req.opts || {});
    else if (req.cmd === 'lineups') out.result = engine.evalLineups(req.req);
    else out.error = 'unknown cmd';
  } catch (e) {
    out.error = String((e && e.stack) || e).slice(0, 800);
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
