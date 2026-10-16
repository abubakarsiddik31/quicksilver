'use strict';
// A Monte-Carlo search over the real Pokémon Showdown rules engine (@pkmn/sim).
//
// The Python agent used to score a turn with hand-written heuristics, so every rule it had not
// been taught (Choice locks, Focus Sash vs Fake Out, Trick Room against Tailwind, ...) was a
// silent blind spot. Here the rules come from the engine itself: we rebuild the live battle,
// try each of our joint choices against the foe's likely replies, and play the game on a few turns
// with a cheap policy to see who is winning.
const { Battle, Dex, PRNG } = require('@pkmn/sim');

const FORMAT = 'gen9doublescustomgame';
const STATS = ['atk', 'def', 'spa', 'spd', 'spe'];

// ---------------------------------------------------------------------------------------------
// rng
function mulberry(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------------
// building a battle from the live state

function toSet(m) {
  const s = m.set;
  return {
    name: s.species,
    species: s.species,
    item: s.item || '',
    ability: s.ability || '',
    moves: s.moves,
    nature: s.nature || 'Hardy',
    gender: '',
    evs: s.evs || {},
    ivs: s.ivs || {},
    level: s.level || 50,
    // the sim defaults an unset Tera type to the species' first type, which is
    // also the right guess for a card that lists none
    teraType: s.teraType || undefined,
  };
}

function build(state, seed) {
  const sides = state.sides;
  const mk = (side) => side.mons.map(toSet);
  const b = new Battle({
    formatid: FORMAT,
    seed: [seed & 0xffff, (seed >>> 8) & 0xffff, (seed >>> 16) & 0xffff, 7],
    p1: { name: 'us', team: mk(sides[0]) },
    p2: { name: 'them', team: mk(sides[1]) },
  });
  const order = (n) => 'team ' + Array.from({ length: Math.min(n, 6) }, (_, i) => i + 1).join('');
  b.makeChoices(order(sides[0].mons.length), order(sides[1].mons.length));
  for (let s = 0; s < 2; s++) injectSide(b, b.sides[s], sides[s]);
  injectField(b, state);
  b.turn = state.turn || b.turn;
  const fs = state.forceSwitch || {};
  let forced = false;
  [fs.ours || [], fs.theirs || []].forEach((flags, sideIdx) => {
    flags.forEach((flag, slot) => {
      const mon = b.sides[sideIdx].active[slot];
      if (flag && mon) {
        mon.switchFlag = true;
        forced = true;
      }
    });
  });
  refreshRequest(b, forced ? 'switch' : 'move');
  return b;
}

function injectSide(b, side, spec) {
  spec.mons.forEach((m, i) => {
    const p = side.pokemon[i];
    if (!p) return;
    if (m.hpFrac <= 0 || m.fainted) {
      p.hp = 0;
      p.status = 'fnt';
      p.fainted = true;
      p.faintQueued = false;
      side.pokemonLeft = Math.max(0, side.pokemonLeft - 1);
      return;
    }
    p.hp = Math.max(1, Math.round(m.hpFrac * p.maxhp));
    if (m.status) {
      p.status = m.status;
      p.statusState = { id: m.status, target: p, time: 2 };
    }
    for (const k of Object.keys(m.boosts || {})) p.boosts[k] = m.boosts[k];
    if (m.itemConsumed) {
      p.item = '';
      p.itemState = { id: '', target: p };
    }
    if (m.terastallized) {
      // the live board says this Pokémon has already Terastallized: adopt that
      // type so typing, STAB and damage match what the opponent actually has
      p.teraType = m.terastallized;
      p.terastallized = m.terastallized;
    }
    p.activeMoveActions = m.moveActions || 0;
    p.activeTurns = m.moveActions || 0;
    if (m.choiceLock && p.hasMove(m.choiceLock)) {
      p.volatiles['choicelock'] = { id: 'choicelock', target: p, move: m.choiceLock };
    }
    if (m.protectedLast) {
      p.volatiles['stall'] = { id: 'stall', target: p, duration: 2, counter: 3 };
    }
    for (const v of m.volatiles || []) {
      try {
        p.addVolatile(v);
      } catch (e) {
        /* an effect the engine cannot start from here: skip it */
      }
    }
  });
  for (const [cond, left] of Object.entries(spec.conditions || {})) {
    const src = side.active.find((p) => p && !p.fainted) || side.active[0];
    try {
      if (!side.addSideCondition(cond, src, b.dex.moves.get(cond))) continue;
    } catch (e) {
      continue;
    }
    if (left && side.sideConditions[cond]) side.sideConditions[cond].duration = left;
  }
}

function injectField(b, state) {
  const src = b.sides[0].active.find((p) => p && !p.fainted) || b.sides[0].active[0];
  if (state.weather) {
    b.field.setWeather(state.weather, src);
    if (state.weatherLeft && b.field.weatherState) b.field.weatherState.duration = state.weatherLeft;
  }
  if (state.terrain) {
    b.field.setTerrain(state.terrain, src);
    if (state.terrainLeft && b.field.terrainState) b.field.terrainState.duration = state.terrainLeft;
  }
  if (state.trickRoom) {
    b.field.addPseudoWeather('trickroom', src);
    if (b.field.pseudoWeather.trickroom) b.field.pseudoWeather.trickroom.duration = state.trickRoom;
  }
}

function refreshRequest(b, kind) {
  if (kind === 'switch') {
    b.makeRequest('switch');
    return;
  }
  for (const side of b.sides) {
    for (const p of side.active) {
      if (!p || p.fainted) continue;
      for (const ms of p.moveSlots) {
        ms.disabled = false;
        ms.disabledSource = '';
      }
      b.runEvent('DisableMove', p);
      for (const ms of p.moveSlots) {
        b.singleEvent('DisableMove', b.dex.getActiveMove(ms.id), null, p);
      }
    }
  }
  b.makeRequest('move');
}

// toJSON() alone shares mutable objects with the original, so a clone must come from the string.
function snapshot(b) {
  return JSON.stringify(b.toJSON());
}

function clone(b) {
  return Battle.fromJSON(typeof b === 'string' ? b : snapshot(b));
}

// ---------------------------------------------------------------------------------------------
// legal options and the heuristic policy

const NEEDS_TARGET = new Set(['normal', 'any', 'adjacentFoe', 'adjacentAlly', 'adjacentAllyOrSelf']);

function slotOptions(b, sideIdx, slot) {
  const side = b.sides[sideIdx];
  const mon = side.active[slot];
  const req = side.activeRequest;
  if (!req) return [{ kind: 'pass', str: 'pass' }];
  if (b.requestState === 'switch') {
    if (!req.forceSwitch || !req.forceSwitch[slot]) return [{ kind: 'pass', str: 'pass' }];
    return switchOptions(side, true);
  }
  if (!mon || mon.fainted || mon.hp <= 0 || !req.active || !req.active[slot]) {
    return [{ kind: 'pass', str: 'pass' }];
  }
  const out = [];
  const act = req.active[slot];
  act.moves.forEach((mv, i) => {
    if (mv.disabled || !mv.id) return;
    const idx = i + 1;
    if (!NEEDS_TARGET.has(mv.target)) {
      out.push({ kind: 'move', id: mv.id, str: `move ${idx}`, target: 0 });
      return;
    }
    const foes = [1, 2];
    const ally = slot === 0 ? -2 : -1;
    // Every adverse status move in the pool (Taunt, Thunder Wave, Spore, Encore, ...) only
    // makes sense aimed at a foe, and a status aimed at our ally is strictly a wasted move —
    // one that rollouts can rank #1 in a doomed slot. Beneficial ally moves are either a
    // different target class (Helping Hand) or damage (Pollen Puff), which keep the ally target.
    const statusOnly = Dex.moves.get(mv.id).category === 'Status';
    const targets = mv.target === 'adjacentAlly' ? [ally] : mv.target === 'adjacentAllyOrSelf' ? [ally, -(slot + 1)] : statusOnly ? foes : [...foes, ally];
    for (const t of targets) {
      out.push({ kind: 'move', id: mv.id, str: `move ${idx} ${t}`, target: t });
    }
  });
  if (!act.trapped && !act.maybeTrapped) out.push(...switchOptions(side, false));
  if (!out.length) out.push({ kind: 'move', id: 'struggle', str: 'move 1', target: 0 });
  return out;
}

function switchOptions(side, forced) {
  const out = [];
  side.pokemon.forEach((p, i) => {
    if (i < side.active.length || p.fainted || p.hp <= 0) return;
    out.push({ kind: 'switch', index: i + 1, species: p.species.name, str: `switch ${i + 1}` });
  });
  return out;
}

function effectiveness(moveType, target) {
  const types = target.getTypes();
  if (!Dex.getImmunity(moveType, types)) return 0;
  let e = 1;
  for (const t of types) e *= Math.pow(2, Dex.getEffectiveness(moveType, t));
  return e;
}

function boostMult(n) {
  return n >= 0 ? (2 + n) / 2 : 2 / (2 - n);
}

// the stat with its boosts, without running events (those need a move in flight)
function statOf(p, stat) {
  return Math.max(1, Math.floor(p.storedStats[stat] * boostMult(p.boosts[stat] || 0)));
}

// expected damage of `mv` from `src` on `tgt`, as a percent of tgt's max HP (a rough rollout estimate)
function damagePct(src, tgt, mv) {
  if (mv.category === 'Status' || !mv.basePower) return 0;
  const cat = mv.category === 'Physical' ? ['atk', 'def'] : ['spa', 'spd'];
  let atkStat = cat[0];
  let defStat = cat[1];
  if (mv.overrideOffensiveStat) atkStat = mv.overrideOffensiveStat;
  if (mv.overrideDefensivePokemon === 'target') {
    /* foul play: target's attack */
  }
  const A = mv.id === 'bodypress' ? statOf(src, 'def') : mv.id === 'foulplay' ? statOf(tgt, 'atk') : statOf(src, atkStat);
  const D = statOf(tgt, defStat);
  let bp = mv.basePower;
  if (mv.multihit) bp *= Array.isArray(mv.multihit) ? 3 : mv.multihit;
  let dmg = Math.floor(Math.floor((22 * bp * A) / D) / 50) + 2;
  const eff = effectiveness(mv.type, tgt);
  if (eff === 0) return 0;
  if (src.hasType(mv.type)) dmg *= 1.5;
  dmg *= eff;
  const item = src.item;
  if (item === 'lifeorb') dmg *= 1.3;
  if (item === 'choiceband' && cat[0] === 'atk') dmg *= 1.5;
  if (item === 'choicespecs' && cat[0] === 'spa') dmg *= 1.5;
  if (src.status === 'brn' && mv.category === 'Physical') dmg *= 0.5;
  const acc = mv.accuracy === true ? 1 : mv.accuracy / 100;
  return ((dmg * acc) / tgt.maxhp) * 100;
}


// A sleep or paralysis on a healthy, non-immune foe costs it several turns; the greedy score used to
// treat Spore like a 6-point utility move, so rollouts never saw the foe's Amoonguss put our team to sleep.
const POWDERS = new Set(['spore', 'sleeppowder']);
const SLEEPERS = new Set(['spore', 'sleeppowder', 'hypnosis', 'sing', 'lovelykiss', 'yawn']);
function ailmentScore(b, sideIdx, opt, mv, foes) {
  const foe = b.sides[1 - sideIdx];
  const tgt = opt.target > 0 ? foe.active[opt.target - 1] : null;
  const pool = tgt ? [tgt] : foes;
  let best = -30;
  for (const t of pool) {
    if (!t || t.fainted || t.hp <= 0) continue;
    let v = SLEEPERS.has(mv.id) ? 62 : mv.id === 'thunderwave' || mv.id === 'glare' ? 34 : 30;
    if (t.status || t.volatiles['substitute']) v = -30;
    else if (POWDERS.has(mv.id) && (t.hasType('Grass') || t.hasItem('safetygoggles') || t.hasAbility('overcoat'))) v = -30;
    else if (mv.id === 'thunderwave' && (t.hasType('Ground') || t.hasType('Electric'))) v = -30;
    else if (mv.id === 'willowisp' && (t.hasType('Fire') || t.hasAbility('guts') || t.hasAbility('flashfire'))) v = -30;
    else if (mv.id === 'toxic' && (t.hasType('Poison') || t.hasType('Steel'))) v = -30;
    else if (SLEEPERS.has(mv.id) && (t.hasAbility('insomnia') || t.hasAbility('vitalspirit') || t.hasAbility('sweetveil'))) v = -30;
    else if (b.field.terrain === 'electricterrain' && SLEEPERS.has(mv.id)) v = -30;
    else if (b.field.terrain === 'mistyterrain') v = -30;
    best = Math.max(best, v);
  }
  return best;
}

function moveScore(b, sideIdx, slot, opt) {
  const side = b.sides[sideIdx];
  const foe = b.sides[1 - sideIdx];
  const me = side.active[slot];
  if (opt.kind === 'pass') return 0;
  if (opt.kind === 'switch') return 2;
  const mv = Dex.moves.get(opt.id);
  const foes = foe.active.filter((p) => p && !p.fainted && p.hp > 0);
  const isAlly = (t) => t < 0;
  if (mv.category !== 'Status') {
    const spread = mv.target === 'allAdjacentFoes' || mv.target === 'allAdjacent';
    const hit = (tgt, mult) => {
      const pct = damagePct(me, tgt, mv) * mult;
      const real = Math.min(pct, (tgt.hp / tgt.maxhp) * 100);
      return real + (pct >= (tgt.hp / tgt.maxhp) * 100 ? 40 : 0);
    };
    let s = 0;
    const prio = mv.priority > 0 ? 1.1 : 1;
    if (spread) {
      for (const f of foes) s += hit(f, 0.75);
      if (mv.target === 'allAdjacent') {
        const ally = side.active[1 - slot];
        if (ally && !ally.fainted) s -= hit(ally, 0.75);
      }
      return s * prio;
    }
    if (isAlly(opt.target)) {
      const ally = side.active[1 - slot];
      return ally && !ally.fainted ? -hit(ally, 1) - 10 : -50;
    }
    const tgt = foe.active[opt.target - 1];
    if (!tgt || tgt.fainted || tgt.hp <= 0) return -50;
    if (mv.id === 'fakeout' || mv.id === 'firstimpression') {
      return me.activeMoveActions === 0 ? hit(tgt, 1) + 75 : -30;
    }
    if (mv.id === 'suckerpunch') return hit(tgt, 1) * 0.7;
    return hit(tgt, 1) * prio;
  }
  // status moves
  switch (mv.id) {
    case 'protect':
    case 'detect':
    case 'spikyshield':
    case 'kingsshield':
    case 'banefulbunker':
    case 'silktrap':
    case 'obstruct':
      return me.volatiles['stall'] ? -20 : 14;
    case 'tailwind':
      return side.sideConditions['tailwind'] ? -20 : 38;
    case 'trickroom': {
      const mine = side.active.filter((p) => p && !p.fainted).reduce((a, p) => a + statOf(p, 'spe'), 0);
      const theirs = foes.reduce((a, p) => a + statOf(p, 'spe'), 0);
      const on = !!b.field.pseudoWeather['trickroom'];
      return on ? (mine > theirs ? 40 : -20) : (mine < theirs ? 42 : -10);
    }
    case 'recover':
    case 'roost':
    case 'softboiled':
    case 'slackoff':
    case 'moonlight':
    case 'morningsun':
    case 'synthesis':
    case 'shoreup':
    case 'milkdrink':
    case 'healorder': {
      // a heal at full health is a wasted turn (a Milotic once Recovered at 202/202 into a Wood Hammer)
      const missing = 1 - me.hp / me.maxhp;
      return missing < 0.2 ? -40 : 70 * missing;
    }
    case 'helpinghand':
      return 12;
    case 'followme':
    case 'ragepowder':
      return 20;
    case 'spore':
    case 'sleeppowder':
    case 'hypnosis':
    case 'sing':
    case 'lovelykiss':
    case 'yawn':
    case 'willowisp':
    case 'thunderwave':
    case 'glare':
    case 'toxic':
      return ailmentScore(b, sideIdx, opt, mv, foes);
    case 'taunt':
      return 18;
    case 'partingshot':
      return 14;
    case 'swordsdance':
    case 'nastyplot':
    case 'calmmind':
    case 'dragondance':
      return 16;
    default:
      return 6;
  }
}

function softmaxPick(rng, scores, temp) {
  let mx = -Infinity;
  for (const s of scores) mx = Math.max(mx, s);
  const w = scores.map((s) => Math.exp((s - mx) / temp));
  const tot = w.reduce((a, v) => a + v, 0);
  let r = rng() * tot;
  for (let i = 0; i < w.length; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return w.length - 1;
}

// A joint choice for one side under the rollout policy.
function policyChoice(b, sideIdx, rng, temp) {
  const picks = [];
  const takenSwitch = new Set();
  for (let slot = 0; slot < 2; slot++) {
    let opts = slotOptions(b, sideIdx, slot).filter((o) => !(o.kind === 'switch' && takenSwitch.has(o.index)));
    if (!opts.length) opts = [{ kind: 'pass', str: 'pass' }];
    const scores = opts.map((o) => moveScore(b, sideIdx, slot, o));
    const o = opts[softmaxPick(rng, scores, temp)];
    if (o.kind === 'switch') takenSwitch.add(o.index);
    picks.push(o);
  }
  return picks;
}

const joinChoice = (picks) => picks.map((p) => p.str).join(', ');
// a side that has no request this step (the foe, while only we must replace a fainted Pokémon) sends ''
const joinFor = (b, sideIdx, picks) => (b.sides[sideIdx].activeRequest ? joinChoice(picks) : '');

// ---------------------------------------------------------------------------------------------
// evaluation and rollouts

function sideValue(side) {
  let v = 0;
  for (const p of side.pokemon) {
    if (p.fainted || p.hp <= 0) continue;
    v += 0.35 + 0.65 * (p.hp / p.maxhp);
  }
  return v;
}

function evaluate(b) {
  if (b.ended) return b.winner === 'us' ? 1 : b.winner === 'them' ? 0 : 0.5;
  const d = sideValue(b.sides[0]) - sideValue(b.sides[1]);
  return 1 / (1 + Math.exp(-1.3 * d));
}

const STEP_ERRORS = { n: 0, first: null };
function step(b, c0, c1) {
  try {
    b.makeChoices(c0, c1);
    return true;
  } catch (e) {
    STEP_ERRORS.n++;
    if (!STEP_ERRORS.first) STEP_ERRORS.first = String((e && e.message) || e).slice(0, 200);
    return false;
  }
}

function playOut(b, rng, depth, temp) {
  for (let d = 0; d < depth && !b.ended; d++) {
    if (b.requestState === 'switch') {
      const c0 = joinFor(b, 0, policyChoice(b, 0, rng, 4));
      const c1 = joinFor(b, 1, policyChoice(b, 1, rng, 4));
      if (!step(b, c0, c1)) return null;
      d--;
      if (b.turn > 40) break;
      continue;
    }
    const c0 = joinFor(b, 0, policyChoice(b, 0, rng, temp));
    const c1 = joinFor(b, 1, policyChoice(b, 1, rng, temp));
    if (!step(b, c0, c1)) return null;
  }
  return evaluate(b);
}

// ---------------------------------------------------------------------------------------------
// the search

const PROTECTS = new Set(['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'silktrap', 'obstruct']);

// The slot's best few options, plus the kinds a damage score under-rates: Protect, the best
// status/field move and the best switch. Those are exactly the plans a greedy ranking never tries.
function diverseOptions(b, sideIdx, slot, k) {
  const opts = slotOptions(b, sideIdx, slot);
  if (opts.length <= k) return opts;
  const scored = opts.map((o) => ({ o, s: moveScore(b, sideIdx, slot, o) })).sort((a, c) => c.s - a.s);
  const picked = scored.slice(0, k).map((x) => x.o);
  const have = new Set(picked);
  const addBest = (pred) => {
    const hit = scored.find((x) => pred(x.o) && !have.has(x.o));
    if (hit) {
      picked.push(hit.o);
      have.add(hit.o);
    }
  };
  const isProtect = (o) => o.kind === 'move' && PROTECTS.has(o.id);
  if (!picked.some(isProtect)) addBest(isProtect);
  if (!picked.some((o) => o.kind === 'switch')) addBest((o) => o.kind === 'switch');
  const isStatus = (o) => o.kind === 'move' && !isProtect(o) && Dex.moves.get(o.id).category === 'Status';
  if (!picked.some(isStatus)) addBest(isStatus);
  return picked;
}

function jointCombos(b, sideIdx, k) {
  const o0 = k ? diverseOptions(b, sideIdx, 0, k) : slotOptions(b, sideIdx, 0);
  const o1 = k ? diverseOptions(b, sideIdx, 1, k) : slotOptions(b, sideIdx, 1);
  const out = [];
  for (const a of o0) {
    for (const c of o1) {
      if (a.kind === 'switch' && c.kind === 'switch' && a.index === c.index) continue;
      out.push([a, c]);
    }
  }
  return out;
}

function comboScore(b, sideIdx, combo) {
  return combo.reduce((s, o, slot) => s + moveScore(b, sideIdx, slot, o), 0);
}

function describe(b, sideIdx, combo) {
  return combo.map((o, slot) => {
    const me = b.sides[sideIdx].active[slot];
    const who = me ? me.species.name : '?';
    if (o.kind === 'pass') return { slot, type: 'pass' };
    if (o.kind === 'switch') return { slot, type: 'switch', species: o.species, from: who };
    return { slot, type: 'move', move_id: o.id, target: o.target, from: who };
  });
}

function search(state, budgetMs, opts) {
  opts = opts || {};
  const t0 = Date.now();
  const rng = mulberry(state.seed || 12345);
  const base = build(state, state.seed || 12345);
  const ours = jointCombos(base, 0, opts.slotK || 6).map((c) => ({ combo: c, pre: comboScore(base, 0, c) }));
  ours.sort((a, b) => b.pre - a.pre);
  // keep the plausible plans, but never crowd out the few distinct move-pairs with duplicates
  const maxOurs = opts.maxOurs || 48;
  let cand = ours.slice(0, maxOurs);
  if (!cand.length) return { ranked: [], rollouts: 0, ms: Date.now() - t0 };

  // The foe's likely joint replies, weighted by how attractive the policy finds them.
  const theirsAll = jointCombos(base, 1, opts.foeK || 5).map((c) => ({ combo: c, pre: comboScore(base, 1, c) }));
  theirsAll.sort((a, b) => b.pre - a.pre);
  const theirs = theirsAll.slice(0, opts.maxTheirs || 16);
  const mx = theirs.length ? theirs[0].pre : 0;
  let tot = 0;
  for (const t of theirs) {
    t.w = Math.exp((t.pre - mx) / 45);
    tot += t.w;
  }
  for (const t of theirs) t.w /= tot || 1;
  if (!theirs.length) theirs.push({ combo: [{ kind: 'pass', str: 'pass' }, { kind: 'pass', str: 'pass' }], w: 1 });

  const baseStr = snapshot(base);
  const depth = opts.depth || 6;
  const stats = cand.map(() => theirs.map(() => ({ sum: 0, n: 0 })));
  let rollouts = 0;
  let rounds = 0;
  const pickReply = () => {
    let r = rng();
    for (let j = 0; j < theirs.length; j++) {
      r -= theirs[j].w;
      if (r <= 0) return j;
    }
    return theirs.length - 1;
  };
  const roll = (i, j) => {
    const c = clone(baseStr);
    c.prng = new PRNG([Math.floor(rng() * 65535), Math.floor(rng() * 65535), Math.floor(rng() * 65535), Math.floor(rng() * 65535)]);
    if (!step(c, joinFor(c, 0, cand[i].combo), joinFor(c, 1, theirs[j].combo))) return;
    const v = playOut(c, rng, depth - 1, 9);
    rollouts++;
    if (v === null) return;
    stats[i][j].sum += v;
    stats[i][j].n += 1;
  };
  const timeLeft = () => budgetMs - (Date.now() - t0);

  // pooled value of plan i: the average over the foe's replies, and its worst reply (shrunk toward an even
  // game so a single unlucky roll cannot condemn a plan)
  const score = (i) => {
    let mix = 0;
    let seen = 0;
    let worst = 1;
    stats[i].forEach((s, j) => {
      if (!s.n) return;
      const v = (s.sum + 1) / (s.n + 2);
      mix += theirs[j].w * v;
      seen += theirs[j].w;
      worst = Math.min(worst, v);
    });
    mix = seen ? mix / seen : 0.5;
    // weight the foe's best reply as much as its average: a plan whose known-bad branch loses
    // hard must not rank #1 on a nice mean (it fed a one-shot-able mon into Overheat in a
    // recorded loss). Replay guard: winning positions pick identical plans at 0.35 and 0.5.
    const wWorst = opts.worstWeight !== undefined ? opts.worstWeight : 0.5;
    return { mix, worst, value: (1 - wWorst) * mix + wWorst * worst };
  };

  // A: screen every plan against sampled replies
  for (let rep = 0; rep < 2 && timeLeft() > 0; rep++) {
    for (let i = 0; i < cand.length && timeLeft() > 0; i++) roll(i, pickReply());
  }
  // B: keep the best, then play each against every reply, round after round
  let alive = cand.map((_, i) => i).sort((x, y) => score(y).value - score(x).value).slice(0, opts.finalists || 12);
  while (timeLeft() > 0 && alive.length) {
    rounds++;
    for (const i of alive) {
      for (let j = 0; j < theirs.length && timeLeft() > 0; j++) roll(i, j);
    }
    if (rounds >= 3 && alive.length > 4) {
      alive = alive.sort((x, y) => score(y).value - score(x).value).slice(0, Math.max(4, Math.ceil(alive.length * 0.6)));
    }
  }

  // Only the finalists are ranked: a screened-out plan has two samples, and the shrinkage toward an even game
  // makes two losses (0.33) look better than twenty (0.05), which ranked untested plans first in lost positions.
  const finalists = new Set(alive);
  const ranked = cand
    .map((c, i) => ({ i, s: score(i), n: stats[i].reduce((a, s) => a + s.n, 0) }))
    .filter((r) => r.n > 0 && (!finalists.size || finalists.has(r.i)))
    .filter((r, _, all) => r.n >= 0.5 * Math.max(...all.map((x) => x.n)))
    .sort((a, b) => b.s.value - a.s.value)
    .map((r) => ({
      choice: joinChoice(cand[r.i].combo),
      slots: describe(base, 0, cand[r.i].combo),
      value: r.s.value,
      mix: r.s.mix,
      worst: r.s.worst,
      samples: r.n,
    }));
  return { ranked, rollouts, failed: STEP_ERRORS.n, firstError: STEP_ERRORS.first, ms: Date.now() - t0, rounds, theirs: theirs.map((t) => ({ choice: joinChoice(t.combo), w: t.w, pre: t.pre })) };
}


// ---------------------------------------------------------------------------------------------
// Team Preview: play each candidate lineup of ours against the foe's likely lineups.
// req = { ours: [[mon x4]], theirs: [{mons:[mon x4], w}], rolls, depth, ms }; a mon is { set }.
function evalLineups(req) {
  const t0 = Date.now();
  const budget = req.ms || 6000;
  const mkSide = (mons) => ({ mons: mons.map((m) => ({ set: m.set, hpFrac: 1 })), conditions: {} });
  const n = req.ours.length;
  const sum = new Array(n).fill(0);
  const wsum = new Array(n).fill(0);
  const worst = new Array(n).fill(1);
  let rolls = 0;
  const rounds = req.rolls || 3;
  for (let r = 0; r < rounds; r++) {
    for (let j = 0; j < req.theirs.length; j++) {
      for (let i = 0; i < n; i++) {
        if (Date.now() - t0 > budget) break;
        const seed = 1000 + r * 977 + i * 131 + j * 17;
        let v;
        try {
          const b = build({ sides: [mkSide(req.ours[i]), mkSide(req.theirs[j].mons)], turn: 1 }, seed);
          v = playOut(b, mulberry(seed ^ 0x9e3779b9), req.depth || 14, 45);
        } catch (e) {
          v = null;
        }
        if (v === null || v === undefined) continue;
        rolls++;
        const w = req.theirs[j].w;
        sum[i] += w * v;
        wsum[i] += w;
        worst[i] = Math.min(worst[i], v);
      }
    }
  }
  return {
    values: sum.map((x, i) => (wsum[i] ? x / wsum[i] : 0.5)),
    worst,
    rollouts: rolls,
    ms: Date.now() - t0,
  };
}

module.exports = { evalLineups, build, search, clone, slotOptions, jointCombos, policyChoice, joinChoice, evaluate, refreshRequest };
