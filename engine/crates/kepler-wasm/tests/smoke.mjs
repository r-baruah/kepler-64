// Node smoke test for the kepler-wasm package.
//   cargo build -p kepler-wasm --release --target wasm32-unknown-unknown
//   wasm-bindgen --target nodejs --out-dir pkg <target>/wasm32-unknown-unknown/release/kepler_wasm.wasm
//   node tests/smoke.mjs
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
const { Engine } = require('../pkg/kepler_wasm.js');
const genPath = fileURLToPath(new URL('../../../data/generations/gen-002.json', import.meta.url));
const gen = readFileSync(genPath, 'utf8');

const eng = new Engine(gen);
eng.newGame();
assert.equal(eng.legalMoves().length, 20);
assert.equal(eng.play('e2e5'), false, 'illegal move rejected');
assert.equal(eng.play('e2e4'), true);

let s = eng.state();
assert.equal(s.turn, 'b');
assert.equal(s.result, null);
assert.ok(s.fen.startsWith('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b'));

const r = eng.search({ depth: 4 });
assert.ok(eng.legalMoves().includes(r.best), 'best move is legal');
assert.ok(r.depth >= 4 && r.nodes > 0 && Array.isArray(r.pv) && r.pv[0] === r.best);
assert.equal(typeof r.scoreKu, 'number');
console.log(`depth 4 after 1.e4: best ${r.best} score ${r.scoreKu.toFixed(3)} ku, ${r.nodes} nodes`);

const bd = eng.breakdown();
assert.equal(bd.field.white.length, 64);
assert.equal(bd.field.black.length, 64);
assert.equal(bd.kings.length, 2);
assert.equal(bd.pieces.length, 32);
assert.equal(bd.generation, 2);
assert.ok(typeof bd.terms.score_white === 'number');
assert.ok(bd.kings.every((k) => typeof k.break_ratio === 'number' && k.body.length >= 4));
assert.ok(bd.pieces.every((p) => Array.isArray(p.hops) && p.hops.length === 3));
const sumTerms = bd.terms.material + bd.terms.bind + bd.terms.space + bd.terms.roche + bd.terms.tide;
assert.ok(Math.abs(sumTerms - bd.terms.score_white) < 1e-4);

// movetimeMs is a node budget.
eng.newGame();
const t = eng.search({ movetimeMs: 50, nps: 100000 });
assert.ok(t.nodes <= 6000, `node budget respected (${t.nodes})`);

// setConstants
eng.setConstants('{"rho": 2.0}');
assert.equal(eng.breakdown().constants.rho, 2.0);
assert.equal(eng.breakdown().constants.lambda_tide, 0);
eng.setConstants('{"lambda_tide": 0.34}');
assert.ok(Math.abs(JSON.parse(eng.constantsJson()).lambda_tide - 0.34) < 1e-6);
assert.throws(() => eng.setConstants('{"lambda_tide": 3}'));
eng.setConstants('{"lambda_tide": 0}');
assert.throws(() => eng.setConstants('{"rho": 99}'));
assert.throws(() => eng.newGame('not a fen'));

// 40 plies of self-play at depth 2, no illegal moves.
eng.newGame();
let plies = 0;
for (; plies < 40; plies++) {
  s = eng.state();
  if (s.result) break;
  const res = eng.search({ depth: 2 });
  assert.ok(eng.play(res.best), `illegal move ${res.best} at ply ${plies}`);
}
console.log(`self-play: ${plies} plies, result ${eng.state().result}`);
assert.ok(plies === 40 || eng.state().result);

// mate in one is reported and the game ends
eng.newGame('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1');
const m = eng.search({ depth: 3 });
assert.equal(m.best, 'a1a8');
assert.equal(m.mateIn, 1);
assert.ok(eng.play('a1a8'));
assert.equal(eng.state().result, '1-0');
assert.equal(eng.state().reason, 'checkmate');

eng.free();
console.log('kepler-wasm smoke test passed');
