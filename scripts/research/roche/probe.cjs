// Replays ladder games with a generation (evaluation only) and logs, per king per ply,
// the spec-1 Roche quantities plus alternative indices computed from the same field.
const root = process.argv[2], genFile = process.argv[3], gamesDir = process.argv[4], out = process.argv[5];
const { Engine } = require(root + '/crates/kepler-wasm/pkg/kepler_wasm.js');
const fs = require('fs');
const e = new Engine(fs.readFileSync(genFile, 'utf8'));
const sq = s => (s.charCodeAt(0) - 97) + 8 * (s.charCodeAt(1) - 49);
function occupancy(fen) {
  const occ = new Array(64).fill(null);
  fen.split(' ')[0].split('/').forEach((row, i) => {
    let f = 0; const r = 7 - i;
    for (const ch of row) { if (/\d/.test(ch)) f += +ch; else { occ[f + 8 * r] = ch; f++; } }
  });
  return occ;
}
const rows = [];
for (const file of fs.readdirSync(gamesDir).filter(f => f.endsWith('.json')).sort()) {
  const d = JSON.parse(fs.readFileSync(gamesDir + '/' + file, 'utf8'));
  d.games.forEach((g, gi) => {
    let fen = g.opening.trim(); if (fen.split(' ').length < 6) fen += ' 0 1';
    e.newGame(fen);
    // who got mated: result 1-0 means black mated (when reason is checkmate)
    const mated = g.reason === 'checkmate' ? (g.result === '1-0' ? 'black' : g.result === '0-1' ? 'white' : null) : null;
    const n = g.moves_uci.length;
    for (let p = 0; p <= n; p++) {
      if (p > 0 && !e.play(g.moves_uci[p - 1])) break;
      if (p < 10) continue;
      const b = e.breakdown(), occ = occupancy(e.state().fen);
      b.kings.forEach((k, ki) => {
        const color = ki === 0 ? 'white' : 'black';
        const own = b.field[color], enemy = b.field[color === 'white' ? 'black' : 'white'];
        const isOwn = ch => ch && ((ch === ch.toUpperCase()) === (color === 'white'));
        const body = k.body.map(sq), ks = sq(k.square);
        const en = body.map(s => enemy[s]), ow = body.map(s => own[s]);
        // escape squares: body squares other than the king's, not holding an own piece
        const esc = body.filter(s => s !== ks && !isOwn(occ[s]));
        const lost = esc.filter(s => enemy[s] > own[s]).length;
        rows.push({ file, gi, ply: p, toEnd: n - p, color, doomed: mated === color && n - p <= 10 ? 1 : 0,
          eta: k.eta, gauge: k.gauge, stress: k.stress, binding: k.binding,
          diff: k.stress - k.binding,
          tidal: Math.max(...en) - Math.min(...en),          // spread of enemy pull across the body
          tidalNet: Math.max(...en.map((x, i) => x - ow[i])) - Math.min(...en.map((x, i) => x - ow[i])),
          escN: esc.length, lost, lostFrac: esc.length ? lost / esc.length : 1,
          onKing: enemy[ks], mat: (color === 'white' ? 1 : -1) * b.terms.material, score: (color === 'white' ? 1 : -1) * b.terms.score_white });
      });
    }
  });
}
fs.writeFileSync(out, JSON.stringify(rows));
console.log('rows', rows.length, 'doomed', rows.filter(r => r.doomed).length);
