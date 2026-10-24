// Depth-6 search from the start position in Node, best of 3 fresh engines.
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const { Engine } = createRequire(import.meta.url)('../pkg/kepler_wasm.js');
const gen = readFileSync(fileURLToPath(new URL('../../../data/generations/gen-002.json', import.meta.url)), 'utf8');
let best = Infinity;
for (let i = 0; i < 3; i++) {
  const e = new Engine(gen);
  const t = performance.now();
  const r = e.search({ depth: 6 });
  const ms = performance.now() - t;
  best = Math.min(best, ms);
  console.log(`depth ${r.depth} nodes ${r.nodes} best ${r.best} ${ms.toFixed(0)} ms`);
  e.free();
}
console.log(`node depth 6 best of 3: ${best.toFixed(0)} ms`);
