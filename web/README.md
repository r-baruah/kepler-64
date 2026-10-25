# Kepler-64 web

SvelteKit + TypeScript site, prerendered, deployed with `@sveltejs/adapter-vercel`. This pass: design tokens and the Play screen (the landing hero is Play at rest). Specs: `docs/v1/06_WEB_PLAN.md` and the 2026-10-25 entries of `docs/v1/07_DECISION_LOG.md`.

## Run

```bash
cd web
npm install
npm run dev -- --host      # http://localhost:5173/ (sync-data runs first)
npm run build              # sync-data, then vite build
npx vitest run
npx svelte-check
```

`?fen=<FEN>` starts the game from a position (testing, for example a mate in one: `/?fen=6k1/5ppp/8/8/8/8/8/R3K3%20w%20-%20-%200%201`).

## Engine

The engine is `kepler-wasm` built with `--target web` into `src/lib/engine/pkg/`. **That folder is committed**, because Vercel has no Rust. Rebuild it after any change to `engine/crates`:

```bash
./scripts/build-engine.sh
```

The script builds with path remapping (so no machine paths end up in the binary) and writes `pkg/ENGINE_COMMIT`: the last commit that touched the engine sources, with `-dirty` if they have uncommitted changes. A CI check comparing that stamp with the engine sources is a follow-up and does not exist yet.

The engine runs in a Web Worker (`src/lib/engine/engine.worker.ts`). Search is synchronous inside wasm, so it cannot be interrupted; "Move now" terminates the worker, starts a new one from the already-compiled module, replays the game and searches 3,000 nodes. Levels use only a `nodes` limit (Quick 60,000, Default 400,000, Strong 1,500,000), so the move is identical on every device.

## Data

`scripts/sync-data.mjs` (run as `prebuild` and `predev`) copies the generation files and the six preset files from `engine/data/` into `static/data/` and writes `static/data/site-facts.json` (self-play games along the champion lineage, computed and cross-checked against `engine/results/`; preset names, labels and the levels are read from the web plan tables). It fails the build if any source is missing. `static/data/` is generated and gitignored (the root `.gitignore` ignores `data/`), so the build host must be able to read `../engine` and `../docs/v1` (on Vercel, enable "include source files outside of the Root Directory").

## Fonts and pieces

Geist and Geist Mono are self-hosted in `static/fonts/` (SIL Open Font License 1.1, licence texts alongside). Pieces are the cburnett set under the BSD licence; sources and licence text in `static/pieces/LICENSE.md`.

## Landing

Below the hero (shown only at rest): the one law (demo board; the rook's squares come from a second engine worker started once the page engine is ready), the piece-weight chart and the Law Ledger cards (data and evidence paths from `sync-data.mjs`, build fails if a file is missing), the open-source band and footer. The header name returns to the landing (asks first if a game is in progress).

## Not in this pass

Sharing (`/g`, OG images), Watch, Lab, Explore, the explanation cards, the GitHub star count, keyboard play, the end-of-game moments card.
