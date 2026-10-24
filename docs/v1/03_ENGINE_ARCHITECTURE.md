# 03 · Engine architecture (v1)

Status: agreed 2026-10-23. Everything is in Rust, in one cargo workspace under `engine/`. The v0 Python package `kepler64/` is frozen. It's used only as a baseline opponent in gate matches, then moved to `legacy/` at release.

## 1. Workspace layout

```text
engine/
  Cargo.toml                 workspace; edition 2024; release profile: lto="thin", codegen-units=1, panic="abort"
  crates/
    kepler-core/             no I/O, no threads. Compiles to native and wasm32-unknown-unknown
      src/
        lib.rs
        params.rs            Constants (the 14 of 02_LAWS_SPEC §1), bounds, JSON load/save, generation metadata
        scalar.rs            Scalar trait: f32 for play, Dual<N> (value + N partials) for training
        state.rs             GameState: cozy_chess::Board + accreted mass A[64] + history hashes; make/unmake
        laws/
          mod.rs
          mass.rs            L1
          field.rs           L2 + L3: attack sets, relocations, hop maps, Φ_W/Φ_B
          energy.rs          L4a binding, L4b space
          roche.rs           L5
          accretion.rs       L6 (state transition, called from state.rs)
        eval.rs              evaluate<S: Scalar>(&GameState, &Constants<S>) -> Breakdown<S>, and a fast path for f32
        search/
          mod.rs             iterative deepening driver, time manager, limits (depth/nodes/movetime/clock)
          alphabeta.rs       PVS, null move, LMR, check extension
          qsearch.rs         quiescence (captures + promotions; checks at the first qply)
          ordering.rs        TT move, MVV-LVA by learned rest mass, killers, history
          tt.rs              transposition table (Zobrist from cozy-chess), depth-preferred + always-replace buckets
        replay.rs            Replay v3 export (see §6)
    kepler-uci/              binary `kepler`: UCI protocol, options (Hash, Threads=1, Depth, Generation, MultiPV)
    kepler-train/            binary `kepler-train`: selfplay, tdleaf, match, ladder, ablate, suite, calibrate
    kepler-wasm/             wasm-bindgen bindings (cdylib) for the website
  data/
    suites/                  EPD test suites (tracked)
    openings/                opening suite for matches (tracked)
    generations/             gen-000.json, gen-001.json, … (tracked; the website reads these)
  results/                   committed match and suite results that public numbers link to (tracked)
  benches/                   criterion benches: perft, eval, search nps
```

**Dependencies (all permissive licences, checked 2026-10-23):**
- `cozy-chess` 0.3 (MIT): board representation, legal move generation, magic attack tables, Zobrist. Verified to build for `wasm32-unknown-unknown`.
- `serde`/`serde_json` (data), `wasm-bindgen` (web only), `criterion` (benches only).
- No GPL dependencies. `shakmaty` is GPL-3 and must not be used.

## 2. Ownership of rules vs laws

- **Rules** (move generation, legality, check, mate, draws) come from `cozy-chess` plus `state.rs` (repetition, 50-move rule, insufficient material).
- **Laws** come only from `laws/` and follow `02_LAWS_SPEC.md` exactly.
- `laws/` may use `cozy-chess` attack tables. That's how L2 reads the rules. It must not contain any chess-strategy constant.

## 3. Evaluation design

- **One generic implementation.** `evaluate<S: Scalar>` computes the full breakdown.
  - `S = f32` in play.
  - `S = Dual<12>` in training: value plus the 12 analytically tuned constants, giving exact gradients by forward-mode differentiation. No JAX and no finite differences, except in tests.
- **Fast path:** `eval_f32(&GameState, &Constants<f32>) -> f32` skips the breakdown allocation. A test asserts that it equals `evaluate::<f32>().score_white` bit-for-bit.
- **Field computation** per piece: a hop-1 bitboard from attack tables. For `c ≥ 2`, a hop-2 bitboard as the union of attack sets from each relocation square, computed with the piece lifted off its square. Shell sizes come from popcounts. Pull is then added to `Φ` square by square: a few hundred float adds per side.
- **Incremental updates are a milestone 2 optimisation** (recompute only pieces whose rays touch the changed squares). Milestone 1 recomputes from scratch, which is simpler to verify.

## 4. Game state and hashing

- `GameState` = `cozy_chess::Board`, `A: [f32; 64]` (accreted excess per square, moved with the piece), and a hash history for repetition.
- Make/unmake keeps `A` exactly (`02_LAWS_SPEC.md` §7).
- The transposition-table key is the cozy-chess Zobrist hash (placement, side, castling, en passant) and **excludes `A`**. Two transposed positions with different accreted masses share a TT entry. That's a documented approximation. If the Law Ledger keeps accretion (`f_acc > 0`), measure its effect with a match against a variant that hashes quantised `A`, and record the result in the decision log.

## 5. Search

Standard and physics-neutral. All of these are allowed because they don't encode chess knowledge in the evaluation:
- iterative deepening;
- aspiration windows (from milestone 2);
- PVS;
- transposition table;
- null-move pruning (not in check, not in pawn-only endings);
- LMR;
- check extension;
- quiescence with stand-pat and delta pruning, using learned rest masses;
- move ordering by TT move, then MVV-LVA by learned rest mass, then killers and history.

Mate scores are `±(30000 − ply)` internally. Threads = 1 in milestones 1–3. Lazy SMP is optional later.

**Time management:** `movetime` exact. With a clock: `time/30 + inc·0.8`, with a hard stop at 3× soft. Always return a legal move. With no completed iteration, return the TT or first legal move.

## 6. Interfaces

### 6.1 UCI (`kepler`)
- Standard UCI.
- `info` lines report `depth`, `seldepth`, `nodes`, `nps`, `time`, `pv` and `score cp` (= round(100·ku)) or `score mate`.
- Options:
  - `Hash` (MB, default 64);
  - `Depth` (default 0 = unlimited);
  - `Generation` (default = latest in `data/generations/`, or a path to a JSON);
  - `UCI_ShowBreakdown` (bool; prints `info string` with the four terms and both kings' `η/ρ`).
- Accreted mass is rebuilt from the `position … moves …` list. That's why UCI play is exact.

### 6.2 WASM (`kepler-wasm`), the contract the website codes against
```ts
class Engine {
  constructor(generationJson: string)             // a gen-XXX.json, or a bare constants object
  newGame(fen?: string): void                     // throws on a bad FEN; accreted mass starts at 0
  play(uciMove: string): boolean                  // false if illegal, or if the game is over
  legalMoves(): string[]                          // UCI, castling as e1g1
  state(): { fen: string; turn: 'w'|'b'; result: '1-0'|'0-1'|'1/2-1/2'|null; reason: string|null;
             inCheck: boolean; ply: number; halfmoveClock: number }
  breakdown(): Report                             // 02_LAWS_SPEC §10, a plain JS object
  search(limits?: { depth?: number; movetimeMs?: number; nodes?: number; nps?: number }): SearchResult
  setConstants(partialJson: string): void         // merged into the current constants; throws if out of bounds
  constantsJson(): string                         // the current constants
  free(): void
}
type SearchResult = { best: string; scoreKu: number /* side to move */; mateIn: number|null;
                      depth: number; seldepth: number; nodes: number; pv: string[] }
```
- `reason` is one of `checkmate`, `stalemate`, `fifty_move_rule`, `insufficient_material`, `threefold_repetition`. The wrapper keeps the hash history, so repetition is exact, and plays through `Position::play`, so accreted mass is carried.
- `breakdown()`, `state()` and `search()` return plain objects (`serde-wasm-bindgen`, JSON-compatible, no `Map`s). `Report` (`kepler-core::eval::Report`) has `terms {material, bind, space, roche, score_white}`, `field {white[64], black[64]}` (square index a1 = 0 ... h8 = 63), `pieces[] {square, piece, color, rest_mass, accreted, carried_mass, emission, energy, hops[3][]}`, `kings[2] {square, body[], stress, binding, eta, gauge, break_ratio}`, `constants`, `generation`. Per-side `energy` sums to the evaluator's `U_W` and `U_B`.
- **Time limits.** `std::time::Instant` panics on `wasm32-unknown-unknown`, so the wasm build has no clock (`kepler_core::clock::Instant` is a stub whose `elapsed()` is 0). `movetimeMs` is therefore turned into a node budget, `nodes = movetimeMs * nps / 1000`, with `nps` defaulting to a conservative 150,000 (override it per call once a device is measured). The search stops at the end of the iteration that reaches the budget, or inside it (`Limits.nodes`); the result is the last completed depth. With no limit at all it searches depth 4.
- The engine runs in a Web Worker. Search is synchronous inside the worker. Cancelling means terminating the worker, since there is no clock or stop flag to poll.
- Build: `cargo build -p kepler-wasm --release --target wasm32-unknown-unknown`, then `wasm-bindgen --target web|nodejs --out-dir crates/kepler-wasm/pkg <wasm>` (CLI version must equal the `wasm-bindgen` crate, 0.2.100). Node smoke test: `node crates/kepler-wasm/tests/smoke.mjs`.
- The UI can call `breakdown()` at any time to draw exactly what the engine computes.

### 6.3 Replay v3 (data the website shows in Watch mode)
JSON, written compactly with numbers rounded to four decimals:
- `{ schema: 3, spec: "v1.0-spec-N", generation, game: {headers}, initialFen, initialBreakdown, result, reason, highlights, plies: [...] }`
- Each ply: `{ ply (1-based), san, uci, fenAfter, breakdown (02 §10, of the position after the move), search: { scoreKu, pov: "white", mateIn, depth, nodes, best, pv } }`
- `search` is run from the position **before** the move, at the replay's fixed depth. `scoreKu` is from White's point of view. For PGN games `best` can differ from the move played; for self-play the move is `best`, except in the first exploration plies.
- Highlights:
  - `firstBreakPly` and `firstBreakColor` (first ply after which a king's `η/ρ ≥ 1`);
  - `biggestSwingPly` and `biggestSwingKu` (the move after which the white-point-of-view search score changed most; scores clamped to ±50 ku);
  - `heaviest` (`{ square, M, piece, color }` after the last ply);
  - `captures` (`{ ply, square, captor, victim, massGained }`).
- `result` is `1-0`, `0-1`, `1/2-1/2` or `*`; `reason` is the game-end reason, `ply_limit` for a capped self-play game, or `game_record_ends` for a PGN that stops early.

Written by `kepler-train replay --gen <json> (--pgn <file> [--game N] | --selfplay --seed S) --depth D --plies N --out <dir>`. v2 (the Python schema) is retired with v0.

## 7. Performance budget (Ryzen 5 5500U, single thread)

| milestone | target |
|---|---|
| M1 | perft(5) start position < 1 s; eval ≥ 200k positions/s; search ≥ 150k nps |
| M2 | search ≥ 1M nps native; WASM ≥ 30% of native; depth 6 in < 1 s in the browser on a mid-range phone |

Measure with `cargo bench`, and record each milestone's numbers in `engine/results/perf-<milestone>.md`.

## 8. Local build rules (8 GB RAM machine)

- `export PATH=$HOME/.cargo/bin:$PATH`. Rust is installed at user level by rustup.
- Build with `cargo build --release -j6`. Never run more than one release build at a time.
- WASM: `cargo build -p kepler-wasm --release --target wasm32-unknown-unknown`, then `wasm-bindgen` (installed from crates.io into the user's cargo bin when milestone 2 needs it).
- Stockfish for matches: set `KEPLER_STOCKFISH` to the binary path. Locally it's in `~/.local/opt/stockfish/`. Never commit absolute paths.
- The v0 baseline engine: set `KEPLER_V0_UCI` to a command that starts `python -m kepler64.uci` inside the v0 virtualenv.

## 9. Tests (all in CI from milestone 1)

- **perft:**

  | position | depth | nodes |
  |---|---|---|
  | start | 5 | 4,865,609 |
  | Kiwipete | 4 | 4,085,603 |
  | pos 3 `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 5 | 674,624 |
  | pos 4 | 4 | 422,333 |
  | pos 5 | 4 | 2,103,487 |
  | pos 6 `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 4 | 3,894,594 |

- Law invariants 1–8 from `02_LAWS_SPEC.md` §11.
- **Search sanity:**
  - mate in 1 and mate in 2 suites must be 100% solved at depth 4;
  - no illegal move in 1,000 random self-play games at depth 2.
- **UCI smoke test:** `uci`, `isready`, `position startpos moves e2e4`, `go depth 4`, a `bestmove` line.
- CI: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --release`. The Python `tests.yml` stays, for v0, until v0 moves to `legacy/`.
