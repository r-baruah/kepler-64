# Speed pass M2: evaluation and search throughput

Machine: AMD Ryzen 5 5500U (Zen 2, 6 cores / 12 threads), Linux, rustc 1.99.1, release profile
(`lto = "thin"`, `codegen-units = 1`), **single thread**, default Gen 0 constants (`c = 2`).
No profiler (`perf`, `valgrind`) was available. Timing came from `kepler-core/examples/bench.rs`
(best of 5 fresh searches, 64 MB hash; evaluations: best of 8 windows of 300 ms over 4,000
seeded random-play positions) and from temporary `rdtsc` counters around the phases of one
evaluation (removed again). The laptop was shared with other builds, so the stage-by-stage
figures below are approximate; the before/after rows were measured back to back, three runs
each, and agreed within about 1%.

## Before and after

| Measure | Before | After | Speed-up |
|---|---|---|---|
| nodes/s, depth 6, start position (38,404 nodes) | 480,000 | 985,000 | 2.05x |
| nodes/s, depth 6, Kiwipete (144,603 nodes) | 445,000 | 1,015,000 | 2.28x |
| evaluations/s (`eval_f32`, 4,000 random-play positions) | 452,000 | 1,035,000 | 2.29x |

Node counts, best moves and score bits are identical before and after (see below).

## What changed, and roughly what each step bought

Stage figures are from the shared machine; read them as sizes, not as exact steps.

1. **Lean `f32` evaluation path** (`laws/fast.rs`, called from the body of `eval_f32`). No
   `Breakdown`, no cone `Vec`, no 64-square `field` copy for output; cones and field accumulation are
   fused in the same visit order (colour, piece type, square), so every field square receives
   its additions in the same order. `n^sigma` comes from a 65-entry table (rebuilt only when
   sigma changes) instead of one `powf` per piece and hop. About 452k to 650k evaluations/s.
2. **Exact evaluation cache in the searcher** (`search/evalcache.rs`): 65,536 direct-mapped
   entries keyed by the full Zobrist hash and a hash of the accreted-mass array, both stored in
   full, so a hit returns the stored `f32` unchanged. Dropped when the constants change. Hit rate
   is 12% (start position) to 27% (Kiwipete) within one fresh search. Worth about 3-5% of nodes/s.
3. **Batched exact `tanhf`** (`laws/tanh.rs`) for the space term. The 64 `tanhf` calls
   dominated (about 40% of an evaluation) and `libm::tanhf` branches on its argument. For
   |x| above 0.2554 (about 92% of squares) the `libm` path is reproduced as straight-line
   arithmetic, operation for operation, with branches turned into selects, so it vectorises.
   Smaller arguments and non-finite values call `libm::tanhf`. The sum over squares is
   unchanged (ascending squares, one accumulator). The fast range is checked against
   `libm::tanhf` for every `f32` bit pattern from 0.2554 to 10 (43.7M values) plus a stride
   sample above 10 and 200,000 random 64-lane batches. About 650k to 820k evaluations/s.
4. **Hop-2 by flood fill** (`laws/field.rs`). For bishops, rooks and queens, hop 2 is computed
   for all relocation squares at once with Kogge-Stone fills instead of one magic lookup each.
   This is exact because everything the lifted piece reaches along its own line is already
   hop 1 or its own square, and nothing else depends on the lifted piece (argument in the code
   comment). Knights, kings and pawns use plain shifts. Hop 3 (`c = 3`) is unchanged. Checked
   against the original definition on 8,000 random occupancies x 64 squares x all pieces x
   `c` in {2, 3} (`fast_hops_match_reference`; a 300,000-occupancy run passed before the
   count was lowered to keep the test fast). `hops_of` is allocation-free (the old frontier
   `Vec` is gone). About 820k to 890k evaluations/s with the shifts and the next item.
5. **Binding energy from a mass table** filled while visiting pieces (no `piece_on` per body),
   same square order and same arithmetic.
6. **Quiescence move generation made lazy** (`search/mod.rs`). Mate/stalemate only needs "is
   there a legal move" (early-exit generation); the capture list is built after the stand-pat
   cutoff and in the same order as filtering the full list. Pruning and ordering untouched.
   About 5% of nodes/s.
7. **CPU-feature dispatch**. `score_white_f32` runs the same code compiled with AVX2, POPCNT
   and BMI1 when the CPU has them (hardware popcount/bit-scan, wider vectors). FMA is not
   enabled, so floating-point results are bit-identical to the baseline build; a test compares
   the baseline build with `eval_f32` for all 8,000 golden evaluations. About 890k to 1,035k
   evaluations/s. On other CPUs and on WASM the portable build is used (slower, same bits).

## Bit-equality evidence

`kepler-core/tests/eval_golden.rs` with `tests/golden/eval_golden.txt`, generated **before** any
change (and not regenerated since):

- 2,000 seeded random-play positions (accretion carried by `Position::play`) x four constant
  sets (Gen 0 `c=2`; `gen-002.json` with `c=1,2,3`): `score_white` bits from `evaluate::<f32>`
  and the side-to-move score bits from `eval_f32`.
- 100 `evaluate::<Dual>` results (value and all 12 partials, bits).
- 40 searches (20 positions x Gen 0 and gen-002), depth 5, fresh searcher each: best move,
  score bits and node count.

All pass unchanged. `kepler-train determinism-check --reference data/determinism/reference.json`
prints OK for `tdleaf-shard`, `sprt-shard` and `evolve-shard`. `cargo test --release -p kepler-core`
passes; `cargo clippy --all-targets -- -D warnings` is clean.

## Tried and not kept

- A memo of `tanhf` results keyed by exact argument bits (4,096 entries): exact, but on random
  positions fewer than half of the 131k sampled arguments are distinct and the table is small,
  so the gain was small; replaced by the batched version.
- Nothing that changed a bit was kept. Nothing reached the golden test and failed: the two
  riskier changes (tanh and hop-2 fill) were each verified against the original by their own
  exhaustive/randomised test before the golden test was run.

## Where the time goes now (cycles per evaluation, approximate, before the last two steps)

hops and field accumulation about 45%, space term about 30%, material/binding/Roche about 10%,
the rest is search overhead. The 1M nodes/s target of `03_ENGINE_ARCHITECTURE.md` §3 is met on
this machine in a single thread at depth 6 on both positions, with the AVX2 build.
