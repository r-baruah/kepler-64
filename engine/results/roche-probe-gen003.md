# Roche-law diagnostics at Gen 3 (2026-10-24)

Why does the spec-1 Roche limit (L5) not earn its place (F14, F19, F27)? This measures it on real games instead of guessing.

**Data:** the 1,000 gen-003 ladder games (`results/ladder-gen003/`, against Stockfish 19 limited to 1500–2300). Every
position from ply 10 on is evaluated with gen-003 (evaluation only, no search) through the WASM engine, and both kings'
L5 quantities are logged. 193,072 king-positions; 10,724 of them belong to a side that is checkmated within 10 plies.

**Reproduce** (from `engine/`, after building the WASM package as in `crates/kepler-wasm/tests/smoke.mjs`):

    node ../scripts/research/roche/probe.cjs $PWD $PWD/data/generations/gen-003.json $PWD/results/ladder-gen003 rows.json
    python3 ../scripts/research/roche/analyse.py rows.json results/ladder-gen003 data/generations/gen-003.json

**Quantities** (for one king, its body `S_K` = its square plus its step squares, as in L5):
- `eta`: the spec-1 index, enemy pull on the body divided by own pull on it.
- `stress`: enemy pull on the body, `Σ_{s∈S_K} Φ_enemy(s)` (the numerator of `eta`).
- `diff`: stress minus own pull on the body.
- `tidal`: spread of enemy pull across the body, max minus min.
- `lostFrac`: share of the king's escape squares (body squares not holding an own piece) where the enemy pulls harder.
- `onKing`: enemy pull on the king's own square.
- `material`, `score`: from that king's side's point of view (score includes the spec-1 Roche term; section 2 and 3 remove it).

Section 1 samples 20,000 negatives per row (seed 1), so its last digit can vary between runs.

## 1. Separation: kings of the side that gets mated, by plies to mate (AUC)

| plies to mate | subset | n | eta | stress | diff | tidal | lostFrac | onKing | material | score |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-4 | all | 3900 | 0.982 | 0.930 | 0.973 | 0.813 | 0.953 | 0.905 | 0.983 | 0.986 |
| 1-4 | material within 1.5 | 77 | 0.942 | 0.947 | 0.893 | 0.885 | 0.846 | 0.907 | 0.701 | 0.776 |
| 5-8 | all | 3900 | 0.951 | 0.883 | 0.934 | 0.781 | 0.886 | 0.843 | 0.979 | 0.984 |
| 5-8 | material within 1.5 | 121 | 0.910 | 0.911 | 0.862 | 0.856 | 0.827 | 0.819 | 0.728 | 0.851 |
| 9-14 | all | 5845 | 0.893 | 0.832 | 0.866 | 0.743 | 0.802 | 0.787 | 0.964 | 0.971 |
| 9-14 | material within 1.5 | 380 | 0.834 | 0.819 | 0.813 | 0.770 | 0.735 | 0.792 | 0.729 | 0.822 |
| 15-24 | all | 9715 | 0.783 | 0.755 | 0.729 | 0.698 | 0.688 | 0.705 | 0.925 | 0.937 |
| 15-24 | material within 1.5 | 1201 | 0.753 | 0.771 | 0.676 | 0.726 | 0.659 | 0.712 | 0.677 | 0.766 |
| 25-40 | all | 15077 | 0.653 | 0.671 | 0.567 | 0.644 | 0.593 | 0.622 | 0.864 | 0.882 |
| 25-40 | material within 1.5 | 4537 | 0.604 | 0.647 | 0.521 | 0.634 | 0.552 | 0.606 | 0.683 | 0.758 |

## 2. Beyond the rest of the evaluation (20 strata matched on the score without Roche; 0.5 = adds nothing)

| plies to the end | eta | stress | diff | tidal | lostFrac | onKing |
|---|---|---|---|---|---|---|
| 5+ | 0.536 | 0.577 | 0.477 | 0.563 | 0.523 | 0.564 |
| 5-14 | 0.620 | 0.721 | 0.546 | 0.724 | 0.602 | 0.698 |
| 15-40 | 0.552 | 0.612 | 0.472 | 0.591 | 0.527 | 0.593 |

## 3. Held-out outcome fit (logistic on score/T; one third of games held out; decisive games only)

| features | held-out log loss | weights (per unit / T) |
|---|---|---|
| score without Roche | 0.4471 | 1.147 |
| + spec-1 Roche term | 0.4410 | 1.116, 2.141 |
| + tidal spread | 0.4326 | 1.106, 0.914 |
| + stress | 0.4130 | 1.079, 0.365 |
| + tidal spread + stress | 0.4128 | 1.082, -0.229, 0.406 |

91746 positions from 975 decisive games.

## Reading

1. The spec-1 index does detect doomed kings (section 1). Its trouble is elsewhere.
2. Once the rest of the evaluation is held fixed (section 2), `eta` adds little. `stress` and `tidal` add more. `diff` adds nothing: the king's own support is already in the evaluation through L4a and L4b, and dividing by it (as `eta` does) dilutes the part that is new.
3. On game outcomes (section 3), adding `stress` improves held-out log loss about 5.6 times as much as the spec-1 Roche term does (0.0341 vs 0.0061). With `stress` present, `tidal` adds nothing further.

**Caveats:** one generation, games against one opponent family, outcome prediction is not playing strength (F10). Only a
Law Ledger SPRT can say whether a law built on this helps.
