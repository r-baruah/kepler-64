# Test suites per generation

Each cell is the number of positions where the engine's move equals a `bm` move, out of the suite size. The engine is `kepler` (release build), one thread, fixed depth, generation file `engine/data/generations/gen-00<k>.json`. Raw rows are in `engine/results/suite-<suite>-gen<k>-d<depth>.json`. Run from `engine/`:

    ./target/release/kepler-train suite --engine ./target/release/kepler --opt Generation=data/generations/gen-00K.json --suite data/suites/SUITE.epd --depth D --out results/suite-SUITE-gen00K-dD.json

| suite | depth | gen-000 | gen-001 | gen-002 | gen-003 |
|---|---|---|---|---|---|
| tactics-own | 4 | 154/200 | 149/200 | 157/200 | 160/200 |
| tactics-own | 6 | 172/200 | 171/200 | 172/200 | 174/200 |
| mate1 (mate in 1) | 4 | 100/100 | 100/100 | 100/100 | 100/100 |
| mate1 | 6 | 100/100 | 100/100 | 100/100 | 100/100 |
| mate2 (mate in 2) | 4 | 100/100 | 100/100 | 100/100 | 100/100 |
| mate2 | 6 | 100/100 | 100/100 | 100/100 | 100/100 |
| v0_closed_blunders (bm) | 4 | 6/12 | 8/12 | 7/12 | 9/12 |
| v0_closed_blunders (bm) | 6 | 7/12 | 8/12 | 8/12 | 7/12 |

On v0_closed_blunders no generation plays any of the 12 `am` moves (0/12 at both depths), which is the milestone 1 bar.

The mate suites are not discriminating between generations: search finds a forced mate of this length regardless of the learned constants, so 100/100 at depth 4 is the expected result and acts as a soundness check, not a progress number. The mate positions come from random playouts and are mostly untidy, so they say little about tactical strength in realistic play.

`tactics-own` is built by `kepler-train suite-build-tactics` from Kepler's own gen-002 ladder games (see the file header): positions where Stockfish 19 at depth 16 finds one move at least 2 pawns better than any other. Stockfish is only the examiner; the suite is never used for training. Bias: the positions are drawn from games gen-002 played against strength-limited Stockfish (UCI_Elo 1320 to 2100), so they reflect those games' mistakes, not chess at large. The spread between generations (about 6 positions) is about one standard error, so this suite does not yet separate generations either (findings F28).

The Win At Chess set was dropped: no copy with a stated licence was found.
