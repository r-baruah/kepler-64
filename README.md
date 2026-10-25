# Kepler-64

**A tiny universe that learned to play chess.**

Kepler-64 is a chess engine with no chess knowledge in it. We gave chess its own physics, built natively for the board and modelled on real physics. Every piece has a mass and pulls along the lines it can legally move. The pull stops at the first piece in the way and reaches at most 2 moves, which is this universe's speed of light. About a dozen constants (the piece masses, how fast pull fades, the weight of each law) were learned by playing itself. Nothing else about chess is written in.

- **It still plays.** Gen 3 scored 53% against Stockfish 19 limited to its 2100 setting, at 30 s + 0.3 s per move, over 200 games. [The full ladder is below](#it-knows-space-not-chess).
- **Every law has to earn its place.** Each one is switched off and tested over thousands of games. Laws taken straight from astrophysics, like a Roche limit that tears a king apart, made the engine weaker and were rejected. They live on as alternate universes you can play.
- **Every number is checkable.** Each result on this page comes from a committed file in [`engine/results/`](engine/results/), and the training and test work units give identical results on different machines.

<!-- TODO (maintainer): kepler.ripuranjan.com is not live yet. Confirm the URL before launch. -->
**[Play in the browser](https://kepler.ripuranjan.com)** · [Law Ledger](#every-law-has-to-earn-its-place) · [Findings](docs/v1/09_FINDINGS.md) · [Open problems](docs/v1/11_OPEN_PROBLEMS.md)

<!-- TODO (maintainer): add a 1600x900 light-mode screenshot of the landing hero as docs/v1/img/hero.png, then uncomment:
![The Kepler-64 board with each side's pull drawn on the squares](docs/v1/img/hero.png)
-->

## It knows space, not chess.

**Not in the evaluation:**
- piece-square tables
- king-safety rules
- pawn-structure rules
- an opening book
- endgame tablebases
- a neural network trained on chess games

**In the evaluation:** one field, about a dozen learned numbers, and the rules of chess, which decide where pull can travel. The search around it is a standard alpha-beta search (iterative deepening, PVS, quiescence, transposition table, null move, LMR). Search techniques are allowed because they say nothing about what a position is worth.

**How it plays.** Gen 3, the generation that has been measured, against Stockfish 19 with `UCI_LimitStrength` at each `UCI_Elo` setting, 30 s + 0.3 s per game, 200 games per rung (100 openings, both colours). Source: [`engine/results/ladder-gen003.json`](engine/results/ladder-gen003.json).

| Stockfish 19 `UCI_Elo` | Kepler W / D / L | Kepler score |
|---|---|---|
| 1500 | 174 / 1 / 25 | 87.3% |
| 1700 | 154 / 6 / 40 | 78.5% |
| 1900 | 123 / 3 / 74 | 62.3% |
| 2100 | 103 / 5 / 92 | 52.8% |
| 2300 | 75 / 10 / 115 | 40.0% |

So Gen 3 is roughly level with Stockfish limited to 2100 at this time control, and below it at 2300. `UCI_Elo` is Stockfish's own strength setting, calibrated on its own scale, and it does not correspond to human or FIDE ratings. The current champion (Gen 4) has no ladder yet. The opponent on the website plays Gen 4 with a fixed node budget per move, and its strength has not been measured.

## Physics, rebuilt for a chessboard

Each idea from real physics has an exact counterpart on the board:

| Real physics | On the chessboard |
|---|---|
| Mass | Every piece has a weight (pawn = 1), learned by self-play. |
| Gravity | A piece pulls along the lines it can legally move, and the pull stops at the first piece in the way. |
| Light speed | Pull takes moves to arrive. It reaches at most 2 moves away and fades with each extra move. |
| Field | Each square belongs, softly, to whichever side pulls on it harder. |
| Exposure | A piece is exposed when the enemy pulls on it harder than its defenders do. |

Distance is counted in moves, never in squares: a rook covers its whole open file in one hop, and a bishop behind its own pawn pulls on that pawn and no further. This is a field on the board's move graph, with no Euclidean distance anywhere. Words like "weight", "pull" and "light speed" each name a quantity with a formula. The full laws, constants and invariants are in [`docs/v1/02_LAWS_SPEC.md`](docs/v1/02_LAWS_SPEC.md).

**The learned constants of Home (Gen 4)**, from [`engine/data/generations/gen-004.json`](engine/data/generations/gen-004.json):

| Constant | Value | What it means |
|---|---|---|
| pawn | 1 | the unit (fixed) |
| knight `m_n` | 2.7 | knight weight, in pawns |
| bishop `m_b` | 3.1 | bishop weight |
| rook `m_r` | 4.2 | rook weight |
| queen `m_q` | 7.4 | queen weight |
| `kappa` | 2.63 | how hard the king itself pulls |
| `alpha` | 0.54 | share of pull kept for each extra move of travel |
| `sigma` | 0.58 | how much a piece's pull spreads thin over the squares it reaches (0 = none, 1 = conserved flux) |
| `c` | 2 | light speed: the most moves pull can travel |
| `lambda_space` | 0.126 | weight of held space (squares owned by the stronger pull) |
| `lambda_bind` | 0.019 | weight of exposure (enemy pull on a piece against its defenders') |
| `beta`, `rho`, `k_roche` | 2.75, 1.56, 3.34 | the spec-1 Roche limit: weight, breaking point and sharpness. Training grew the weight, but switching it off makes no measurable difference (see the Ledger) |
| `f_acc` | 0 | accretion (captures absorbing weight): off |
| `lambda_tide` | 0 | tidal load on the king (spec-2): off |

All scores are in Kepler units (ku), where 1 ku is the weight of one pawn.

## How it learned

**Self-play only.** Training is TDLeaf(λ), the method from KnightCap (Baxter, Tridgell & Weaver, 1998), run here in self-play only: the engine plays itself and nudges each constant so that its evaluation predicts the next positions and the final result better. No human games, no Stockfish labels, no opening book. Stockfish is the examiner, never the teacher.

**Every generation has to beat its parent.** A candidate is accepted only by a sequential test (SPRT: elo0 = 0, elo1 = 10, α = β = 0.05, pentanomial model, up to 4,000 games) against the generation it came from. Gains below are engine against engine at fixed depth, 95% interval, from [`engine/data/generations/index.json`](engine/data/generations/index.json):

| Gen | What changed | Self-play games | Gain over parent | Evidence |
|---|---|---|---|---|
| 0 (First Light) | Hand-set: human piece values 3/3/5/9, law weights from a one-off calibration ([`calibration-gen000.md`](engine/results/calibration-gen000.md)) | 0 | none (the starting point) | [`gen-000.json`](engine/data/generations/gen-000.json) |
| 1 | TDLeaf from Gen 0, 11 rounds × 500 games, depth 3 | 5,500 | +132 [+98, +168], 400-game match at depth 4, after an SPRT pass at depth 3 | [`confirm-gen001t1-vs-gen000-d4.json`](engine/results/confirm-gen001t1-vs-gen000-d4.json), [`sprt-gen001t1-vs-gen000-d3.json`](engine/results/sprt-gen001t1-vs-gen000-d3.json) |
| 2 | Gen 1 with the two laws the Ledger cut switched off (accretion, binding); no new training | 0 | +81 [+71, +92], fixed-depth SPRT against the parent, depth 4, 4,000 games | [`gen001b-vs-gen001.json`](engine/results/gen001b-vs-gen001.json) |
| 3 | TDLeaf from Gen 2, 20 rounds × 2,000 games, depth 4, on GitHub Actions | 40,000 | +53 [+43, +63], fixed-depth SPRT against the parent, depth 4, 4,000 games | [`gen003a-vs-gen002.json`](engine/results/gen003a-vs-gen002.json) |
| 4 (Home) | TDLeaf from Gen 3, 20 rounds × 2,000 games, depth 4; tidal load tried, then switched off | 40,000 | +30 [+20, +40], fixed-depth SPRT against the parent, depth 4, 4,000 games | [`gen004b-vs-gen003.json`](engine/results/gen004b-vs-gen003.json) |

In all, Home's lineage learned from 85,500 games of self-play (the `training.games` fields of the generation files). These gains are against the parent only. They are not ratings.

**From equal weights.** As a control, two runs started with the knight, bishop, rook and queen all at the same weight (4.43, so the army weighs the same as with human values) and trained for 30 rounds × 2,000 games at depth 4, with different seeds. Both settled near knight 2.6, bishop 3.3, rook 4.4, queen 8.2, and the two runs agree: as multiples of the knight, bishop 1.25, rook 1.68, queen 3.11 and 3.12. The order queen > rook > bishop > knight came out of self-play alone. These are the weights this universe's laws need; they make no claim about "true" piece values, and the absolute scale is lower than the human one. Evidence: [`control-equal-masses-a/`](engine/results/control-equal-masses-a/), [`control-equal-masses-b/`](engine/results/control-equal-masses-b/) (findings F20, F24).

**A correction we made.** The first training run made the engine weaker (−80 Elo, rejected by the gate; [`gen001-light.md`](engine/results/gen001-light.md)). Gen 1 then worked, and at first we credited the learned training temperature. A pre-registered test with three arms from Gen 0 showed otherwise: with quiet leaves and unfinished games dropped, a temperature fixed at 4 gave +156 over Gen 0, against +133 for the learned one, and learned against fixed 4 came out at −34 [−44, −24]. Gen 1's jump came from those two training fixes. The temperature's value matters (fixed 6 against fixed 4 is −70), and learning it lands near a good value, slowly. Evidence: [`engine/results/attrib/`](engine/results/attrib/) (finding F32).

## Every law has to earn its place

The Law Ledger plays each law against a copy of the same universe without it. The number is the Elo of having the law: positive means the law helps. Each row is an SPRT of 4,000 games at depth 4, engine against engine, and names the generation it was measured on.

| Law | Gen | Elo of having it [95%] | Verdict | Evidence |
|---|---|---|---|---|
| The field as a whole (all its terms), against counting material alone | 1 | +330 [+316, +345] | Kept | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| Spreading (`sigma`) | 1 | +129 [+118, +140] | Kept | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| Held space | 1 | +111 [+101, +122] | Kept | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| Light speed 2 moves, against 3 | 1 | +14 [+4, +24]; 1 move ties 2 (+3 [−7, +13]) | Kept at 2 | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| King emission (`kappa`) | 1 | +8 [−1, +18] | Inconclusive | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| Learned masses, against 3/3/5/9 | 1 | +6 [−4, +16] | Inconclusive | [`ledger-gen001.md`](engine/results/ledger-gen001.md) |
| Binding (exposure) | 1, then 3 | −15 [−25, −5] at Gen 1; +35 [+25, +45] at Gen 3 | Came back | [`ledger-gen001.md`](engine/results/ledger-gen001.md), [`ledger-gen003.md`](engine/results/ledger-gen003.md) |
| Accretion (captures absorb weight) | 1 | −103 [−113, −93] | Rejected | [`ledger-gen001-acc0.json`](engine/results/ledger-gen001/ledger-gen001-acc0.json) |
| Roche limit, spec-1 (a king torn apart) | 1, 3, 4 | −71 [−81, −61] switched on at Gen 1; −9 [−19, +1] at Gen 3; −6 [−16, +4] at Gen 4 | Rejected | [`ledger-gen001.md`](engine/results/ledger-gen001.md), [`ledger-gen003-roche0.json`](engine/results/ledger-gen003-roche0.json), [`ledger-gen004-roche0.json`](engine/results/ledger-gen004-roche0.json) |
| Tidal load, spec-2 (enemy pull on the king's body) | 3, 4 | added to Gen 3 at weights 0.05 / 0.10 / 0.34: −31 / −77 / −248; trained in from 0: −24 [−34, −14] | Rejected | [`ledger-gen003-tide005.json`](engine/results/ledger-gen003-tide005.json), [`ledger-gen003-tide010.json`](engine/results/ledger-gen003-tide010.json), [`ledger-gen003-tide034.json`](engine/results/ledger-gen003-tide034.json), [`ledger-gen004tide-tide0.json`](engine/results/ledger-gen004tide-tide0.json) |

Reading it: the field is worth a lot, and spreading and held space do most of the work. Binding was cut at Gen 1 and earned its place back at Gen 3, so a verdict holds for the generation it was measured on. We tried two physical laws for a king being torn apart; neither made the engine stronger. Home plays with accretion and tidal load off, and with the spec-1 Roche limit at no measurable effect. The caveats are in each file: one law changed at a time, no re-tuning, engine against engine.

**Rejected laws live on as alternate universes.** Their formulas stay in the spec, and you can switch them back on and play them on the website:

| Universe | Its rule |
|---|---|
| **Home** | Our universe. The strongest so far (Gen 4). |
| **Genesis** | Every piece was born the same weight. It found its own values by playing itself, with accretion on. |
| **First Light** | Before it learned anything: human piece values, laws set by hand. |
| **Hunger** | Captures absorb weight: a piece grows by eating. Rejected, it played better without. Home's constants with this law switched on. |
| **Roche** | A king can be torn apart by enemy pull, like a moon too close to its planet. Two versions were tried; neither made the engine stronger. Home's constants with this law turned up. |
| **Undertow** | Enemy pull drags on the king's whole body. Rejected, it made the engine weaker. Home's constants with this law switched on. |

Hunger, Roche and Undertow are variants of Home that were never trained themselves.

## Why v1 exists

Kepler-64 v0 (Python and JAX, frozen in [`kepler64/`](kepler64/)) had a physics story built on a tidal tensor, a Roche limit and gravitational waves. When we measured it, removing those headline terms changed the best move in 0% of 120 test positions; v0 really played on material and straight-line distance to the king ([`docs/v1/01_V0_AUDIT.md`](docs/v1/01_V0_AUDIT.md)). Its field also passed through pieces and ignored how they move, and a bug in its capture rule distorted the material count. So v1 was rebuilt from scratch in Rust, with a field shaped by the movement rules and every law tested in the Ledger. At equal depth, v1's untrained Gen 0 beat v0 100 games to 0 ([`m1-v1gen000-vs-v0-d3.json`](engine/results/m1-v1gen000-vs-v0-d3.json)).

## Run it

### The UCI engine

You need Rust; [`engine/rust-toolchain.toml`](engine/rust-toolchain.toml) pins the version and rustup picks it up.

```bash
cd engine
cargo build --release
./target/release/kepler        # speaks UCI on stdin/stdout
```

To use it in a chess GUI such as Cute Chess or Arena, add `engine/target/release/kepler` as a UCI engine and set its `Generation` option to the full path of `engine/data/generations/gen-004.json`. Without that option the engine uses its built-in hand-set constants, which are not Home. From a terminal:

```
uci
setoption name Generation value /path/to/kepler-64/engine/data/generations/gen-004.json
position startpos
go depth 8
```

Other options: `Hash` (MB), `Depth` (a fixed depth limit), `UCI_ShowBreakdown` (print every term of the evaluation). The engine thinks in ku; the `cp` it prints is `round(100 × ku)`, a unit conversion for GUIs that expect it.

Natively it searches about 1 million nodes per second on one thread of a Ryzen 5 5500U laptop (985,000 from the start position and 1,015,000 on Kiwipete at depth 6; [`engine/results/perf-m2.md`](engine/results/perf-m2.md)).

### Training, matches and suites

Everything is a subcommand of `kepler-train`: `cargo run --release -p kepler-train -- <command> --help`. The method, statistics and gates are in [`docs/v1/04_TRAINING_AND_GATES.md`](docs/v1/04_TRAINING_AND_GATES.md). Two examples, from [`engine/results/suites-per-generation.md`](engine/results/suites-per-generation.md) and [`docs/v1/08_DISTRIBUTED_COMPUTE.md`](docs/v1/08_DISTRIBUTED_COMPUTE.md) §4:

```bash
# run a test suite at a fixed depth (from engine/)
./target/release/kepler-train suite --engine ./target/release/kepler --opt Generation=data/generations/gen-00K.json --suite data/suites/SUITE.epd --depth D --out results/suite-SUITE-gen00K-dD.json

# one batch of an SPRT between two generations
kepler-train sprt-shard --a new.json --b old.json --openings balanced-200.epd --pairs 50 --seed <S_k> --depth 4 --threads 4 --out s-k.json
```

### Reproducibility

Heavy runs (training generations, the Ledger, the ladder) are split into work units: one binary, one or two JSON inputs, a seed and a count, writing one JSON file. Units can run on any Linux x86-64 machine, and the results merge back into `engine/results/` ([`docs/v1/08_DISTRIBUTED_COMPUTE.md`](docs/v1/08_DISTRIBUTED_COMPUTE.md)).

The determinism contract (rules D1 to D8 in §3a: pinned toolchain, pure-Rust maths, depth or node limits only, seeded randomness, a canonical result hash) is checked by three reference units. They gave the same `result_hash` on the maintainer's laptop (Ryzen 5 5500U, Fedora) and on a GitHub Actions runner (EPYC 7763, Ubuntu), and with 1 and 3 threads ([`engine/data/determinism/reference.json`](engine/data/determinism/reference.json), finding F12). Rerun the check with:

```bash
# from engine/
./target/release/kepler-train determinism-check --reference data/determinism/reference.json --engine-dir .
```

Ladder games against Stockfish use clocks and Stockfish's own randomness, so they are statistics with error bars and are never used for training.

### The website

```bash
cd web
npm install
npm run dev
```

The engine runs in the browser as WebAssembly (`kepler-wasm`), with no server. The built package is committed in `web/src/lib/engine/pkg/`; see [`web/README.md`](web/README.md) to rebuild it. Gen 3 in Chrome 155 on an Android phone (the browser hides the exact model) reached depth 6 in 333 ms on Kiwipete, 64 ms from the start position and 149 ms in a quiet middlegame; Chromium 156 on the Ryzen laptop took 307, 55 and 134 ms ([`engine/results/browser/phone-android-chrome155.json`](engine/results/browser/phone-android-chrome155.json), [`desktop-linux-chromium156.json`](engine/results/browser/desktop-linux-chromium156.json)). Node counts match the native engine at the same depth (finding F34).

## Open problems and contributing

Kepler-64 is a small universe with open questions. Four are written up in [`docs/v1/11_OPEN_PROBLEMS.md`](docs/v1/11_OPEN_PROBLEMS.md), each with a test for when it is solved:

1. **The king problem.** Two Roche-limit designs failed; find a king-danger law, built only from the field, that passes the Ledger.
2. **Make accretion work.** Find a version of mass transfer on capture that passes the Ledger, or show why none can.
3. **Any new law.** Borrowed from real physics or invented for this board, judged by the Ledger against the champion.
4. **Explain a result (no code needed).** Why a fixed temperature beats a learned one, and why the self-found piece values land where they do.

The one rule: a new law must be physics from the field, a quantity of pull, mass, exposure or held squares ([`02_LAWS_SPEC.md`](docs/v1/02_LAWS_SPEC.md) §9). No chess heuristics. To take one on, open an issue naming the problem number.

## Repo layout

```text
engine/              v1, a Rust workspace
  crates/
    kepler-core/     board, laws, search, exact gradients
    kepler-uci/      the UCI engine (binary: kepler)
    kepler-train/    self-play, TDLeaf, SPRT, ladder, suites, work units
    kepler-wasm/     the engine for the browser
  data/              generations, opening suite, test suites, ablations
  results/           every published number, as committed files
docs/v1/             specs and plans: laws, architecture, training, findings
web/                 the website (SvelteKit)
kepler64/            v0 (Python + JAX), frozen baseline
scripts/audit/       the v0 audit scripts
```

## Cite

If you use Kepler-64 in research or benchmarks, please cite it. The metadata is in [`CITATION.cff`](CITATION.cff), and GitHub's "Cite this repository" button reads it.

```bibtex
@software{baruah-kepler64,
  title  = {Kepler-64: a chess engine whose evaluation is a physics-style field learned by self-play},
  author = {Baruah, Ripuranjan},
  url    = {https://github.com/r-baruah/kepler-64}
}
```

## About the name

The name has layers. **Kepler** is for Johannes Kepler, who found the laws of the planets from observations; this engine finds its constants from its own games. **64** is for the squares of the board. And **Kepler-64b** is a real planet: a gas giant that orbits a pair of stars while a second pair circles far out, the first planet known in a four-star system, spotted in 2012 by volunteers of the Planet Hunters project. A world pulled between suns seemed right for a board pulled between two sides.

## Credits and licence

Kepler-64 is made by Ripuranjan Baruah and released under the [MIT licence](LICENSE).

- **Stockfish 19** is the examiner only: the ladder, suite verification and the opening-balance filter. It is never the teacher.
- **Chess pieces:** the "cburnett" set by Colin M. L. Burnett, BSD 3-clause licence ([`web/static/pieces/LICENSE.md`](web/static/pieces/LICENSE.md)).
- **Fonts:** Geist and Geist Mono, SIL Open Font License.
- **Photograph:** Earthrise, NASA AS17-152-23272 (Apollo 17, 1972), public domain ([`web/static/images/CREDITS.md`](web/static/images/CREDITS.md)).
- **Move generation:** [`cozy-chess`](https://crates.io/crates/cozy-chess) (MIT).
