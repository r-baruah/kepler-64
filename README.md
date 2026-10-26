<div align="center">

# Kepler-64

### A tiny universe that learned to play chess.

**[Play it in your browser](https://kepler.ripuranjan.com)** · [The idea](#the-idea-in-30-seconds) · [What happened](#what-happened) · [The universes](#the-universes) · [Open problems](docs/v1/11_OPEN_PROBLEMS.md)

</div>

Richard Feynman once compared physics to watching a game of chess without knowing the rules: you watch long enough and slowly work out the laws underneath.

Kepler-64 runs that the other way round. It already knows the rules of chess. What it doesn't know is how to play, so it was given a universe instead: pieces with mass, a pull that travels across the board, a speed limit, laws. Then the universe was left to work out the rest by playing against itself.

There is no chess knowledge inside. No opening book, no "knights belong in the centre", no king-safety rules, no neural network trained on human games. Just a handful of physical laws, about a dozen constants it learned by itself, and the question of whether that is enough to play chess.

It is.

## The idea in 30 seconds

**1. Pieces have mass.** Every piece has a weight. Nobody told it what a knight or a queen is worth; it learned those weights by playing itself.

**2. Pull travels along the moves.** A piece pulls on the squares it could move to, and the pull stops at the first piece in the way. A rook pulls down its open file; a bishop behind its own pawn pulls on that pawn and no further. In this universe the rules of chess are the shape of space.

**3. Pull has a speed limit.** It reaches at most 2 moves ahead and fades with each extra move: this universe's speed of light.

**4. Squares belong to whoever pulls harder.** That gives the board territory. A piece is in danger when the enemy pulls on it harder than its own side does.

**5. Every law has to earn its place.** Each law is switched off and the universe plays thousands of games against a copy of itself without it. Laws that make it stronger stay. Laws that don't are thrown out, and live on as alternate universes you can still play.

That's the whole judgement. A standard search looks ahead through the moves; the universe says which positions it likes.

This is physics built natively for a chessboard, modelled on the real thing. It's not a claim that chess is astrophysics. The full laws, with every formula, are in [`docs/v1/02_LAWS_SPEC.md`](docs/v1/02_LAWS_SPEC.md).

## What happened

**It worked out sensible piece values by itself.** Started with every piece at the same weight, it settled near knight 2.6, bishop 3.3, rook 4.4, queen 8.2 (pawn = 1). A second run with different games landed in the same place. ([control run A](engine/results/control-equal-masses-a/), [control run B](engine/results/control-equal-masses-b/))

**It holds its own.** Gen 3 scored 53% against Stockfish 19 limited to its 2100 setting, at 30 seconds plus 0.3 per move, over 200 games. That's Stockfish's own strength scale, not a human rating. [The full ladder is below.](#against-stockfish)

**The universe rejected the most literal astrophysics.** The laws taken most directly from space, like a Roche limit that tears a king apart the way a planet tears apart a moon that comes too close, made the engine weaker. Two designs were tried. The universe threw both out. ([the Law Ledger](#the-law-ledger))

**It learned from nothing but itself.** 85,500 games of self-play across four generations, each accepted only after beating its parent in a statistical test. No human games, no engine to copy from. Stockfish is the examiner, never the teacher.

## The universes

Rejected laws aren't deleted. Every universe below is playable on the [website](https://kepler.ripuranjan.com).

| Universe | What's different there |
|---|---|
| **Home** | Ours. The strongest so far (Gen 4). |
| **Genesis** | Every piece was born the same weight, and found its own value by playing itself. |
| **First Light** | Before it learned anything: human piece values, laws set by hand. |
| **Hunger** | Captures absorb weight: a piece grows by eating. The universe played better without it. |
| **Roche** | A king can be torn apart by enemy pull, like a moon too close to its planet. Neither version made it stronger. |
| **Undertow** | Enemy pull drags on the king's whole body. It made the engine weaker. |

Hunger, Roche and Undertow are Home with one law switched back on. They were never trained that way, so they show what the law does, not how strong a universe built around it could be.

## How it works, the simple version

1. **Board to masses.** Each piece becomes a body with its learned weight.
2. **Masses to field.** Pull spreads out along each piece's legal moves, stopping at blockers and fading with distance in moves.
3. **Field to judgement.** The position is scored from three things: the weights on the board, the squares each side holds, and how exposed each piece is to enemy pull.
4. **Search picks the move.** A conventional alpha-beta search looks ahead and asks the universe which future it prefers.

Training adjusts the dozen constants (the weights, how fast pull fades, how much each law counts) so the universe's judgement predicts how its own games turn out. Everything is in Kepler units, where one pawn weighs 1.

## Why there is a v1

The first version (v0, Python and JAX, kept in [`kepler64/`](kepler64/)) used the real formulas of astrophysics almost literally: Newtonian gravity across the board, a tidal tensor, a Roche limit, even gravitational waves. When we measured what actually drove its moves, the headline physics changed the best move in 0% of 120 test positions. It was really playing on material and how close enemy pieces stood to the king ([`docs/v1/01_V0_AUDIT.md`](docs/v1/01_V0_AUDIT.md)).

The lesson was that raw formulas don't fit a chessboard: straight-line distance means nothing when a rook can't see past a pawn. So v1 was rebuilt from scratch in Rust, with physics native to the board: distance counted in moves, pull blocked by pieces, and every law made to prove itself. At equal depth, v1's untrained first universe beat v0 100 games to 0 ([`m1-v1gen000-vs-v0-d3.json`](engine/results/m1-v1gen000-vs-v0-d3.json)).

## How this was built

Kepler-64 is a solo project by someone whose world is space and astronomy, not machine learning or engine development. It was built with AI as a core tool, openly: most of the code was written with Claude (Anthropic) as the coding assistant, the way a growing share of software and research is written today. The concept, the laws, the experiments, and every decision about what stays, what gets cut and what may be claimed are the author's.

That's also why everything here is checkable rather than taken on trust. Every number on this page comes from a committed result file, the laws are written out in full in the spec, and the experiments can be rerun.

## The details

Everything below is for engine developers and the curious. Every number links to the file it comes from.

<details>
<summary><b>Against Stockfish</b></summary>

<a id="against-stockfish"></a>

Gen 3 against Stockfish 19 with `UCI_LimitStrength` at each `UCI_Elo` setting, 30 s + 0.3 s per move, 200 games per rung (100 openings, both colours). Source: [`engine/results/ladder-gen003.json`](engine/results/ladder-gen003.json).

| Stockfish 19 `UCI_Elo` | Kepler W / D / L | Kepler score |
|---|---|---|
| 1500 | 174 / 1 / 25 | 87.3% |
| 1700 | 154 / 6 / 40 | 78.5% |
| 1900 | 123 / 3 / 74 | 62.3% |
| 2100 | 103 / 5 / 92 | 52.8% |
| 2300 | 75 / 10 / 115 | 40.0% |

`UCI_Elo` is Stockfish's own strength setting and does not correspond to human or FIDE ratings. The current champion (Gen 4) has no ladder yet. The opponent on the website plays Gen 4 with a fixed node budget per move, and its strength has not been measured.

</details>

<details>
<summary><b>The learned constants of Home (Gen 4)</b></summary>

From [`engine/data/generations/gen-004.json`](engine/data/generations/gen-004.json):

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

</details>

<details>
<summary><b>Generations</b></summary>

Training is TDLeaf(λ), the method from KnightCap (Baxter, Tridgell and Weaver, 1998), run here in self-play only. A candidate is accepted only by a sequential test (SPRT: elo0 = 0, elo1 = 10, α = β = 0.05, pentanomial model, up to 4,000 games) against the generation it came from. Gains are engine against engine at fixed depth, with 95% intervals, from [`engine/data/generations/index.json`](engine/data/generations/index.json). They are not ratings.

| Gen | What changed | Self-play games | Gain over parent | Evidence |
|---|---|---|---|---|
| 0 (First Light) | Hand-set: human piece values 3/3/5/9, law weights from a one-off calibration | 0 | (the starting point) | [`gen-000.json`](engine/data/generations/gen-000.json), [`calibration-gen000.md`](engine/results/calibration-gen000.md) |
| 1 | TDLeaf from Gen 0, 11 rounds × 500 games, depth 3 | 5,500 | +132 [+98, +168], 400-game match at depth 4, after an SPRT pass at depth 3 | [`confirm-gen001t1-vs-gen000-d4.json`](engine/results/confirm-gen001t1-vs-gen000-d4.json), [`sprt-gen001t1-vs-gen000-d3.json`](engine/results/sprt-gen001t1-vs-gen000-d3.json) |
| 2 | Gen 1 with the two laws the Ledger cut switched off; no new training | 0 | +81 [+71, +92], fixed-depth SPRT against the parent, depth 4 | [`gen001b-vs-gen001.json`](engine/results/gen001b-vs-gen001.json) |
| 3 | TDLeaf from Gen 2, 20 rounds × 2,000 games, depth 4, on GitHub Actions | 40,000 | +53 [+43, +63], fixed-depth SPRT against the parent, depth 4 | [`gen003a-vs-gen002.json`](engine/results/gen003a-vs-gen002.json) |
| 4 (Home) | TDLeaf from Gen 3, 20 rounds × 2,000 games, depth 4; tidal load tried, then switched off | 40,000 | +30 [+20, +40], fixed-depth SPRT against the parent, depth 4 | [`gen004b-vs-gen003.json`](engine/results/gen004b-vs-gen003.json) |

**From equal weights.** Two control runs started with knight, bishop, rook and queen all at 4.43 (so the army weighs the same as with human values) and trained for 30 rounds × 2,000 games at depth 4 with different seeds. Both settled near knight 2.6, bishop 3.3, rook 4.4, queen 8.2; as multiples of the knight, bishop 1.25, rook 1.68, queen 3.11 and 3.12. These are the weights this universe's laws need; they make no claim about "true" piece values. Evidence: [`control-equal-masses-a/`](engine/results/control-equal-masses-a/), [`control-equal-masses-b/`](engine/results/control-equal-masses-b/) (findings F20, F24).

**A correction we made.** The first training run made the engine weaker (−80 Elo, rejected by the gate; [`gen001-light.md`](engine/results/gen001-light.md)). Gen 1 then worked, and at first we credited the learned training temperature. A pre-registered test with three arms from Gen 0 showed otherwise: with quiet leaves and unfinished games dropped, a temperature fixed at 4 gave +156 over Gen 0, against +133 for the learned one, and learned against fixed 4 came out at −34 [−44, −24]. Gen 1's jump came from those two training fixes. Evidence: [`engine/results/attrib/`](engine/results/attrib/) (finding F32).

</details>

<details>
<summary><b>The Law Ledger</b></summary>

<a id="the-law-ledger"></a>

Each law plays against a copy of the same universe without it. The number is the Elo of having the law: positive means it helps. Each row is an SPRT of 4,000 games at depth 4, engine against engine.

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

Binding was cut at Gen 1 and earned its place back at Gen 3, so a verdict holds for the generation it was measured on. Home plays with accretion and tidal load off, and with the spec-1 Roche limit at no measurable effect. One law changed at a time, no re-tuning, engine against engine.

</details>

<details>
<summary><b>Run it</b></summary>

**The UCI engine.** You need Rust; [`engine/rust-toolchain.toml`](engine/rust-toolchain.toml) pins the version.

```bash
cd engine
cargo build --release
./target/release/kepler        # speaks UCI on stdin/stdout
```

In a chess GUI such as Cute Chess or Arena, add `engine/target/release/kepler` as a UCI engine and set its `Generation` option to the full path of `engine/data/generations/gen-004.json`. Without that option the engine uses its built-in hand-set constants, which are not Home.

```
uci
setoption name Generation value /path/to/kepler-64/engine/data/generations/gen-004.json
position startpos
go depth 8
```

Other options: `Hash` (MB), `Depth` (a fixed depth limit), `UCI_ShowBreakdown` (print every term of the evaluation). The `cp` it prints is `round(100 × ku)`, a unit conversion for GUIs. Natively it searches about 1 million nodes per second on one thread of a Ryzen 5 5500U ([`engine/results/perf-m2.md`](engine/results/perf-m2.md)).

**Training, matches and suites.** Everything is a subcommand of `kepler-train`: `cargo run --release -p kepler-train -- <command> --help`. The method, statistics and gates are in [`docs/v1/04_TRAINING_AND_GATES.md`](docs/v1/04_TRAINING_AND_GATES.md).

**Reproducibility.** Heavy runs are split into work units (one binary, a JSON input, a seed, a count, one JSON output) that run on any Linux x86-64 machine ([`docs/v1/08_DISTRIBUTED_COMPUTE.md`](docs/v1/08_DISTRIBUTED_COMPUTE.md)). Three reference units gave the same result hash on the author's laptop and on a GitHub Actions runner, and with 1 and 3 threads ([`engine/data/determinism/reference.json`](engine/data/determinism/reference.json)). Ladder games against Stockfish use clocks, so they are statistics with error bars and are never used for training.

**The website.**

```bash
cd web
npm install
npm run dev
```

The engine runs in the browser as WebAssembly, with no server. On an Android phone (Chrome 155) it reached depth 6 in 333 ms on Kiwipete, 64 ms from the start position and 149 ms in a quiet middlegame ([`engine/results/browser/phone-android-chrome155.json`](engine/results/browser/phone-android-chrome155.json)).

</details>

<details>
<summary><b>Repo layout</b></summary>

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

</details>

## Open problems

Kepler-64 is a small universe with open questions, each with a test for when it's solved ([`docs/v1/11_OPEN_PROBLEMS.md`](docs/v1/11_OPEN_PROBLEMS.md)):

1. **The king problem.** Two Roche-limit designs failed. Find a king-danger law, built only from the field, that the Ledger keeps.
2. **Make accretion work.** Find a version of mass transfer on capture that passes the Ledger, or show why none can.
3. **Any new law.** Borrowed from real physics or invented for this board, judged against the champion.
4. **Explain a result, no code needed.** Why a fixed temperature beats a learned one, and why the self-found piece values land where they do.

The one rule: a new law has to be physics from the field (pull, mass, exposure, held squares), never a chess heuristic ([`02_LAWS_SPEC.md`](docs/v1/02_LAWS_SPEC.md) §9). To take one on, open an issue naming the problem number.

## About the name

The name has layers. **Kepler** is for Johannes Kepler, who found the laws of the planets from observations; this engine finds its constants from its own games. **64** is for the squares of the board. And **Kepler-64b** is a real planet: a gas giant that orbits a pair of stars while a second pair circles far out, the first planet known in a four-star system, spotted in 2012 by volunteers of the Planet Hunters project. A world pulled between suns seemed right for a board pulled between two sides.

## Cite

The metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button reads it.

```bibtex
@software{baruah-kepler64,
  title  = {Kepler-64: a chess engine whose evaluation is a physics-style field learned by self-play},
  author = {Baruah, Ripuranjan},
  url    = {https://github.com/r-baruah/kepler-64}
}
```

## Credits and licence

Made by Ripuranjan Baruah, released under the [MIT licence](LICENSE).

- **Stockfish 19** is the examiner only (the ladder, suite checks, the opening-balance filter), never the teacher.
- **Chess pieces:** the "cburnett" set by Colin M. L. Burnett, BSD 3-clause licence ([`web/static/pieces/LICENSE.md`](web/static/pieces/LICENSE.md)).
- **Fonts:** Geist and Geist Mono, SIL Open Font License.
- **Photograph:** Earthrise, NASA AS17-152-23272 (Apollo 17, 1972), public domain ([`web/static/images/CREDITS.md`](web/static/images/CREDITS.md)).
- **Move generation:** [`cozy-chess`](https://crates.io/crates/cozy-chess) (MIT).
- **Coding assistant:** Claude, by Anthropic.
