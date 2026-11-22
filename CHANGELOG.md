# Changelog

All notable changes to Kepler-64 are recorded here. Every number points to the committed file it comes from.

## [1.0.0-beta.1] - 2026-11-23

First public beta of v1, the Rust rewrite. Kepler-64 is an experiment in playing chess with a standard lookahead search and an evaluation made only of four physics formulas built for a 2D board. There is no neural network, no opening book and no hand-written chess heuristic in the evaluation. Self-play is used only to tune its constants and to test each law.

It is a beta because the website and documentation are still being finished (see "Still open" below).

### Downloads

| Archive | Platform |
|---|---|
| `kepler-v1.0.0-beta.1-linux-x86_64.tar.gz` | Linux x86-64 (glibc 2.35 or newer) |
| `kepler-v1.0.0-beta.1-windows-x86_64.zip` | Windows x86-64 |
| `kepler-v1.0.0-beta.1-macos-arm64.tar.gz` | macOS on Apple silicon |

Each archive holds the UCI engine (`kepler`, or `kepler.exe` on Windows), `gen-004.json` (Home, the Gen 4 champion), `LICENSE` and `USAGE.txt`. Add the binary to any UCI chess GUI, then set its `Generation` option to the full path of `gen-004.json`. Without that option the engine plays with built-in provisional constants, which are not Home. From a terminal:

```
uci
setoption name Generation value /full/path/to/gen-004.json
position startpos
go depth 6
```

`SHA256SUMS.txt` has the checksums. The macOS binary is not signed: if macOS refuses to run it, clear the quarantine flag once with `xattr -d com.apple.quarantine ./kepler`.

No download needed to play: [kepler.ripuranjan.com](https://kepler.ripuranjan.com) runs the same engine in the browser.

### Engine
- Rust workspace: `kepler-core` (board, laws, search, exact gradients), `kepler-uci`, `kepler-train` and `kepler-wasm`, implementing [`v1.0-spec-2`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/docs/v1/02_LAWS_SPEC.md). Home is a spec-1 universe (tidal load at 0).
- The evaluation is four formulas in [`laws/mod.rs`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/crates/kepler-core/src/laws/mod.rs): a field whose pull travels along legal move lines, stops at the first piece in the way, spreads over the squares it reaches (`1 / n_h^σ`) and ends past light speed `c = 2` moves; rest mass; binding energy `M_q (Φ_enemy − Φ_own)`; and held space `tanh(Φ_W − Φ_B)`.
- Search: PVS, quiescence, transposition table, null move, LMR, check extension, killers and history. About 1 million nodes per second on one thread of a Ryzen 5 5500U ([`perf-m2.md`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/results/perf-m2.md)).
- Training: TDLeaf(λ) in self-play with exact forward-mode derivatives (`Dual` numbers). A candidate is accepted only by SPRT against its parent. Training work units give the same result hash on different machines and thread counts ([`reference.json`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/data/determinism/reference.json)).

### Results
- **Home** ([`gen-004.json`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/data/generations/gen-004.json)): knight 2.72, bishop 3.12, rook 4.24, queen 7.39 (pawn = 1), `κ` 2.63, `α` 0.54, `σ` 0.58, `c` 2, `λ_space` 0.126, `λ_bind` 0.019. +30 [+20, +40] Elo over Gen 3 at depth 4. Generations 1 to 4 used 85,500 self-play games in total. Home also still has the spec-1 Roche limit switched on (`β` 2.75), where the Ledger measures no effect: −6 [−16, +4] Elo.
- **The physics against counting material alone**, with the identical search: +330 [+316, +345] Elo at depth 4 over 4,000 games, a score of 87% ([`ledger-gen001.md`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/results/ledger-gen001.md)). At equal time (100 ms per move), where material-only searches deeper, untrained Gen 0 still scored 68.8%, +137 [+91, +188] over 200 games ([`m1-gate.md`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/results/m1-gate.md)).
- **Law Ledger** (one law switched at a time, 4,000-game SPRT at depth 4). Kept: the field, spreading (+129), held space (+111), light speed 2 over 3 (+14). Binding energy was cut at Gen 1 (−15) and came back at Gen 3 (+35). Rejected: accretion (−103), the spec-1 Roche limit (−71 at Gen 1) and spec-2 tidal load (−24 when trained in).
- **Equal-mass controls:** two runs starting knight, bishop, rook and queen at the same weight both settled near knight 2.6, bishop 3.3, rook 4.4, queen 8.2 ([run A](https://github.com/r-baruah/kepler-64/tree/v1.0.0-beta.1/engine/results/control-equal-masses-a), [run B](https://github.com/r-baruah/kepler-64/tree/v1.0.0-beta.1/engine/results/control-equal-masses-b)).
- **Stockfish 19 ladder for Gen 3** (`UCI_LimitStrength`, 30 s + 0.3 s per move, 200 games per rung): 87.3% against `UCI_Elo` 1500, 78.5% against 1700, 62.3% against 1900, 52.8% against 2100 and 40.0% against 2300 ([`ladder-gen003.json`](https://github.com/r-baruah/kepler-64/blob/v1.0.0-beta.1/engine/results/ladder-gen003.json)). `UCI_Elo` is Stockfish's own scale, not a human rating. Gen 4 has no ladder yet.
- Against v0, the frozen Python prototype, untrained Gen 0 won 100 games to 0 at depth 3.

### Website
- Play in the browser at [kepler.ripuranjan.com](https://kepler.ripuranjan.com). The engine runs as WebAssembly in a Web Worker, with no server; depth 6 took at most 333 ms on an Android phone.
- Six universes to play: Home, Genesis, First Light, Hunger, Roche and Undertow. Live pull and light-cone overlays, PGN export, light and dark themes, and a citation sheet with BibTeX.

### Still open before 1.0.0
- Website pages Watch, Explore and Lab, and a new v1 Book.
- Moving the frozen v0 prototype (`kepler64/`) to `legacy/`.
- A Stockfish ladder for Gen 4, and measured strength for the browser's node-budget levels.
