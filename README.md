# Kepler-64

**A physics-inspired, differentiable chess evaluator built with JAX.**

[![Tests](https://img.shields.io/badge/tests-33%20passing-2f855a)](#current-status)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-differentiable-orange)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-2448b8)](LICENSE)

Kepler-64 asks an unreasonable but testable question:

> What if a chess position were evaluated by gravity?

It plays standard legal chess with conventional alpha-beta search. The unusual part is its static evaluator: pieces contribute softened gravitational fields, king pressure is measured with force and finite-difference tidal terms, captures retain mass, and 13 scalar coefficients can be optimized through JAX.

**Status: research prototype.** Kepler-64 is currently weak at chess. The interesting result is an inspectable evaluator and training pipeline, not competitive playing strength or a claim that chess is literally astrophysics.

- [Run it](#quickstart)
- [Generate an interactive decision replay](#decision-replay)
- [Read the evaluator](kepler64/core/evaluate.py)
- [Inspect the tests](kepler64/tests)
- [See what is active and experimental](#implementation-status)

## Why this exists

Most chess engines separate search from a position evaluator. Kepler-64 keeps conventional chess rules and search, but replaces a conventional piece-square evaluator with a compact set of differentiable, physics-inspired features.

That boundary matters:

- **Discrete:** legal move generation, captures, checkmate, alpha-beta, quiescence, sorting.
- **Differentiable:** gravitational fields, tidal measurements, binding, material and structural terms, outcome loss, policy-ranking loss.

The project is useful as an experiment in interpretable evaluator design: every score can be decomposed into named terms, trained constants can be persisted, and each candidate move can be recorded for later inspection.

## Decision replay

The replay generator produces a self-contained technical artifact containing:

- the chessboard and selected move;
- the gravitational potential field and tidal stress;
- ranked legal candidates and native Kepler scores;
- evaluation shifts and weighted term contributions;
- search depth, node counts, cutoffs, and elapsed time;
- interactive HTML, GIF, PNG frames, summary chart, and JSON.

Generate a short self-play replay:

```bash
python scripts/generate_replay.py \
  --depth 1 \
  --candidates 8 \
  --max-plies 6 \
  --output replay-output
```

Or analyze a PGN:

```bash
python scripts/generate_replay.py path/to/game.pgn \
  --game 0 \
  --depth 2 \
  --candidates 8 \
  --max-plies 40 \
  --output replay-output
```

Open `replay-output/index.html` in a browser. Higher search depths are substantially slower because every legal root move receives a full-window search.

The replay is evidence of observability, not evidence of playing strength. Native Kepler scores are not centipawns.

## How the evaluator works

### Board representation

The 64 squares are fixed 2D coordinates. Pieces begin with signed masses based on conventional chess values:

| Piece | Absolute mass |
|---|---:|
| Pawn | 1 |
| Knight | 3 |
| Bishop | 3 |
| Rook | 5 |
| Queen | 9 |
| King | 1000 |

The large king mass makes the king easy to identify in the mass vector; diagonal self-energy is removed from binding calculations so the king does not overwhelm every positional relationship.

### Softened field

The evaluator computes a Plummer-softened potential and force field on the board lattice. A sigmoid distance gate controlled by `c` limits interaction reach:

```text
gate(distance) = sigmoid(c - distance)
```

`c` is an interaction-reach parameter inspired by finite propagation. It is not a stateful speed of light and does not make threats arrive several plies later.

### Tidal index

The potential field is sampled on the board and a finite-difference tidal tensor is evaluated at each king. Its largest eigenvalue, `lambda_max`, defines the board-scale tidal index:

```text
eta = lambda_max * Rg^3 / mref^2
```

This is an engineered dimensionless normalization inspired by Roche/Hill scaling. It is not a physically calibrated Roche limit.

### Score decomposition

From White's perspective, the live evaluator combines:

```text
score = enemy_king_tidal_stress
      - own_king_tidal_stress
      + force_on_enemy_king
      - force_on_own_king
      + binding_energy_difference
      + material_difference
      + optional parent-to-child delta terms
```

The delta terms measure what a move changed: tidal stress, mass-centroid advance, attack concentration, and mass-distribution entropy. Their default values are neutral unless present in a trained constants artifact.

### Capture accretion

Captured mass does not disappear completely. The capturer retains 80% of the captured piece's mass:

```text
new_mass = capturer_mass + sign(capturer) * 0.8 * abs(captured_mass)
```

This state persists through later moves and is handled consistently in search, training data, and replay analysis. Per-piece radius growth and non-king disruption are not implemented game mechanics.

## Training

The trainer optimizes 13 scalar leaves:

```text
G, eps, c, roche, bonus, kgain, gamma, Rg, mat_gain,
lambda_delta, com_gain, inertia_gain, entropy_gain
```

Two supervised objectives are available:

1. **Outcome loss:** fit a position score to win, draw, or loss labels.
2. **Policy loss:** rank an expert move above its legal alternatives.

Phase 1 can run outcome-only; Phase 2 can add policy supervision. Gradients flow through the evaluator and loss, not through legal move generation or alpha-beta decisions.

The current `trained_constants.json` is an experimental artifact. A launch-quality training run still needs a versioned dataset manifest, game-level validation split, multi-seed report, and checkpoint-selection criteria.

## Current status

| Evidence | Current state |
|---|---|
| Unit tests | 33 passing |
| Python compilation | Passing |
| Legal chess | Standard moves and checkmate rules |
| Elo | Not measured |
| Reproducible latency benchmark | Not published |
| Public training metrics | Not published |
| Competitive strength | Not established |

Run the verification suite:

```bash
pytest -q
python -m compileall -q kepler64 scripts
```

The current tests cover field calculations, tidal behavior, board conversion, constants, mass transitions, evaluator decomposition, and replay recording. Broader randomized differential move-generation tests against `python-chess` are still needed.

## Implementation status

| Component | Status |
|---|---|
| Gravitational potential and force fields | Active |
| Finite-difference tidal terms | Active |
| Force, binding, material, and move-delta terms | Active |
| Alpha-beta, quiescence, null-move pruning | Active |
| Capture accretion | Active |
| Outcome and policy training | Active |
| Interactive replay and score decomposition | Active |
| Parameter-perturbation ensemble | Optional, not default play |
| Leapfrog/Verlet rollout | Experimental, not wired into evaluation |
| Online observer update | Experimental, not wired into play |
| Lorentz-inspired mass scaling | Experimental, not wired into play |
| Image-derived constant initialization | Optional experiment |
| Per-piece radius and non-king disruption | Not implemented |
| BayesElo ladder | Not implemented |

Keeping these maturity levels separate is intentional. A module's presence in the repository does not imply that it affects the default engine.

## Quickstart

```bash
git clone https://github.com/r-baruah/kepler-64.git
cd kepler-64
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -e ".[dev]"
```

Play one move:

```python
import chess

from kepler64 import RocheEngine

engine = RocheEngine()
board = chess.Board()

move = engine.play(board, depth=2)
print(move)
```

Evaluate a position through the adapter:

```python
import chess

from kepler64 import Board, RocheEngine

engine = RocheEngine()
board = Board.from_chess(chess.Board())

score = engine.evaluate(board)
print(score)
```

## Repository map

```text
kepler64/
  core/          gravity, tidal math, evaluator, board and mass transitions
  search/        alpha-beta, quiescence, move ordering
  training/      loaders, losses, optimization and ablations
  analysis/      per-position and per-game decision recording
  viz/           glass-box plots and stakeholder replay generation
  match/         UCI match harness
  multiverse/    optional and experimental Layer 2 modules
  tests/         unit and regression tests
scripts/
  train.py
  generate_replay.py
```

## Limitations

- The physics is an inductive bias, not a claim that chess obeys astrophysical laws.
- Piece masses begin from conventional chess material values.
- The evaluator includes an explicit material term.
- Search and move generation are conventional and discrete.
- The distance gate is static; it does not propagate a historical gravity wave.
- Some experimental modules are not integrated into default play.
- The trained artifact does not yet have a published provenance report.
- No Elo, statistically meaningful match result, or synchronized performance benchmark is currently claimed.

## Questions worth testing

The next useful results are falsifiable comparisons:

- Do trained gravitational and tidal terms rank expert moves above chance?
- Do they add predictive value beyond material alone?
- Does capture accretion improve or harm move quality?
- Are learned constants stable across datasets and random seeds?
- Does fixed-shape JAX evaluation improve wall-clock search throughput?
- Which evaluator terms survive held-out validation and ablation?

Contributions that improve measurement, baselines, data provenance, search correctness, training stability, or replay clarity are especially welcome.

## Why keep building it?

Kepler-64 began as a deliberately unreasonable question: what would a chess evaluator look like if its primitive vocabulary were mass, potential, binding, and tidal stress rather than piece-square tables?

It is not a strong engine yet. It is a compact experiment in whether an interpretable, differentiable physical model can learn anything useful about chess. If that premise makes you smile and then makes you want to inspect the code, the project is doing its job.

## License

[MIT](LICENSE)
