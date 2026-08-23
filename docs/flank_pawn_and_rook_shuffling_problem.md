# Engine Playstyle Problem, Code Diagnostics & Physics Roadmap

## Executive Summary

This document captures the core gameplay problems observed in Kepler-64 (mindless flank pawn pushes, free piece blunders, flat eval bar, and missed checkmates), provides a **deep-dive diagnostic audit of the codebase**, and details a vision-aligned technical roadmap to fix these issues without diluting the engine's pure astrophysics evaluation model.

---

## 1. Observed Gameplay Problems

During self-play and engine matches, Kepler-64 exhibits several severe playstyle flaws:

1. **The "Outward Comet" Flank Pawn Bias ($a4/h4/a5/h5$)**:
   - The King acts as a supermassive black hole ($M = 1000.0$).
   - The pawns on the outer edges ($a$- and $h$-files) sit at the spatial boundaries of the board, furthest from the King's gravity well.
   - Experiencing the weakest attraction, these flank pawns behave like **outward comets**—more free to break away from the gravitational field.
   - Pushing them alters global potential fields without incurring an immediate tactical penalty at shallow search depths.
2. **Free Piece Blunders (Dropping Pawns, Knights, Rooks)**:
   - Kepler frequently moves valuable pieces (Knights, Rooks, Queen) onto attacked squares or leaves threatened pieces hanging, allowing the opponent to capture massive mass for free.
3. **Missed 1-Move Checkmates**:
   - Even when a 1-move checkmate threat or opportunity exists, the engine fails to see it and continues repeating neutral moves or blundering pieces.
4. **Flat, Static Evaluation Bar**:
   - On paper, the evaluator combines complex physical terms (tidal disruption, Plummer potential, binding energy, field entropy).
   - In practice, the evaluation bar is virtually motionless during self-play, fluctuating by negligible decimals ($0.01 - 0.03$).

---

## 2. Code Diagnostics & Root Causes

A thorough audit of the codebase ([minimax.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/search/minimax.py), [evaluate.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/core/evaluate.py), and [constants.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/core/constants.py)) revealed the exact technical root causes:

### Diagnostic 1: Multiverse is Unwired in Live Search
- In [minimax.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/search/minimax.py#L22-L30), `_score_position` accepts `use_multiverse: bool = False`.
- However, inside `negamax`, `best_move`, `_quiesce`, and `root_sweep`, `_score_position` is invoked **without passing `use_multiverse=True` or PRNG keys**.
- In `_quiesce`, child batches use `batch_score` which calls single-instance `score_white` (Layer-1 static evaluation) directly.
- **Finding:** During live gameplay (`RocheEngine.play`), **The Multiverse is 100% dormant / unwired**. The engine operates on a single static physics realization per turn, failing to leverage Bayesian Model Averaging across parallel universes to reject fragile moves.

### Diagnostic 2: Zero-Gain Delta Terms & Low Material Weight (Flat Eval Bar)
- In [constants.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/core/constants.py#L38-L47):
  ```python
  lambda_delta: float = 0.0   # weight of tidal-rate delta term
  com_gain: float = 0.0        # center-of-mass advance delta
  inertia_gain: float = 0.0   # moment-of-inertia delta
  entropy_gain: float = 0.0    # coordination-vs-scatter delta
  mat_gain: float = 0.3        # material weight
  ```
- In [evaluate.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/core/evaluate.py#L187-L230), the move-sensitivity `delta` terms are the **only** terms that compute dynamic score differences between sibling moves.
- Because all four `delta` gains default to `0.0`, they are deactivated.
- Additionally, `mat_gain` is scaled down to `0.3` (capturing a Knight worth mass 3 yields only a $+0.9$ score shift), while binding energy ($\gamma=0.25$) and tidal terms shift by tiny decimals ($0.01 - 0.03$).
- **Finding:** Every legal move evaluates to nearly identical numbers (e.g. `+0.02` vs `+0.03`), causing the evaluation bar to stay flat and motionless during self-play.

### Diagnostic 3: Quiescence Search Blindness & Shallow Horizon
- In `_quiesce` ([minimax.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/search/minimax.py#L129-L133)):
  ```python
  caps = [m for m in board.legal_moves() if board.is_capture(m)]
  if not caps:
      return alpha
  caps = caps[:10]
  ```
- Quiescence search **only considers capture moves** (capped at 10). It does **not** evaluate quiet moves that save hanging pieces or check evasions.
- **Finding:** At depth 3, if Kepler moves a piece to an unsafe square (hanging piece), saving that piece is a *quiet move* (not a capture), so Quiescence search ignores it. The engine does not foresee the opponent taking the piece until the opponent actually plays the capture. And because `mat_gain = 0.3`, losing a piece is not penalized heavily enough compared to minor field potential shifts.

---

## 3. Core Constraint: Preserving the Astrophysics Vision

> **Crucial Rule:** We must **NOT** dilute Kepler-64's core research vision by slapping standard classical Piece-Square Tables (PSTs) or hand-coded chess heuristics onto the evaluator.

Kepler-64 asks a unique research question: *"What if a chess position were evaluated by gravity?"*

If we force standard chess heuristics into the evaluator, we paper over the physical evaluator's flaws rather than solving them through physics. Classical chess concepts (center control, king safety, piece coordination, pawn structure) must **EMERGE natively out of astrophysics equations, proper parameter wiring, and deep search**.

---

## 4. Comprehensive Technical Blueprint & Action Plan

To resolve these issues while remaining 100% true to the physics vision, we execute the following technical plan:

### 1. Wire Up Multiverse in Live Search
- Update `_score_position` in [minimax.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/search/minimax.py) to pass `use_multiverse=True` with PRNG keys.
- Update `_quiesce` and `best_move` so live search evaluates positions across $K=8$ parallel universes, rejecting fragile move blunders.

### 2. Activate Dynamic Delta Gains & Re-balance Material Weight
- Update default parameters in [constants.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/core/constants.py):
  - `com_gain = 1.0` (Center-of-mass advance)
  - `inertia_gain = 0.5` (Moment of inertia compaction)
  - `lambda_delta = 1.0` (Tidal disruption rate)
  - `mat_gain = 1.0` (Full material weight)
- This provides strong dynamic differentiation between sibling moves, making the evaluation bar dynamic and responsive.

### 3. Extend Quiescence Search for Hanging Pieces & Checks
- Modify `_quiesce` in [minimax.py](file:///c:/Users/Ripuranjan%20Baruah/Desktop/Kapler-64/kepler64/search/minimax.py) to include check evasions and evaluate threatened hanging pieces.

### 4. Astrophysics Solutions for Flank Pawns
- **Moment of Inertia ($I = \sum m_i \cdot r_i^2$)**:
  Penalize army spatial dispersion relative to Center of Mass ($\text{CoM}$), stopping overextended $a4/h4$ pushes.
- **King Accretion Shield**:
  Penalize breaching the pawns ($f2, g2, h2$) around the King's $1000\,M$ gravity well, treating shield pushes as an increase in King tidal susceptibility ($\eta_{\text{self}}$).

### 5. Search Engine Optimizations & JAX Tuning
- **Transposition Table (Zobrist Hash)**: Cache evaluated physics states to boost search depth from depth 3 to depth 8–10+.
- **JAX Parameter Tuning (`jax.grad`)**: Optimize physical constants $\theta = \{G, \epsilon, c, \text{roche}, \text{bonus}, \text{kgain}, \gamma, \text{com\_gain}, \text{inertia\_gain}\}$ against 100,000+ Grandmaster PGN games.
