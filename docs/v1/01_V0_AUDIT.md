# 01 · Audit of the v0 engine (2026-10-23)

This is why v1 exists. v0 is the Python + JAX engine in `kepler64/` (branch `backend/improvements`, commit `fa8ae8c`), measured with its trained constants (`kepler64/training/trained_constants.json`). Every number below can be reproduced with the scripts in `scripts/audit/`, run from a checkout that has the v0 virtualenv:

| script | what it measures |
|---|---|
| `scripts/audit/v0_terms.py` | how much each evaluation term contributes, and how often removing it changes the chosen move |
| `scripts/audit/v0_launder.py` | material through a capture-recapture line (accretion bug) |
| `scripts/audit/v0_closed.py` | moves v0 chooses in closed positions (depth 3) |
| `scripts/audit/v0_probe.py` | quick-mate traps and seconds per move |

Stockfish 19 (depth 18) was the referee for move quality. The resulting positions are in `engine/data/suites/v0_closed_blunders.epd`.

---

## 1. Which terms decide moves

Setup: 120 positions from random play (8–50 plies), and every legal move scored against its parent, so the move-sensitive terms are live.

| term | mean size (ku) | spread across sibling moves | best move changes if removed |
|---|---|---|---|
| force at enemy king (sigmoid gauge, bonus ≈ 300) | 62.5 | 26.8 | **22.5%** |
| force at own king | 44.2 | 27.0 | **25.8%** |
| material | 13.4 | 4.1 | **17.5%** |
| centre-of-mass advance | 0.76 | 1.13 | 3.3% |
| attack moment of inertia | 0.63 | 0.96 | 1.7% |
| Verlet tidal drift | 0.02 | 0.04 | 0.8% |
| tidal tensor / Roche η (enemy) | 0.010 | 0.006 | **0%** |
| tidal tensor / Roche η (own) | 0.008 | 0.007 | **0%** |
| Δη (tidal rate) | 0.010 | 0.020 | 0% |
| entropy | 0.015 | 0.035 | 0% |
| binding energy | 0 | 0 | 0% |
| gravitational waves (Peters–Mathews) | 0 | 0 | 0% |
| Schwarzschild overlap | 0 | 0 | 0% |

Random positions are noisier than real games, but the gap is far too large for that to change the conclusion.

**Findings:**
- v0 effectively plays on two things: material, and how close enemy pieces are to the king in straight-line distance (a sigmoid with a ≈300-unit scale).
- The tidal tensor and Roche η, the centre of the public story, never changed a move.
- About 5 of the 17 constants do measurable work: `eps`, `c`, `roche`, `bonus`, `kgain`, `mat_gain`, plus a little `com_gain` and `inertia_gain`.
- `mat_gain` sits near its hand-set lower bound (1.5) and `entropy_gain` near its upper bound (6.0). The bounds are holding those values, not learning.
- The trained-constants file stores 14 of the 17 leaves. `lambda_gw`, `lambda_sch` and `dt_drift` are absent, and they were 0 / unused anyway.

## 2. Closed positions: blunders

v0 at depth 3 played both sides from two closed openings (French Advance, King's Indian closed centre), 6 plies each from 4 starting points. Stockfish 19 at depth 18 judged each move:
- **13 of 24 moves lost ≥ 2 pawns** compared with the best move.
- Several were one-move oversights a depth-3 search should never make:
  - Ng5 leaving Bd3 to be taken (−734);
  - …Qxd4 ignoring the hanging bishop (−1473);
  - …Bf5 hanging to exf5 (−562);
  - Bg5 missing a free bishop (−782).

The search sees these captures. The evaluation outweighs them: moving a piece nearer the enemy king (Euclidean) can score more than a whole bishop.

## 3. Accretion launders material

French Advance after `1.e4 e6 2.d4 d5 3.e5 c5 4.c3 Nc6 5.Nf3 Qb6`, line `6.Qd3 c4 7.Qxc4 dxc4 8.Bxc4`. White has given a queen for two pawns.

| after | material (v0) | total (v0) |
|---|---|---|
| Qd3 | −0.51 | +6.94 |
| c4 | −0.58 | +1.16 |
| Qxc4 | +3.93 | +11.94 |
| dxc4 | −30.21 | −41.54 |
| **Bxc4** | **+1.94** | **+7.87** |

Each capture moves 80% of the victim's mass to the captor. The pawn that takes the queen becomes ≈ 8.8 units heavy, and the bishop that takes that pawn gets ≈ 7 of it back. v0 counts carried mass as material, so losing heavy pieces in an exchange chain is nearly free. That's an evaluation bug, separate from the field problems.

## 4. Structural problems in the laws

1. **The field ignores the board.** The potential uses straight-line distance, gated by `sigmoid(c − distance)`. It passes through pieces and ignores how each piece moves. A knight behind a pawn wall pulls as hard as one on an open file.
2. **Defenders count for nothing.** King danger is computed from the enemy army's field only. Your own pieces near your king add no protection.
3. **"Speed of light" has two unrelated meanings:** a distance gate in the field, and the velocity accumulator behind the Lorentz mass boost, which in practice is an anti-repetition heuristic.
4. **The "Roche" threshold acts on force magnitude, not on the tidal index.** The tidal tensor is computed, then effectively unused.
5. **Several terms are heuristics with physics names:** centre-of-mass advance, attack moment of inertia, entropy.
6. **Fixed piece masses 1/3/3/5/9** are human chess values, which contradicts "no chess knowledge written in".

## 5. Speed

v0 takes **2.6–3.9 s per move at depth 3** on a Ryzen 5 5500U (CPU JAX). That's too slow for a browser and caps playing strength.

## 6. What didn't reproduce

Quick-mate traps (Scholar's, Fool's patterns) against a simple scripted attacker didn't trap v0 at depth 3. Earlier reports of "mated in 2–3 moves" came from the retired browser lookalike, not from this engine. That evidence is weak. The v1 tactics suite (`04_TRAINING_AND_GATES.md` §3) replaces it.

## 7. Consequences for v1

Each finding maps to a v1 decision in `02_LAWS_SPEC.md`:

| finding | v1 answer |
|---|---|
| field ignores board | L2 rule-shaped space |
| two speeds of light | L3, light speed in moves only |
| defenders ignored | L4a binding energy, L5 binding term |
| decorative tidal tensor | L5 literal Roche index on the king's body |
| accretion laundering | L1 material at rest mass, L6 learned fraction |
| human piece values | L1 learned masses |
| dead terms | removed |
| slow | Rust core (`03_ENGINE_ARCHITECTURE.md`) |
