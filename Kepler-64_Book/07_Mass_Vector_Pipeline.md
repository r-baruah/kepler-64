# Chapter 7 — From Chess Position to Gravitational State: The Mass Vector & Core Pipeline

> **Part B — The Core Engine.** All physics concepts from Part A are now assembled. This chapter and the next show exactly how Kepler-64 turns a board into a score. No new concept is introduced here — only the wiring.

---

## 7.1 Intuition: a chess position is just 64 numbers

To run the physics, the engine needs one thing: **for each of the 64 squares, how much mass is there, and whose is it?** That is the **mass vector** $\mathbf m \in \mathbb{R}^{64}$. A pawn on e4 is `m[e4] = +1` (white, positive). A black queen on d8 is `m[d8] = -9` (black, negative). An empty square is `0`.

The sign (white `+`, black `-`) is only a **color tag** so the engine can tell "your masses" from "enemy masses" for source-attribution (§7.4). The gravity itself uses $|m|$ so it stays attractive regardless of sign (§2.5).

> **Intuition box:** The board becomes a 64-element shopping list of weights. The physics kernel reads the list and computes the field. Sign is just a name-tag for "which team."

## 7.2 The math: mass assignments

The piece→mass map is deliberately the standard chess **material values**, repurposed as gravitational mass (`core/constants.py:50-57`):

| Piece | Mass | Note |
|---|---|---|
| Pawn | 1 | light |
| Knight | 3 | |
| Bishop | 3 | |
| Rook | 5 | |
| Queen | 9 | heavy |
| **King** | **1000** | intentionally enormous so its self-gravity dominates local pawns |

Rationale for the huge King mass (`core/constants.py:49`): the King's own gravity must overwhelm nearby pawns so it behaves like a local "star" and so disruption is governed by *external* tearing, not its own trivial self-field.

> **Honest framing (say it like this, everywhere).** Yes — the mass ladder 1/3/3/5/9 is inherited from chess, not derived from astrophysics. The right way to present it: **mass is the universe's baryon budget**, and chess material is simply the unit system we chose to measure it in, exactly as astronomers measure mass in solar masses. What makes this *not* plain material counting is what the physics does with the budget: the same number feeds gravitational pull, tidal stress, binding energy, momentum flux and entropy — and its *scale* (`mat_gain`) plus every coupling constant are learned, not decreed. A reviewer should hear: "chess values are the units; gravity is the law; learning sets the coupling," never "we reinvented material counting."

## 7.3 The mass_vector() implementation (two boards)

There are two board representations (Chapter 12):

1. **`Board` (python-chess adapter), `core/board.py:57-79`.** Builds a lookup table `_MASS = [0,1,3,3,5,9,1000]`, indexes by piece type, then multiplies by sign. The sign comes from `python-chess` piece colors. Uses one JAX gather + one multiply (vectorized, no per-square Python loop in the hot path).
2. **`FastBoard` (pure-NumPy), `core/fastboard.py:337-343`.** Encodes pieces as signed `int8` (white `+`, black `-`), so mass is a single `np.sign(pieces) * LUT[|pieces|]` — the fastest path, no python-chess at all.

Both produce the same $(64,)$ signed vector. `FastBoard.mass_vector()` is what the search hot path uses.

> **⚠ [ISSUE: PIECE-KEYS] (P0, Code Review v1 BUG 2):** An earlier `PIECE_MASSES` used single-letter keys `"P","N",…` while `board.py` queried full names `"PAWN","KNIGHT",…` → `KeyError` on every mass lookup. Fixed by switching to a numeric LUT indexed by python-chess piece type. Confirm no code still does string-keyed mass lookups.

> **⚠ [ISSUE: SIGNED-MASS-CLAIM] (P1, Code Review v1 BUG 6):** An earlier version's docstring promised *signed* masses but the code returned all-positive. Current `mass_vector()` does return signed masses (white +, black −). The sign is consumed only for king-finding and source-attribution, never inside `force_field` (which uses `|m|`). Correct.

## 7.4 The evaluation pipeline — assembled

`evaluate.py` is the heart. From a mass vector it computes (all from **White's** perspective; the side-to-move sign is applied at the end, `core/evaluate.py:130-134`):

**Step 1 — split by color.** `white_m = |m| where m>0`, `black_m = |m| where m<0` (`core/evaluate.py:65-67`).

**Step 2 — source-attributed fields.** Compute *your* force/potential and *enemy's* force/potential separately:

```python
F_w = force_field(white_m, eps, G, c)   # your masses' pull everywhere
F_b = force_field(black_m, eps, G, c)   # enemy masses' pull everywhere
U_w = potential_field(white_m, eps, G, c)
U_b = potential_field(black_m, eps, G, c)
```

**Step 3 — find the Kings** via `_king_idx`, which locates the square whose `|m| ≈ 1000` and whose sign matches the side (`core/evaluate.py:35-38`). The match uses `jnp.isclose(..., atol=5.0)` so accretion-shifted Kings (e.g. a King at `1008` after eating a pawn) are still found, and `_king_found` guards the disruption/force terms so a missing King neutralizes them instead of silently scoring at square a1.

> **(Resolved)** The former [ISSUE: KING-FALLBACK] and [ISSUE: KING-MASS-FRAGILE] are both closed: `_king_idx` now matches with `atol=5.0`, and the eval zeroes the disruption/force terms via `_king_found` when either King is missing (no silent a1 fallback).

**Step 4 — disruption from the opponent only.** Your King is stressed by *enemy* masses; enemy King by *your* masses:

```python
eta_w = _eta(U_b, wk)   # enemy masses tearing YOUR king  -> bad for White
eta_b = _eta(U_w, bk)   # your masses tearing ENEMY king -> good for White
```

(See §3, §4 for `_eta`.) This **source-attribution** fixes an earlier perverse incentive where capturing an enemy piece *lowered* total force on the enemy King and looked bad (`core/evaluate.py:8-12`).

**Step 5 — force-sigmoid tactical penalty (soft, differentiable).** Instead of a boolean `is_checkmate()` (which has zero gradient, §7.5), the tactical term is a smooth sigmoid on the force magnitude at each King, flipped around the learned Roche threshold:

$$\text{bonus} = +M\,\sigma\!\bigl(k(\lVert \vec F_{\text{white on enemy K}}\rVert - \rho_{\text{roche}})\bigr)$$
$$\text{pen}   = -M\,\sigma\!\bigl(k(\lVert \vec F_{\text{black on your K}}\rVert - \rho_{\text{roche}})\bigr)$$

Code (`core/evaluate.py:83-84`): `bonus_b` and `pen_w`. This is the fix for the inverted-sign bug (see §7.6).

**Step 6 — global field-energy edge.** More disruptive *piece* reach (excluding Kings, whose 1000-mass would dwarf everything) gives a small edge (`core/evaluate.py:86-94`):

$$\text{edge} = \gamma\,(E_{\text{white}} - E_{\text{black}}),\quad E = \sum_i \lVert \vec F_{\text{pieces},i}\rVert^2$$

**Step 7 — material edge.** A clean `mat_gain * (Σ white_m − Σ black_m)` term (`core/evaluate.py:99`) rewards winning material and gives the search a varied gradient so it doesn't drift to a flat equilibrium. `mat_gain` is a learned leaf (init 2.0, `core/constants.py`): a captured rook moves the score ~10, a queen ~18 — decisively above the ~1–4 positional noise floor, which is the fix for the measured "give away free rooks/pieces on the a/h files" behaviour.

**Step 8 — Verlet tidal drift (impending collapse).** A short Leapfrog projection advances each King's continuous coordinate under the OPPONENT's field, and the change in tidal stress over the horizon is the drift signal, gated by the `lambda_drift` leaf (the 14th trainable constant, init 1.0):

$$\text{drift} = \lambda_{\text{drift}}\big(\Delta\eta_{\text{on black K}} - \Delta\eta_{\text{on white K}}\big)$$

Source-attributed like η (§7.4 Step 4): white's field tearing the black King is good for White when *growing*; black's tearing the white King is bad when growing. This is the "threat a few plies out" signal (§6.5), read at the continuous projected position with an exact analytical Hessian.

**Step 9 — Gravitational-wave energy-loss edge (Peters–Mathews).** Accelerating binary masses radiate gravitational waves at $dE/dt \propto G^4 m_1^2 m_2^2 / r^5$ — tight pairs of heavy bodies bleed energy fastest. Kepler-64 prices this in per army (`_gw_radiation`, `core/evaluate.py`): kings excluded (detector discipline, §7.4), diagonal masked (a mass does not radiate against itself),

$$\text{gw} = \lambda_{\text{gw}}\Big(\textstyle\sum_{i \neq j}^{\text{black}} \tfrac{m_i^2 m_j^2}{(d_{ij}+\varepsilon)^5} - \sum_{i \neq j}^{\text{white}} \tfrac{m_i^2 m_j^2}{(d_{ij}+\varepsilon)^5}\Big)$$

$\lambda_{\text{gw}}$ is the **15th learnable leaf**, initialized at exactly $0.0$ with bounds $[0, 10]$: the universe ships *without* wave losses and is bit-identical to the pre-GW physics until training decides otherwise (a regression test pins this neutrality; a second pins that a huddled army radiates strictly more than a spread one).

**Final score (White perspective):**

$$\text{score} = \eta_b - \eta_w + \text{bonus}_b + \text{pen}_w + \text{global\_edge} + \text{material} + \text{drift} + \text{gw}$$

(`EvalTerms` in `core/evaluate.py` carries all twelve weighted terms plus `total`, so the Glass Box and replay can show every contribution separately.) Positive = good for White; `evaluate()` flips it for Black to move.

## 7.5 Why no boolean checkmate in the differentiable path

A boolean `is_checkmate()` returns `−∞` and has **zero gradient** — you cannot backprop through a branch. If the eval were `Physics + (−∞ if checkmate)`, gradient descent would never learn $G, \varepsilon, c$ from mating positions (Audit §2, P II §3). The soft sigmoid penalty is the differentiable stand-in. Note: the **search** layer *does* use a hard `-10000` for actual checkmate (`core/search/minimax.py:28-30`) — that's fine because search isn't differentiated, only training is.

> **⚠ [ISSUE: TACTICAL-SIGN-INVERTED] (P0, Code Review v2 BUG 1):** A prior `_score_core` was *always* White-perspective but applied `pen_b = -50*sigmoid(...)` to **Black's** king under force — meaning White was *penalized* for threatening Black. The sign was backwards; it could tank Elo by hundreds. The current code uses `bonus_b` (good for White) and `pen_w` (bad for White) correctly (`core/evaluate.py:83-84`). Verify the live eval uses both terms with correct signs.

## 7.6 Project link: two entry points

- `_score_core` — **traced** constants, for training (`core/evaluate.py:104-110`). `jax.grad` flows into all leaves.
- `_score_core_static` — **static** constants (compile-time), for fast inference/search (`core/evaluate.py:113-119`). XLA folds $G,\varepsilon,c$ into the kernel.

Both share `_score_body` (`core/evaluate.py:61`). This split is the clean way to keep $c$ learnable (§5.5) while staying fast at play time.

## 7.7 Forward link

The next chapter explains the **218-pad vmap trick** — how all candidate moves are scored in one XLA-compiled kernel, enabling the sub-millisecond sweep — and the quiescence/alpha-beta search that uses it.

**Cross-references:** Force/potential → §2. Tidal/η → §3, §4. Training uses `_score_core` → §13. Search uses `_score_core_static` via `batch_score` → §8.
