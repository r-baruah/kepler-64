# Chapter 12 — Training Through Physics: Loss, Data, and the Ablation Gate

## 12.1 Intuition: learning from real games, not from heuristics

To make the physics *play good chess*, the engine is trained on **real game data** — Lichess puzzles and the author's own games — using two supervisory signals:

1. **Outcome:** did White win / draw / lose? (a logistic classification target)
2. **Policy:** on a puzzle, did the engine's physics score rank the *expert's* move highest among all legal moves? (behavioral cloning through the gravity kernel)

Gradient descent then adjusts the constants so the physics predictions match reality. No chess rule is ever fed in by hand.

> **Intuition box:** Show the engine 5,000 photos of cats and dogs; it learns the boundary. Show it 5,000 positions and who won; it learns the gravitational "boundary" between winning and losing positions.

## 12.2 The math: the loss function

`training/loss.py` defines a scalar, JIT-compatible loss (`training/loss.py:31-99`):

$$\mathcal L = \underbrace{\mathcal L_{\text{outcome}}}_{\text{logistic}} \;+\; 0.5\,\underbrace{\mathcal L_{\text{policy}}}_{\text{expert-move CE}} \;+\; \underbrace{\mathcal L_{\text{prior}}}_{\text{soft } c\text{-bound}}$$

**Outcome term (logistic):** $y = (Y+1)/2 \in \{0,1\}$ (0 = Black win … 1 = White win). The White-perspective score $S$ maps via sigmoid:

$$\mathcal L_{\text{outcome}} = -\frac{1}{N}\sum_i \Big[y_i \log\sigma(S_i) + (1-y_i)\log\sigma(-S_i)\Big]$$

This is binary cross-entropy on "does the score predict the winner?" (`training/loss.py:62-64`).

**Policy term:** for each position, score all $K$ child moves from the side-to-move perspective (flip for Black), mask dummies to `−∞`, and take the log-softmax onto the expert move's index:

$$\mathcal L_{\text{policy}} = -\frac{1}{N}\sum_i \log \frac{e^{s_{\text{expert},i}/\tau}}{\sum_k e^{s_{k,i}/\tau}}$$

The temperature $\tau$ (default 2.0) flattens the softmax so the large material term doesn't zero out the physics gradients (`training/loss.py:42-43, 73`). An optional **pairwise margin** term also pushes the expert move above the best alternative by at least `margin` (`training/loss.py:81-95`).

**Prior term:** the soft monotonicity prior on $c$ (`training/loss.py:98`, see §5.3):

$$\mathcal L_{\text{prior}} = 0.1\max(0,2-c) + 0.1\max(0,c-10)$$

Note it uses `jnp.maximum` (differentiable) — the Code Review v2 BUG 6 fix.

## 12.3 The training loop

`training/train.py:86-182` runs mini-batch **Adam** (or SGD fallback) with:

- **gradient clipping** `max_norm = 1.0` (prevents a noisy batch from launching `roche`/`eps` to extremes),
- **per-step projection** of all **14** leaves into physical bounds (`_LO`/`_HI`, `training/train.py:82-83`) — the 9 original physics constants, the 4 delta-term gains, and the Verlet drift gain `lambda_drift` (the 14th leaf, init 1.0),
- **`fix_G`** option freezing $G=1.0$ (§9.4),
- batch size 128 (keeps the working set < ~1 GB on the author's 8 GB laptop),
- CPU-only JAX tuned to use all 12 logical threads (`training/train.py:31-38`).

Validation is split 80/20 and reports **top-1** and **MRR** (Mean Reciprocal Rank) of the expert move (`training/train.py:216-226`, `policy_metrics` in `loss.py:110-141`) — so you can see real progress, not just training-set overfitting.

## 12.4 The ablation gate — the integrity guardrail

This is the most important honesty check in the whole project (`Kepler-64 Audit` Honest/Dishonest table, `Kepler-64 Scaffold.md` "credibility gate"). The claim *"gravity does real work"* is only credible if:

> Train two engines — one with **learned $G$ (and other constants)**, one with **$G=1$ fixed (no learning)** — and show the **Elo delta**. If the delta is < 50 Elo, the gravity isn't doing meaningful work and the thesis collapses.

`training/ablation.py` is the script for this. The README/abstract should publish the table: `G=1 fixed` vs `G learned`. **Without it, the central thesis is unverified.**

> **⚠ [ISSUE: ABLATION-MISSING-FROM-ARTIFACTS] (P0 for credibility, Audit):** The ablation table is described as a "first-class README artifact" but is not yet present in the README. Shipping the engine without it leaves the headline claim ("gravity learned via gradient descent improves play") unsubstantiated. **This is the single most important deliverable for a skeptic.** Produce it and put it in the README.

## 12.5 Current trained constants (reality check)

`training/trained_constants.json` after a run:

```
G=1.32, eps=0.71, c=4.82, roche=0.05,
bonus=49.0, kgain=4.56, gamma=0.0, Rg=0.19, mref=3.5, mat_gain=0.3
```

Observations:

- `roche` hit its floor `0.05` (see §4.4 warning — a negative/extreme roche silently kills terms).
- `gamma = 0.0` — the global field-energy edge term was driven to **zero** by training, a sign it was noisy relative to material/disruption on the small dataset (verification doc §Part 2 root cause 3).
- `Rg = 0.19` — far below the init 1.0; with $R_g^3$ in η this heavily scales disruption. Confirm this is intended, not a collapse.
- `mat_gain = 0.3` — material dominates; the engine is "count material + small physics," which undermines the thesis unless physics terms are competitive (verification doc Suggestion 7).

> **Note (rebalanced defaults, 2026-08).** The values above are from an early training run. The **default** constants have since been rebalanced twice. First (2026-08-15) the physics levers were lifted to make the tidal/disruption signal first-class. Then (2026-08-17), after the "sacrifice-bug" diagnosis — a positional advance scored +240 while capturing a free rook registered +5, so the depth-3 search gave away rooks/pawns on the a/h files — the gains were re-bounded so **material stays decisive** and the physics levers act as refinement, not override: `bonus` 220→300, `lambda_delta` 1→2, `entropy_gain` 2.5→4, `com_gain` 25→1, `inertia_gain` 8→1, `gamma` 0.25→0, and `mat_gain` 1→2 (a captured rook now moves the score ~10, a queen ~18 — decisively above the ~1–4 positional noise floor). The cohesion (`gamma`) term was zeroed because the *absolute* cohesion gap structurally punished development: starting from the packed home rank, every developing move unbinds the army (Nf3 costs ~2.4 units vs h3 ~0.9), so the search preferred flank shuffles; cohesion-as-advantage is already covered by the move-sensitive inertia term. `gamma` stays a trainable leaf so training can learn its sign from data. A **14th leaf `lambda_drift`** (init 1.0) gates the new Verlet tidal-drift term (§6.5, §7.4). The multiverse posterior (§13.2) now perturbs all 14 leaves. The leaf packing/persistence is centralized in `core/constants.py` (`TRAINABLE_LEAVES`, `leaves_to_array`/`array_to_leaves`, `save_constants`/`load_constants`) so the optimizer, loss, and JSON artifact share one source of truth.

> **⚠ [ISSUE: SMALL-DATA-COLLAPSE] (P1, verification doc Part 2):** A 200-puzzle + 178-game run (378 examples) overfit and drove `roche` and `gamma` to harmful extremes, *dropping* puzzle accuracy from 0.075 to 0.055. Fixes: scale to 5k+ puzzles / 10k+ positions, use Adam + cosine LR, add validation split, two-phase training (outcome then policy). See `verification_and_training_suggestions.md` for the full priority list.

## 12.6 Forward link

Layer 2 ("the Multiverse") extends training into a *distribution* over constants and a self-observing loop. Chapter 13.

**Cross-references:** Differentiable learning → §9. Constants as leaves → §5. The score being trained → §7.
