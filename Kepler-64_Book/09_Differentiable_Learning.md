# Chapter 9 — Implicit Differentiation and Learning $G$ Through Physics

> **Read me first (no background needed).** This chapter explains the closest thing to magic in the project: **the universe tunes its own laws.** The engine's constants — gravity's strength, the fluff size, the speed of light, the collapse threshold — start as guesses. Then the engine plays/studies thousands of real games, and after each prediction it asks: "how wrong was I, and whose fault is that?" Calculus's chain rule, applied automatically by software (autodiff), apportions the blame backwards through the entire physics — from a wrong game outcome all the way down to "gravity was 0.3% too strong here." Every constant then takes one tiny step in the helpful direction. Repeat a few thousand times and the laws settle into whatever shape makes the physics agree with reality.

## 9.1 Intuition: the engine teaches itself the laws of its universe

Normally, an engine's evaluation has hand-set weights (e.g. "a knight on the rim is worth −0.3"). Kepler-64 instead has *physical constants* ($G, \varepsilon, c, \rho_{\text{roche}}, \dots$) and lets **gradient descent** set them by looking at real game outcomes. The engine doesn't read a chess book; it reads thousands of games and asks: *"which values of $G$ make my physics score agree with who actually won?"*

This is only possible because **the entire pipeline is differentiable** — every operation from Plummer gravity through the tidal tensor to η is a smooth function whose derivative with respect to the constants exists and is computed automatically by JAX.

> **Intuition box:** Imagine tuning a guitar by ear. Each twist of a peg (a constant) changes the sound (the score); you listen to the error (wrong prediction) and nudge the peg the right way. JAX computes *which way and how much* to nudge, automatically, for all pegs at once.

## 9.2 The math: gradient descent in one picture

We have a loss $\mathcal L(\theta)$ that measures how wrong the engine's predictions are, where $\theta = (G, \varepsilon, c, \rho_{\text{roche}}, \text{bonus}, k_{\text{gain}}, \gamma, R_g, m_{\text{gain}})$. Gradient descent updates:

$$\theta \leftarrow \theta - \eta_{\text{lr}}\,\nabla_\theta \mathcal L$$

$\nabla_\theta \mathcal L$ is the vector of "how much does the loss change if I wiggle each constant." **Automatic differentiation (autodiff)** computes this exactly by applying the chain rule through every operation in the pipeline — no manual derivative formulas needed.

The chain rule through the pipeline looks like:

$$\frac{\partial \mathcal L}{\partial G} = \frac{\partial \mathcal L}{\partial \text{score}}\cdot\frac{\partial \text{score}}{\partial \eta}\cdot\frac{\partial \eta}{\partial \lambda_1}\cdot\frac{\partial \lambda_1}{\partial U}\cdot\frac{\partial U}{\partial F}\cdot\frac{\partial F}{\partial G}$$

Every factor is a derivative JAX knows how to compute because each step (einsum, sigmoid, sqrt, eigenvalue) is a primitive with a known Jacobian.

## 9.3 Why "differentiable" is the whole point

- A **boolean** `is_checkmate` has derivative 0 → cannot learn from mates (§7.5).
- A **Python `max`** in the $c$-prior breaks tracing → cannot learn $c$ (§5.3).
- A **plain `float`** constant stored only as a dataclass field is not traced → cannot learn it (§5.5).
- A **Python `for` loop** in the Verlet rollout is not differentiable through traced values → must use `jax.lax.scan` (§6.4).

The codebase is written to avoid all four traps. That is what makes the project's central claim — *"the gravitational constant was learned via gradient descent"* — technically true rather than marketing.

## 9.4 What is learned vs. what is fixed

Learned leaves — **15 of them**, packed in `TRAINABLE_LEAVES` order (`core/constants.py`, single source of truth shared by the loss, trainer, and JSON persistence):

$G$, $\varepsilon$, $c$, $\rho_{\text{roche}}$, bonus, $k_{\text{gain}}$, $\gamma$, $R_g$, $m_{\text{gain}}$, $\lambda_{\text{delta}}$, $c_{\text{om\_gain}}$, $i_{\text{nertia\_gain}}$, $e_{\text{ntropy\_gain}}$, $\lambda_{\text{drift}}$, $\lambda_{\text{gw}}$.

That is: the nine original physics/scale knobs, the four move-sensitivity (delta) gains, the Verlet-drift gain, and the gravitational-wave gain (init 0.0 — see Ch.7 Step 9).

Fixed by design:

- **$m_{\text{ref}} = 3.5$** — a unit scale, deliberately *not* trained. Keeps η well-conditioned.
- **$G$ is frequently frozen during training** (`fix_G=True`). Why? Because in the η formula $G$ cancels (§4.3), so training otherwise collapses $G \to 0$ and kills the force terms. When $G$ is frozen it is set to 1.0. The *other* constants still absorb the scaling.
- **Piece masses** (1,3,3,5,9,1000) are fixed — they are the "vocabulary" of the universe, not learned parameters (Ch.7 honest-framing note).
- **The velocity decay rate** (`VELOCITY_DECAY = 0.985` per ply) is currently a documented fixed hyperparameter rather than a leaf — promoting it would renumber the leaf vector, so it is deferred until the next training-phase change.

> **⚠ [ISSUE: G-NONIDENTIFIABLE] (P1, training doc & Audit):** $G$ is largely non-identifiable in the tidal index (it cancels in η). The project handles this by freezing $G$. But the README's flagship line — *"the gravitational constant was learned via gradient descent"* — is only literally true when `fix_G=False`. Be precise: "several constants including $\varepsilon, c, \rho_{\text{roche}}$ are learned; $G$ is typically frozen for identifiability." The credibility-gate artifact shows which leaves actually moved and by how much.

## 9.5 Forward link

With differentiable physics established, Part C turns to engineering: the board representation and search tree (Chapter 10), the Glass Box visualizer (Chapter 11), then training end-to-end (Chapter 12) and Layer 2 (Chapter 13).

**Cross-references:** Constants as leaves → §5. The pipeline being differentiated → §7. Loss & training → §12.
