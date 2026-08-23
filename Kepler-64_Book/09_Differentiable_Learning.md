# Chapter 9 — Implicit Differentiation and Learning $G$ Through Physics

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

Learned leaves (`training/train.py:61-64`, 9 scalars): $G, \varepsilon, c, \rho_{\text{roche}}, \text{bonus}, k_{\text{gain}}, \gamma, R_g, m_{\text{gain}}$.

Fixed by design:

- **$m_{\text{ref}} = 3.5$** — a unit scale, deliberately *not* trained (`core/evaluate.py:24-29`, `training/loss.py:57`). Keeps η well-conditioned.
- **$G$ is frequently frozen during training** (`fix_G=True`, `training/train.py:120, 231`). Why? Because in the η formula $G$ cancels (§4.3), so training otherwise collapses $G \to 0$ and kills the force terms. When $G$ is frozen it is set to 1.0. The *other* constants still absorb the scaling.
- **Piece masses** (1,3,3,5,9,1000) are fixed — they are the "vocabulary" of the universe, not learned parameters.

> **⚠ [ISSUE: G-NONIDENTIFIABLE] (P1, training doc & Audit):** $G$ is largely non-identifiable in the tidal index (it cancels in η). The project handles this by freezing $G$ (and noting it in `train.py:9-10`). But the README's flagship line — *"the gravitational constant was learned via gradient descent"* — is only literally true when `fix_G=False`. Be precise: "several constants including $\varepsilon, c, \rho_{\text{roche}}$ are learned; $G$ is typically frozen for identifiability." The current `trained_constants.json` shows $G=1.32$ (learned in that run), so it *can* move; just don't overclaim.

## 9.5 Forward link

Chapter 13 details the loss function and training loop. First, Part C covers the board/search/visualizer (Chapters 10–12) and then Layer 2 (Chapters 14–16), before training in Chapter 13. We'll reorganize: the next chapters cover the board representation and search (10–11), then training (13), visualizer (12), and Layer 2 (14–16). Let me proceed with the board and search now.

**Cross-references:** Constants as leaves → §5. The pipeline being differentiated → §7. Loss & training → §13.
