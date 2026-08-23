# Chapter 5 — The Speed of Light $c$: Retarded Potential as a Reach Gate

> **Read me first (no background needed).** In this universe, influence takes time to travel. Every piece broadcasts its gravity like a radio station, and the signal moves at a finite speed — $c$ squares per ply (a ply being one single move by one side). Set $c$ small and only nearby pieces "hear" each other; set it large and the whole board hears everything instantly. Crucially, $c$ is not fixed by decree: it is one of the learnable knobs, and training settles on whatever delay makes the universe's judgment match real game outcomes. Since August 2026 there is exactly **one** speed of light in the codebase — the same $c$ also governs how a moving piece's mass inflates with speed (Chapter 13.5), so relativity and gravity share one law, not two.

## 5.1 Intuition: gravity is not instantaneous

In classical chess, if you move a queen to attack the enemy King, the threat is *immediate*. In real physics, changes in the gravitational field propagate at the **speed of light** $c$. If $c$ is finite, a queen moved to the far side of the board does not *instantly* stress the enemy King — the "gravity wave" takes time to arrive.

In Kepler-64, $c$ is measured in **squares per ply** (one ply = one move). A small $c$ means gravity only reaches nearby squares; a queen sacrifice on h8 takes several plies to "register" as a threat to a King on a1. This is the **retarded potential**: the force at a square depends on where masses *were* $r/c$ plies ago, not where they are now.

> **Intuition box:** Drop a stone in a pond. The ripple reaches the far edge only after a delay. The King only "feels" your queen once the ripple arrives. $c$ is how fast the ripple crosses the board.

## 5.2 The math: the sigmoid reach gate

Rather than maintaining a full rolling history buffer (the original "Extended Absurdity" idea, `Kepler-64 Extended Absurdity.md` §1), the implemented engine encodes retardation as a **smooth gate** on each pairwise interaction (`core/gravity.py:34`):

$$\text{gate}(d_{ij}) = \sigma\!\bigl(c - d_{ij}\bigr) = \frac{1}{1 + e^{-(c - d_{ij})}}$$

where $d_{ij}$ is the board distance. Properties:

- If $d_{ij} \ll c$: gate ≈ 1 → the mass contributes fully (gravity has "arrived").
- If $d_{ij} \gg c$: gate ≈ 0 → the mass contributes nothing yet (gravity "hasn't arrived").
- The transition is centered at distance $d = c$, and it is **smooth/differentiable** everywhere — essential for learning.

The force and potential then become

$$\vec F_i = G \sum_j |m_j|\,\sigma(c - d_{ij})\,\frac{\vec r_i - \vec r_j}{(d_{ij}^2+\varepsilon^2)^{3/2}}$$

$$U_i = -G \sum_j |m_j|\,\frac{\sigma(c - d_{ij})}{\sqrt{d_{ij}^2+\varepsilon^2}}$$

This single gate is what makes $c$ a **real, differentiable input** to the evaluation. It is the practical realization of the retarded potential without storing history (`core/gravity.py:9-13`).

**Worked example.** Let $c = 4$ squares/ply. A mass at distance 1: gate = $\sigma(4-1)=\sigma(3)\approx 0.95$ → nearly full. A mass at distance 12 (board diagonal): gate = $\sigma(4-12)=\sigma(-8)\approx 0.0003$ → essentially absent. So with $c=4$, only pieces within a few squares meaningfully interact — exactly the "delayed ripple" story.

## 5.3 Why $c$ must be bounded: the monotonicity prior

If $c$ were unconstrained, gradient descent would push it to $\infty$ — *instantaneous* gravity is always the easiest thing to optimize (no delay to model). But that **destroys the retardation story entirely** (the Layer-2 "multiverse" depends on finite $c$). So $c$ carries a **two-sided prior** (`core/constants.py:37-45`):

$$\text{prior}(c) = -\lambda_{\text{fast}}\max(0,\,2 - c)\;-\;\lambda_{\text{slow}}\max(0,\,c - 10)$$

This penalizes $c < 2$ (too slow → engine goes evaluation-blind within a shallow search) and $c > 10$ (too fast → retardation vanishes). The sweet spot is $c \approx 3\!-\!6$ squares/ply, giving 1–3 ply delays for cross-board threats. Note the prior is written with `jnp.maximum`, **not** Python's `max`, so it stays differentiable (see §5.5).

> **⚠ [ISSUE: PYTHON-MAX-PRIOR] (P0/P1, Code Review v1 BUG 3 & v2 BUG 5/6):** An earlier version used Python's built-in `max()` in the $c$-prior. Inside `jax.grad`, a Python `max` either crashes or silently treats the traced $c$ as a concrete value, **breaking gradient flow through the prior** — which defeats the entire point of learning $c$. The fix (`jnp.maximum`) is present in `core/constants.py:45`. The *same* bug existed in `training/loss.py` and was fixed there too (Code Review v2 BUG 6). Confirm both use `jnp.maximum`.

## 5.4 The math link to the retarded Green's function (Layer 2)

The audit's "converged" framing (`Kepler-64 Audit` §A) notes that the retarded potential $\Phi(\vec x, t) = \text{source}(\vec x, t - r/c)$ is precisely the **retarded Green's function** of a massless scalar field — the same object a quantum-field propagator uses. So "evaluate the N-body potential through the retarded Green's function, sampling the constants from a posterior" fuses the multiverse (Chapter 13.2) and the delay into one rigorous equation. The implemented gate $\sigma(c-d)$ is a smooth, differentiable proxy for "has the signal arrived yet."

## 5.5 Project link: $c$ is a learned, prior-bounded leaf

In code, $c$ is initialized at 4.0 (`core/constants.py:17`) and is a **leaf** that gradient descent moves. The training clips it to $[1.0, 10.0]$ (`training/loss.py:51`) — matching the prior's intended bounds. The credibility-gate artifact (`training/trained_constants_gate.json`, 2026-08-23) learned $c = 3.51$, comfortably inside the sweet spot — training moved it, kept it physical, and did not let it run away.

> **One law, enforced.** The same $c$ now drives the Lorentz motion boost ($u = v/(v+c)$) at every search call site — `FastBoard.mass_vector(c_lorentz=...)` receives `Constants.c`; the fallback constant `LORENTZ_C_DEFAULT` exists only for ad-hoc boards constructed outside a game. A regression test pins the invariant `child_mass_vector(...) == child.mass_vector(c)` for arbitrary $c$, so no second hardcoded light speed can silently return.

> **⚠ [ISSUE: c-NOT-TRACED] (P1, Code Review v1 BUG 8):** A prior version declared `c: float` in a `@dataclass` and expected it to be differentiable. A plain Python `float` is **not** a JAX-traced type, so `jax.grad` could not move it. The current architecture avoids this by routing *all* constants through `_score_core(masses, G, eps, c, ...)` as **positional scalar arguments** (`core/evaluate.py:104-110`), which *are* traced. The lesson: never store a learnable constant only as a dataclass float attribute; pass it as a function argument during training. Confirm `train.py` and `loss.py` pass `c` positionally (they do: `training/loss.py:48`).

## 5.6 Forward link

We now have all the physics operators: force, potential, tidal tensor, η, and the $c$-gate. The last foundation chapter explains **automatic differentiation** — the mechanism that makes every constant *learnable* — which is what turns this physics into a trainable evaluation.

**Cross-references:** Force/potential with gate → `core/gravity.py` (§2, §7). Multiverse sampling of $c$ → §13.2. Training through physics → §12.
