# Chapter 6 — The Verlet Rollout: Projecting Collapse Without Deep Search

> **Read me first (no background needed).** A photograph tells you where a ball *is*; physics lets you predict where it is *going*. Kepler does both. Besides scoring the position as it stands, it briefly simulates each King as a marble rolling on the curved gravitational surface — just four tiny steps ahead — and checks whether the rolling is *toward* tidal collapse. That "impending doom" reading (`dη/dt`, the rate of change of tidal stress) catches attacks that a static snapshot would miss. The simulation uses a special integrator (Leapfrog) chosen because it conserves energy, so the prediction is honest rather than a numerical illusion.

## 6.1 Intuition: simulate the King's future, don't just snapshot it

A single evaluation tells you the tidal stress *right now*. But a good engine also asks: *"if nothing changes, will the King cross its Roche limit a few moves from now?"* Instead of searching deep (expensive), Kepler-64 **simulates** the King's continuous coordinate forward a few tiny steps under the local gravitational force — a "what-if" projection. If the projected King drifts into a disruption zone, that's a warning even before minimax sees it.

> **Intuition box:** Like rolling a marble on a curved surface to see where it ends up. The King is the marble; the gravitational field is the curved surface. A few small rolls show whether it falls into the well.

## 6.2 The math: Leapfrog (Verlet) integration

The King's position $\vec p$ obeys Newton's second law under the local force $\vec F(\vec p)$:

$$\frac{d^2\vec p}{dt^2} = \vec F(\vec p)$$

We step it forward with a **Leapfrog** (a form of **Verlet**) integrator, which is **symplectic** — it conserves energy approximately, so the projection doesn't artificially gain or lose energy and drift to nonsense. The update (time step $dt$):

$$\vec v \leftarrow \vec v + \vec F(\vec p)\,dt$$
$$\vec p \leftarrow \vec p + \vec v\,dt$$

Starting from the King's square coordinate, repeated for `steps` iterations. **Location (2026-08-23):** the standalone `core/verlet.py` module was removed as dead code; the single live implementation of this integrator is `_eta_drift` inside `core/evaluate.py`. Everything below describes that implementation.

**Why symplectic matters here.** A non-symplectic integrator (e.g. plain forward Euler) leaks energy, so a stable King might *appear* to fly apart, or a doomed one might *appear* safe. Symplectic conservation makes the rollout **honest**: if the projected King crosses the Roche threshold, that's a real tendency of the field, not a numerical artifact.

## 6.3 Efficiency: only the force at the King

A naive rollout recomputes the full $(64,64)$ field every step. The implementation instead computes the force at **only the King's current continuous position** — an $O(64)$ gather-sum, 64× cheaper. As of the Phase-1 audit fix, the per-step force uses the **same light-speed reach gate as every static term** — the rollout simulates the same universe the score lives in:

```python
def _step(carry, _):                     # inside core/evaluate._eta_drift
    p, v = carry
    d = _COORDS - p
    r2 = jnp.sum(d**2, axis=-1) + eps**2
    r = jnp.sqrt(r2)
    gate = jax.nn.sigmoid(c - r)         # <-- same gate as potential_field
    a = G * jnp.einsum("i,ij->j", am * gate / (r2 * r), d)
    v = v + a * dt
    p = p + v * dt
    return (p, v), p
```

> **(Resolved)** [ISSUE: VERLET-FULL-FIELD] and [ISSUE: VERLET-INT-KING] are both closed in the current single implementation; the `lax.scan` body indexes the King with JAX-native indexing (no Python `int()`), so the term stays `vmap`/`jit`-safe.
>
> **(Resolved, 2026-08-23)** [ISSUE: UNGATED-DYNAMICS] *(new)* — earlier the rollout integrated **ungated** forces while every static η term used the sigmoid(c − d) field: dynamics simulated a different universe than the score. Fixed; a property test pins drift → <1e-6 for a distant attacker at tiny $c$.

## 6.4 Differentiability: the rollout is a JAX loop

For the rollout to feed gradients back into $G, \varepsilon, c$, it must be written with JAX-native control flow. A normal Python `for` loop over `steps` is **not** differentiable through traced values. The implementation uses `jax.lax.scan` (`core/evaluate.py`, `_eta_drift`), which is the correct, differentiable construct.

## 6.5 $d\eta/dt$ — the Verlet tidal-drift term (implemented)

The time-derivative of η (Chapter 4.6) lives here. The rollout projects the King's continuous coordinate forward under the **opponent's** field, and the evaluation reads the tidal index at the projected location versus now — the "impending collapse" signal a static snapshot cannot see. It is wired into the live score as the `lambda_drift`-gated term (§7.4), a learnable leaf (14th in `TRAINABLE_LEAVES` order; a 15th leaf, `lambda_gw`, was added later — see Ch.14.6).

The tidal index at the *continuous* projected position is read with an exact analytical Hessian of the gated Plummer potential (`_tidal_tensor_at_pos`, `core/evaluate.py`) — the continuous analogue of `tidal_tensor_at`'s finite differences — so arbitrarily small drifts resolve without snapping to a lattice square. The projection itself is a `jax.lax.scan` Leapfrog (§6.2–6.4), so the whole term is differentiable and participates in training.

> **(Resolved)** The former [ISSUE: DETA-IN-VERLET] — "the missing $d\eta/dt$" — is now implemented as `_eta_drift` in `core/evaluate.py`, source-attributed per King (white's field tears the black King, black's tears the white King), and gated by `lambda_drift`.

## 6.6 Project link and current usage

`rollout()` is now wired into the **live evaluation hot path** via `_eta_drift` (§7.4). For each King, a short Leapfrog projection advances its coordinate under the opponent's field (source-attributed, matching η's tearing-field semantics), and the change in tidal stress over the horizon is the drift signal. It is a genuine, differentiable part of the score — not just a forward-simulation module.

## 6.7 Forward link

Foundations are complete. Part B assembles these into the actual evaluation pipeline: how a chess position becomes a mass vector, how the gravity→tidal→η→score chain runs, and the 218-pad trick that makes it sub-millisecond.

**Cross-references:** Force field → §2. Tidal tensor / η → §3, §4. The score that consumes η → §7. The 218-pad sweep → §8.
