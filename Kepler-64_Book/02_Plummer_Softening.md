# Chapter 2 — Plummer Softening and the Static Distance Matrix

## 2.1 Intuition: the "divide by zero" problem

Newton's law has a trap: when two masses occupy the **same square** (or get arbitrarily close), $r \to 0$ and the force $F = G m_1 m_2 / r^2$ shoots to infinity. On a chessboard, pieces can be adjacent or even (during intermediate computations) coincident. An infinite force is not just physically ugly — it produces `NaN`s that kill gradient descent.

The fix used across N-body astrophysics is **softening**: pretend every mass is slightly fuzzy, smeared over a small radius $\varepsilon$ (the *softening length*). The denominator becomes $r^2 + \varepsilon^2$ instead of $r^2$, so at $r=0$ the force is finite: $F \to G m_1 m_2 / \varepsilon^2$. This is the **Plummer** softening, named after the Plummer model of a star cluster.

> **Intuition box:** Softening is like giving every planet a soft, fluffy edge. Two fluffy planets can't get infinitely close because their edges overlap first — the force levels off instead of exploding.

## 2.2 The math: Plummer-softened gravity

The softened acceleration at square $i$ due to all masses is

$$\vec F_i = G \sum_{j} m_j\,\frac{\vec r_i - \vec r_j}{\bigl(\lVert \vec r_i - \vec r_j\rVert^2 + \varepsilon^2\bigr)^{3/2}}$$

Compare to §1.2: we only added $\varepsilon^2$ inside the cube-root denominator. Everywhere $r \gg \varepsilon$ the formula is identical to Newton; only very close to a mass does it flatten.

In Kepler-64 the masses are taken as **absolute values** so gravity stays attractive regardless of color (sign is used only to identify the side, see §7.3):

$$\vec F_i = G \sum_j |m_j|\,\frac{\vec r_i - \vec r_j}{\bigl(d_{ij}^2 + \varepsilon^2\bigr)^{3/2}},\qquad d_{ij} = \lVert \vec r_i - \vec r_j\rVert$$

**Worked example.** Same queen (9) at a1 and pawn (1) at e1 as before, but now $\varepsilon = 0.5$. Distance $d=4$:

$$F = G\cdot 1 \cdot \frac{4}{(4^2 + 0.5^2)^{3/2}} = G\cdot\frac{4}{(16.25)^{1.5}} = G\cdot\frac{4}{65.5} = G\cdot 0.0611$$

Barely different from the unsoftened $0.0625$ — softening only matters at short range, exactly as intended.

## 2.3 The math: the static $(64\times 64)$ distance matrix — the key speed insight

Here is the elegant part that makes the whole engine fast. **The chessboard never moves.** Squares a1…h8 are always at the same coordinates. So the pairwise distances $d_{ij} = \lVert \vec r_i - \vec r_j\rVert$ depend only on the board geometry, never on the pieces. We compute them **once** and reuse them forever.

Define three static tensors (`core/gravity.py:20-23`):

- `_COORDS` — the $(64,2)$ coordinates of every square.
- `_DIFF[i,j] = _COORDS[i] - _COORDS[j]` — a $(64,64,2)$ tensor.
- `_DIST2[i,j] = \lVert\_DIFF[i,j]\rVert^2$ — the static $(64,64)$ distance-squared matrix.
- `_DIST[i,j]` — the raw distance.

Because `_DIST2` is static, the only thing that changes from position to position is the **mass vector** $\mathbf m \in \mathbb{R}^{64}$. The entire force field is therefore one Einstein summation (`einsum`) over the masses:

```python
r2   = _DIST2 + eps * eps            # (64,64) static + scalar
gate = jax.nn.sigmoid(c - _DIST)     # (64,64) retarded-potential gate, see §5
inv  = m * gate / (r2 * jnp.sqrt(r2))# (64,64)
F    = jnp.einsum("ij,ijd->id", inv, _DIFF)   # (64,2)
return G * F
```

The `einsum` `"ij,ijd->id"` says: for each target square $i$ and each source square $j$, multiply `inv[i,j]` by the vector `_DIFF[i,j]`, and sum over $j$ to get the $(2,)$ force at $i$. **No Python loops.** This is the "headline flex" of the project.

> **Intuition box:** Imagine gluing 64 magnets to fixed spots on a board. Only their strengths change each turn; their positions never do. So you precompute all the "how-far-apart" numbers once and reuse them for every move.

## 2.4 The scalar potential field $U$

Besides the force (a vector), we also need the **gravitational potential** $U$ — a single number per square telling you how "deep" the well is. For a single mass, $U = -G m / r$. With softening and the gate:

$$U_i = -G \sum_j |m_j|\,\frac{\sigma(c - d_{ij})}{\sqrt{d_{ij}^2 + \varepsilon^2}}$$

where $\sigma$ is the sigmoid gate (Chapter 5). This is what the visualizer heat-maps (`viz/glassbox.py`, Chapter 15) and what the tidal tensor differentiates (Chapter 3). Code: `core/gravity.py:40-45`.

## 2.5 Project link and the softening constant $\varepsilon$

$\varepsilon$ is another learned leaf (`core/constants.py:16`, init 0.5). It trades off between "sharp, realistic close-range forces" and "numerically safe, smooth gradients." The training clamps it to $[0.01, 20.0]$ (`training/loss.py:50`).

> **Project link:** The force field `force_field()` and potential `potential_field()` in `core/gravity.py` are the only places gravity is computed. Every downstream concept (tidal tensor, η, tactical penalty, field-energy edge, Verlet rollout) calls one of these.

> **⚠ [ISSUE: SIGN-CONFUSION] (P1, from Code Review v2 BUG 20 / v1 BUG 20):** An earlier design intended `mass_vector()` to return *signed* masses (white positive, black negative) so the tactical direction is recoverable. But gravity must use $|m|$. The current `force_field` correctly does `m = jnp.abs(masses)`. The danger is any future refactor that drops the `abs` and lets opposite-colored masses *repel* (which would be electromagnetism, not gravity). Keep `jnp.abs` at the entry of every gravity routine.

## 2.6 Forward link

We have a smooth, fast, differentiable force and potential field. The next chapter shows how to extract the **tidal tensor** — the Hessian of that potential — at the King, and why its largest eigenvalue is the actual "win condition."

**Cross-references:** Force field code → `core/gravity.py` (Ch 7). Tidal tensor → §3. The $\sigma(c-d)$ gate (retarded potential) → §5.
