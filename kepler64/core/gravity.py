"""N-body gravitational force field — the headline flex.

chess:  every occupied square contributes attraction to every other square.
physics: Plummer-softened Newtonian gravity. Because the board geometry is
         fixed, the (64,64) distance matrix is STATIC and precomputed once; the
         entire force computation is a single einsum over a (64,) mass vector.
         No Python loops, no per-eval distance recomputation.

The speed of light `c` (squares/ply) acts as a GRAVITATIONAL REACH GATE: a mass
at distance d contributes with weight sigmoid(c - d). When c is small, only
nearby masses have arrived (gravity propagates at finite speed); when c is large
the field is effectively instantaneous. This is the retarded-potential concept
and it makes `c` a real, differentiable input to the evaluation.
"""

import jax
import jax.numpy as jnp

# Static square coordinates -> (x, y) in [0,7]^2. Computed ONCE.
_COORDS = jnp.array([(i % 8, i // 8) for i in range(64)], dtype=jnp.float32)
_DIFF = _COORDS[:, None, :] - _COORDS[None, :, :]  # (64, 64, 2)
_DIST2 = jnp.sum(_DIFF**2, axis=-1)  # (64, 64) static
_DIST = jnp.sqrt(_DIST2)


def _soft_r2(eps: float) -> "jnp.ndarray":
    """Plummer-softened squared distance matrix + eps^2 (shared by both fields)."""
    return _DIST2 + eps * eps


def force_field(masses: "jnp.ndarray", eps: float, G: float = 1.0, c: float = 10.0) -> "jnp.ndarray":
    """Return the (64, 2) gravitational acceleration at every square.

    F_i = G * sum_j gate(d_ij) * m_j (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^{3/2}
    gate(d) = sigmoid(c - d): distant masses are delayed when c is small.
    """
    m = jnp.abs(masses)
    r2 = _soft_r2(eps)
    gate = jax.nn.sigmoid(c - _DIST)
    inv = m * gate / (r2 * jnp.sqrt(r2))
    F = jnp.einsum("ij,ijd->id", inv, _DIFF)  # (64, 2)
    return G * F


def potential_field(masses: "jnp.ndarray", eps: float, G: float = 1.0, c: float = 10.0) -> "jnp.ndarray":
    """Scalar Plummer potential U_i = -G * sum_j gate(d_ij) * |m_j| / sqrt(r2 + eps^2)."""
    m = jnp.abs(masses)
    r2 = _soft_r2(eps)
    r = jnp.sqrt(r2)
    gate = jax.nn.sigmoid(c - _DIST)
    return -G * jnp.einsum("ij,j->i", gate / r, m)
