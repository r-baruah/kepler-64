"""Conditional fluid layer — ONLY if grounded in path-integral variance.

chess:  laminar regions = stable plans; turbulent regions = sharp tactics.
physics: U_uncertain(x,y) = Var_k[ Eval_k(x,y) ] from the Multiverse samples.
        Stokes flow v = -grad(U_uncertain)/mu over the 8x8 lattice. If
        U_uncertain is NOT the path-integral variance, this module is dead
        weight — so the public API requires `eval_samples`.
"""

import jax.numpy as jnp


def uncertainty_field(eval_samples):
    """eval_samples: (K, 64) evals across Multiverse samples -> (64,) variance."""
    return jnp.var(jnp.asarray(eval_samples), axis=0)


def stokes_flow(uncertainty, mu: float = 1.0):
    """v = -grad(U_uncertain)/mu over the 8x8 lattice (central differences)."""
    U = jnp.asarray(uncertainty).reshape(8, 8)
    gy, gx = jnp.gradient(U)
    vx = -gx / mu
    vy = -gy / mu
    return jnp.stack([vx, vy], axis=-1).reshape(64, 2)
