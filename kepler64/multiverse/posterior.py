"""2A "the Multiverse" — Bayesian model average over possible physics.

chess:  instead of asking "which move is best under one set of rules?", the
        engine asks "which move is best across possible universes?"
physics: sample K realizations of (G, eps, c) from the posterior; the
        evaluation is their average with EQUAL posterior mass (true Bayesian
        model average). The retarded Green's function G_ret(θ_i) is the
        mathematical core: the engine does not know which universe's c governs
        the delay reaching the enemy King.

NOTE: this is *not* worst-case adversarial hedging. If you want robust
optimization under parameter uncertainty, replace the uniform mean with a
different risk-weighting — but then call it that, not "Bayesian average".
"""

import jax
import jax.numpy as jnp

from ..core.evaluate import score_white
from ..core.constants import Constants


def _perturb(base: Constants, key, sigma: float = 0.1) -> Constants:
    k1, k2, k3 = jax.random.split(key, 3)
    return Constants(
        G=base.G * (1.0 + sigma * jax.random.normal(k1)),
        eps=base.eps * (1.0 + sigma * jax.random.normal(k2)),
        c=jnp.clip(base.c * (1.0 + sigma * jax.random.normal(k3)), 1.0, 10.0),
        roche=base.roche,
    )


def multiverse_score_white(masses, base: Constants, key, K: int = 8, sigma: float = 0.1):
    """Eval = (1/K) * sum_i Eval(theta_i); equal posterior mass over universes."""
    keys = jax.random.split(key, K)
    consts = [_perturb(base, k, sigma) for k in keys]
    scores = jnp.stack([score_white(masses, c) for c in consts])
    return float(jnp.mean(scores))
