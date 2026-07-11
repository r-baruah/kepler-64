"""Training loop — gradient descent through the physics engine.

physics: optimize G, eps, c, roche with plain SGD so the gravitational constant
        is genuinely learned, not hand-picked. `c` is a traced jnp array, so
        jax.grad flows into it (and the c-prior in loss.py) — fixing the
        "c is not differentiable" bug.
chess:  the result is an engine whose evaluation mimics structurally sound
        positions without ever being told a chess rule.
"""

import jax
import jax.numpy as jnp

from ..core.constants import Constants
from .loss import loss as loss_fn


def _to_arr(c: Constants) -> "jnp.ndarray":
    return jnp.array([c.G, c.eps, c.c, c.roche], dtype=jnp.float32)


def _from_arr(a) -> Constants:
    return Constants(
        G=float(a[0]),
        eps=float(a[1]),
        c=float(jnp.clip(a[2], 1.0, 10.0)),
        roche=float(a[3]),
    )


def train(base: Constants, samples, steps: int = 200, lr: float = 0.05, fix_G: bool = False):
    """samples: list of (mass_vector, outcome) where outcome in {-1,0,+1}."""
    M = jnp.stack([jnp.asarray(m, dtype=jnp.float32) for m, _ in samples])
    Y = jnp.array([oc for _, oc in samples], dtype=jnp.float32)
    arr = _to_arr(base)
    if fix_G:
        arr = arr.at[0].set(1.0)

    grad_fn = jax.grad(lambda a: loss_fn(a, M, Y))
    for _ in range(steps):
        g = grad_fn(arr)
        if fix_G:
            g = g.at[0].set(0.0)
        arr = arr - lr * g
        arr = arr.at[2].set(jnp.clip(arr[2], 1.0, 10.0))
    return _from_arr(arr)
