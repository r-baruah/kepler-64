"""Training loop — gradient descent through the physics engine.

physics: optimize G, eps, c, roche, bonus, kgain, gamma with plain SGD so the
         gravitational "weights" are genuinely learned from real play, not
         hand-picked. `c` is a traced jnp array so jax.grad flows into it (and
         the c-prior in loss.py) — fixing the "c is not differentiable" bug.
chess:  the result is an engine whose evaluation mimics structurally sound
         positions / expert moves without ever being told a chess rule.

Optional policy data (position, expert-move) from a strong engine makes the
eval rank expert moves highest — the legitimate route to real strength.
"""

import jax
import jax.numpy as jnp

from ..core.constants import Constants
from .loss import loss as loss_fn


def _to_arr(c: Constants) -> "jnp.ndarray":
    return jnp.array([c.G, c.eps, c.c, c.roche, c.bonus, c.kgain, c.gamma],
                     dtype=jnp.float32)


def _from_arr(a) -> Constants:
    return Constants(
        G=float(a[0]),
        eps=float(a[1]),
        c=float(jnp.clip(a[2], 1.0, 10.0)),
        roche=float(a[3]),
        bonus=float(jnp.clip(a[4], 0.01, 500.0)),
        kgain=float(jnp.clip(a[5], 0.01, 50.0)),
        gamma=float(jnp.clip(a[6], 0.0, 50.0)),
    )


def train(base: Constants, samples, steps: int = 200, lr: float = 0.05,
          fix_G: bool = False, moves_m=None, expert_idx=None, mask=None):
    """samples: list of (mass_vector, outcome) with outcome in {-1,0,+1}.

    Policy: pass moves_m (N,K,64) child mass vectors, expert_idx (N,) and
    mask (N,K) to add the expert-move ranking term.
    """
    M = jnp.stack([jnp.asarray(m, dtype=jnp.float32) for m, _ in samples])
    Y = jnp.array([oc for _, oc in samples], dtype=jnp.float32)
    n = len(samples)

    if moves_m is not None and expert_idx is not None:
        moves_m = jnp.asarray(moves_m, dtype=jnp.float32)
        expert_idx = jnp.asarray(expert_idx, dtype=jnp.int32)
        if mask is None:
            mask = jnp.ones((moves_m.shape[0], moves_m.shape[1]), dtype=jnp.float32)
        else:
            mask = jnp.asarray(mask, dtype=jnp.float32)
        has_policy = jnp.array(1.0)
    else:
        moves_m = jnp.zeros((n, 1, 64), dtype=jnp.float32)
        expert_idx = jnp.zeros((n,), dtype=jnp.int32)
        mask = jnp.ones((n, 1), dtype=jnp.float32)
        has_policy = jnp.array(0.0)

    arr = _to_arr(base)
    if fix_G:
        arr = arr.at[0].set(1.0)

    grad_fn = jax.grad(lambda a: loss_fn(a, M, Y, moves_m, expert_idx, has_policy, mask))
    for _ in range(steps):
        g = grad_fn(arr)
        if fix_G:
            g = g.at[0].set(0.0)
        arr = arr - lr * g
        arr = arr.at[0].set(jnp.clip(arr[0], 0.01, 50.0))
        arr = arr.at[2].set(jnp.clip(arr[2], 1.0, 10.0))
        arr = arr.at[4].set(jnp.clip(arr[4], 0.01, 500.0))
        arr = arr.at[5].set(jnp.clip(arr[5], 0.01, 50.0))
        arr = arr.at[6].set(jnp.clip(arr[6], 0.0, 50.0))
    return _from_arr(arr)
