"""Loss: logistic on game outcomes + c monotonicity prior.

physics: backpropagate through the entire physics engine (Plummer -> tidal ->
        eta) using a logistic loss on win/draw/loss, so the gravitational
        constant is genuinely learned by gradient descent. The c prior keeps the
        retarded-potential story alive and is written with jnp.maximum so it is
        differentiable when c is a traced constant.

This module is jit-compatible: pass `params` (a jnp array [G, eps, c, roche]),
`M` (N,64) mass matrix, and `Y` (N,) outcomes in {-1, 0, +1} (White's view).
"""

import jax
import jax.numpy as jnp

from ..core.evaluate import _score_core


@jax.jit
def loss(params, M, Y):
    G, eps, c, roche = params[0], params[1], params[2], params[3]
    c = jnp.clip(c, 1.0, 10.0)  # monotonicity prior (hard clamp)

    S = jax.vmap(lambda m: _score_core(m, G, eps, c, roche))(M)  # (N,)
    y = (Y + 1.0) / 2.0  # 0 (black win) .. 1 (white win)
    # Numerically stable binary cross-entropy: avoids log(1 - sigmoid(S)) which
    # is NaN when S is large (sigmoid(S) -> 1, 1 - p -> 0/negative under JIT).
    ce = -jnp.mean(y * jax.nn.log_sigmoid(S) + (1.0 - y) * jax.nn.log_sigmoid(-S))

    # soft monotonicity prior on c (differentiable)
    prior = 0.1 * jnp.maximum(0.0, 2.0 - c) + 0.1 * jnp.maximum(0.0, c - 10.0)
    return ce + prior
