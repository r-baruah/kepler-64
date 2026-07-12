"""Loss: outcome prediction + expert-move policy, through the physics engine.

chess:  two supervisory signals, both derived from REAL games / strong engines
        (no chess heuristics invented by us):
          * outcome  — logistic on win/draw/loss (the old signal).
          * policy   — the physics score should rank the expert's move above
                       all legal alternatives (behavioural cloning through the
                       gravity kernel).
physics: backpropagate through the ENTIRE physics engine (Plummer -> tidal ->
        eta -> sigmoid) into G, eps, c, roche, AND the disruption scales
        (bonus, kgain, gamma) and Rg (king extent). Those are the "weights" —
        real, physical knobs, not a faked evaluation. The `c` monotonicity
        prior is preserved.

`params` layout (jnp array of 8): [G, eps, c, roche, bonus, kgain, gamma, Rg].
"""

import jax
import jax.numpy as jnp

from ..core.evaluate import _score_core


@jax.jit
def _unpack(p):
    return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]


@jax.jit
def loss(params, M, Y, moves_m, expert_idx, has_policy, mask=None):
    """All-scalar, jit-compatible.

    M           : (N,64) mass vectors of positions            (outcome + policy)
    Y           : (N,) outcomes in {-1,0,+1} from White's view (outcome only)
    moves_m     : (N, K, 64) mass vectors of each child move   (policy only)
    expert_idx  : (N,) index of the expert move among the K    (policy only)
    has_policy  : bool scalar — 0 disables the policy term
    mask        : (N, K) 1 for real child moves, 0 for padded dummies
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg = _unpack(params)
    G = jnp.clip(G, 0.01, 50.0)  # gravity must stay attractive (positive)
    c = jnp.clip(c, 1.0, 10.0)  # monotonicity prior (hard clamp)
    Rg = jnp.clip(Rg, 0.1, 10.0)  # king extent must stay physical

    # ---- outcome term ----------------------------------------------------
    S = jax.vmap(lambda m: _score_core(m, G, eps, c, roche, bonus, kgain, gamma, Rg))(M)
    y = (Y + 1.0) / 2.0  # 0 (black win) .. 1 (white win)
    ce = -jnp.mean(y * jax.nn.log_sigmoid(S) + (1.0 - y) * jax.nn.log_sigmoid(-S))

    # ---- policy term: expert move should score highest -------------------
    # Score every child move from the side-to-move perspective, mask dummies
    # to -inf so they can never be selected, then cross-entropy onto expert.
    def _policy_row(child_m, msk):
        side = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg))(child_m)
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        return jax.nn.log_softmax(side)

    logp = jax.vmap(_policy_row)(moves_m, mask)            # (N, K)
    policy = -jnp.mean(jnp.take_along_axis(logp, expert_idx[:, None], axis=1))
    policy = jnp.where(has_policy > 0.0, policy, 0.0)

    # ---- soft monotonicity prior on c ------------------------------------
    prior = 0.1 * jnp.maximum(0.0, 2.0 - c) + 0.1 * jnp.maximum(0.0, c - 10.0)
    return ce + 0.5 * policy + prior
