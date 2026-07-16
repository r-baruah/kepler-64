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
from ..core.constants import Constants as _Constants


@jax.jit
def _unpack(p):
    return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]


@jax.jit
def loss(params, M, Y, moves_m, expert_idx, has_policy, mask=None, turns=None):
    """All-scalar, jit-compatible.

    M           : (N,64) mass vectors of positions            (outcome + policy)
    Y           : (N,) outcomes in {-1,0,+1} from White's view (outcome only)
    moves_m     : (N, K, 64) mass vectors of each child move   (policy only)
    expert_idx  : (N,) index of the expert move among the K    (policy only)
    has_policy  : bool scalar — 0 disables the policy term
    mask        : (N, K) 1 for real child moves, 0 for padded dummies
    turns       : (N,) 0 = White to move, 1 = Black to move (policy side-flip)
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg = _unpack(params)
    G     = jnp.clip(G,     0.01,  50.0)   # gravity must stay attractive (positive)
    eps   = jnp.clip(eps,   0.01,  20.0)   # Plummer softening must stay positive
    c     = jnp.clip(c,     1.0,   10.0)   # monotonicity prior (hard clamp)
    roche = jnp.clip(roche, 0.05,  20.0)   # disruption threshold must stay positive
    Rg    = jnp.clip(Rg,    0.1,   10.0)   # king extent must stay physical
    # mref / mat_gain are FIXED unit scales (not trained weights) — the tidal
    # index must stay well-conditioned and the material edge must stay 1:1.
    _mref = _Constants().mref
    _mat = _Constants().mat_gain

    # ---- outcome term ----------------------------------------------------
    # _score_core is White-perspective; Y is from White's view, so this is
    # consistent for both sides to move.
    S = jax.vmap(lambda m: _score_core(m, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, _mat))(M)
    y = (Y + 1.0) / 2.0  # 0 (black win) .. 1 (white win)
    ce = -jnp.mean(y * jax.nn.log_sigmoid(S) + (1.0 - y) * jax.nn.log_sigmoid(-S))

    # ---- policy term: expert move should score highest -------------------
    # Score every child from the SIDE-TO-MOVE perspective (flip for Black),
    # mask dummies to -inf, then cross-entropy onto the expert move.
    def _policy_row(child_m, msk, turn):
        white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, _mat))(child_m)
        side = jnp.where(turn > 0.0, white, -white)  # Black-to-move: best = most negative White score
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        return jax.nn.log_softmax(side)

    if turns is None:
        turns = jnp.zeros((moves_m.shape[0],), dtype=jnp.float32)
    logp = jax.vmap(_policy_row)(moves_m, mask, turns)            # (N, K)
    policy = -jnp.mean(jnp.take_along_axis(logp, expert_idx[:, None], axis=1))
    policy = jnp.where(has_policy > 0.0, policy, 0.0)

    # ---- soft monotonicity prior on c ------------------------------------
    prior = 0.1 * jnp.maximum(0.0, 2.0 - c) + 0.1 * jnp.maximum(0.0, c - 10.0)
    return ce + 0.5 * policy + prior


@jax.jit
def _row_scores(child_m, msk, turn, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, _mat):
    white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, _mat))(child_m)
    side = jnp.where(turn > 0.0, white, -white)
    side = jnp.where(msk > 0.5, side, -jnp.inf)
    return side


def policy_accuracy(constants, M, Y, turns, moves_m, mask, expert_idx) -> float:
    """Fraction of positions where the physics score ranks the expert move #1.

    Vectorised (double-vmap) — no Python loop over examples, so it is cheap
    enough to call every few epochs on the validation split.  Returns a float
    in [0, 1].
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg = (
        constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg)
    _mref = _Constants().mref
    _mat = _Constants().mat_gain
    M = jnp.asarray(M, dtype=jnp.float32)
    turns = jnp.asarray(turns, dtype=jnp.float32)
    moves_m = jnp.asarray(moves_m, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    expert_idx = jnp.asarray(expert_idx, dtype=jnp.int32)

    def _one(child_m, msk, turn):
        side = _row_scores(child_m, msk, turn, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, _mat)
        return jnp.argmax(side) == expert_idx

    correct = jax.vmap(_one)(moves_m, mask, turns)
    return float(jnp.mean(correct.astype(jnp.float32)))
