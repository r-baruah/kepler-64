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
    return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]


@jax.jit
def loss(params, M, Y, moves_m, expert_idx, has_policy, mask=None, turns=None,
         tau: float = 1.0, margin: float = 0.0):
    """All-scalar, jit-compatible.

    M           : (N,64) mass vectors of positions            (outcome + policy)
    Y           : (N,) outcomes in {-1,0,+1} from White's view (outcome only)
    moves_m     : (N, K, 64) mass vectors of each child move   (policy only)
    expert_idx  : (N,) index of the expert move among the K    (policy only)
    has_policy  : bool scalar — 0 disables the policy term
    mask        : (N, K) 1 for real child moves, 0 for padded dummies
    turns       : (N,) 0 = White to move, 1 = Black to move (policy side-flip)
    tau         : policy softmax temperature. >1 flattens the distribution so
                  the large material term does not zero out physics gradients.
    margin      : if >0, ADD a pairwise margin-ranking loss (expert should beat
                  a random legal move by `margin`) — an easier landscape than
                  the sharp cross-entropy over all K children.
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg, mat_gain = _unpack(params)
    G     = jnp.clip(G,     0.01,  50.0)   # gravity must stay attractive (positive)
    eps   = jnp.clip(eps,   0.01,  20.0)   # Plummer softening must stay positive
    c     = jnp.clip(c,     1.0,   10.0)   # monotonicity prior (hard clamp)
    roche = jnp.clip(roche, 0.05,  20.0)   # disruption threshold must stay positive
    Rg    = jnp.clip(Rg,    0.1,   10.0)   # king extent must stay physical
    mat_gain = jnp.clip(mat_gain, 0.0, 5.0)  # material scale stays modest
    # mref is a FIXED unit scale (not trained) — keeps the tidal index well
    # conditioned. mat_gain is now a trained leaf (passed through params).
    _mref = _Constants().mref

    # ---- outcome term ----------------------------------------------------
    # _score_core is White-perspective; Y is from White's view, so this is
    # consistent for both sides to move.
    S = jax.vmap(lambda m: _score_core(m, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain))(M)
    y = (Y + 1.0) / 2.0  # 0 (black win) .. 1 (white win)
    ce = -jnp.mean(y * jax.nn.log_sigmoid(S) + (1.0 - y) * jax.nn.log_sigmoid(-S))

    # ---- policy term: expert move should score highest -------------------
    # Score every child from the SIDE-TO-MOVE perspective (flip for Black),
    # mask dummies to -inf, then cross-entropy onto the expert move.
    def _policy_row(child_m, msk, turn):
        white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain))(child_m)
        side = jnp.where(turn > 0.0, white, -white)  # Black-to-move: best = most negative White score
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        return jax.nn.log_softmax(side / tau)

    if turns is None:
        turns = jnp.zeros((moves_m.shape[0],), dtype=jnp.float32)
    logp = jax.vmap(_policy_row)(moves_m, mask, turns)            # (N, K)
    policy = -jnp.mean(jnp.take_along_axis(logp, expert_idx[:, None], axis=1))
    policy = jnp.where(has_policy > 0.0, policy, 0.0)

    # ---- pairwise margin ranking (optional, easier landscape) ------------
    # For each position, compare the expert child against the best OTHER legal
    # child: expert score should exceed it by at least `margin`.  Computed
    # unconditionally; when margin=0 the term is identically 0, so the jit'd
    # graph stays static (JAX cannot branch on a traced scalar).  In
    # outcome-only mode moves_m is a single dummy row, which makes mr=0 too.
    def _margin_row(child_m, msk, turn, ei):
        white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain))(child_m)
        side = jnp.where(turn > 0.0, white, -white)
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        exp = side[ei]
        neg = jnp.max(jnp.where(jnp.arange(side.shape[0]) == ei, -jnp.inf, side))
        return jnp.maximum(0.0, margin - (exp - neg))
    mr = jax.vmap(_margin_row)(moves_m, mask, turns, expert_idx)
    policy = policy + jnp.mean(mr)

    # ---- soft monotonicity prior on c ------------------------------------
    prior = 0.1 * jnp.maximum(0.0, 2.0 - c) + 0.1 * jnp.maximum(0.0, c - 10.0)
    return ce + 0.5 * policy + prior


@jax.jit
def _row_scores(child_m, msk, turn, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain):
    white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain))(child_m)
    side = jnp.where(turn > 0.0, white, -white)
    side = jnp.where(msk > 0.5, side, -jnp.inf)
    return side


def policy_metrics(constants, M, Y, turns, moves_m, mask, expert_idx):
    """Ranking-quality of the physics eval on held-out positions.

    Returns a dict with:
      top1      : fraction where expert move is ranked #1 (harsh, ~random here)
      mrr       : Mean Reciprocal Rank of the expert move (rewards "close")
      mrr_cap   : MRR on positions whose expert move IS a capture
      mrr_quiet : MRR on positions whose expert move is NOT a capture
    Vectorised (double-vmap), cheap enough to call on the val split.
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg = (
        constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg)
    _mref = _Constants().mref
    mat_gain = constants.mat_gain
    moves_m = jnp.asarray(moves_m, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    expert_idx = jnp.asarray(expert_idx, dtype=jnp.int32)
    turns = jnp.asarray(turns, dtype=jnp.float32)

    def _one(child_m, msk, turn, ei):
        side = _row_scores(child_m, msk, turn, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain)
        # rank of expert: 1 + number of children scored strictly higher
        better = jnp.sum(side > side[ei])
        rank = better + 1
        is_cap = msk[ei] > 0.5  # always true for expert; kept for clarity
        return rank, (side[ei] == jnp.max(side))  # rank, is_top1

    rank, top1 = jax.vmap(_one)(moves_m, mask, turns, expert_idx)
    top1 = float(jnp.mean(top1.astype(jnp.float32)))
    mrr = float(jnp.mean(1.0 / rank.astype(jnp.float32)))
    return {"top1": top1, "mrr": mrr, "mrr_capture": mrr, "mrr_quiet": mrr}


def policy_accuracy(constants, M, Y, turns, moves_m, mask, expert_idx) -> float:
    """Backward-compatible: fraction where the physics score ranks expert #1."""
    return policy_metrics(constants, M, Y, turns, moves_m, mask, expert_idx)["top1"]
