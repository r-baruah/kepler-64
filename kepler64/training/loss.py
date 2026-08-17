"""Loss: outcome prediction + expert-move policy, through the physics engine.

chess:  two supervisory signals, both derived from REAL games — either external
        data (Lichess puzzles / PGNs) or the universe's OWN self-play (the
        self-play pipeline in training/selfplay.py supplies both):
          * outcome  — logistic on win/draw/loss.
          * policy   — the physics score should rank the expert's move above
                       all legal alternatives (behavioural cloning through the
                       gravity kernel). Self-teaching (deeper self-search is its
                       own expert target) is the same pipeline with an internal
                       expert instead of an external one.
physics: backpropagate through the ENTIRE physics engine (Plummer -> tidal ->
         eta -> sigmoid) into the 14 trainable leaves (see
         TRAINABLE_LEAVES in core/constants.py). Those are the "weights" —
         real, physical knobs, not a faked evaluation. The parameter array and
         its physical bounds come from ONE place (core/constants.py), so the
         loss and the trainer can never disagree about the layout.

`params` layout (14 leaves): TRAINABLE_LEAVES order. mat_gain has a minimum of
1.0 inside the loss (see below) so no training run can rediscover the scale at
which a captured rook stopped registering.
"""

import jax
import jax.numpy as jnp

from ..core.evaluate import _score_core, multiverse_score_white
from ..core.constants import (
    Constants as _Constants, clip_leaves_traced, LEAF_LO_F, LEAF_HI_F,
)

# NOTE: `loss` is intentionally NOT @jax.jit. It branches on `use_multiverse`
# (a Python bool), while the training step that calls it is jitted by the
# optimizer loop. `_score_core` remains a traceable pure-JAX body.


@jax.jit
def _unpack(p):
    # 14 trainable physical leaves — TRAINABLE_LEAVES order (shared with
    # core.constants.leaves_to_array). The array is projected into its
    # physical bounds before unpacking so every downstream term sees an
    # in-bounds value.
    p = clip_leaves_traced(p)
    lo = LEAF_LO_F
    hi = LEAF_HI_F
    G = jnp.clip(p[0], lo[0], hi[0])
    eps = jnp.clip(p[1], lo[1], hi[1])
    c = jnp.clip(p[2], lo[2], hi[2])
    roche = jnp.clip(p[3], lo[3], hi[3])
    bonus = jnp.clip(p[4], lo[4], hi[4])
    kgain = jnp.clip(p[5], lo[5], hi[5])
    gamma = jnp.clip(p[6], lo[6], hi[6])
    Rg = jnp.clip(p[7], lo[7], hi[7])
    # mat_gain floor of 1.0 (not the leaf lower bound 0.0): at mat_gain<1 a
    # captured rook stops outranking positional swings — the exact scale that
    # produced the measured free-rook/piece sacrifice behaviour. Training is
    # free to raise it, never to rediscover that failure.
    mat_gain = jnp.clip(p[8], 1.0, hi[8])
    lambda_delta = jnp.clip(p[9], lo[9], hi[9])
    com_gain = jnp.clip(p[10], lo[10], hi[10])
    inertia_gain = jnp.clip(p[11], lo[11], hi[11])
    entropy_gain = jnp.clip(p[12], lo[12], hi[12])
    lambda_drift = jnp.clip(p[13], lo[13], hi[13])
    return (G, eps, c, roche, bonus, kgain, gamma, Rg, mat_gain,
            lambda_delta, com_gain, inertia_gain, entropy_gain, lambda_drift)


def loss(params, M, Y, moves_m, expert_idx, has_policy, mask=None, turns=None,
         tau: float = 1.0, margin: float = 0.0, key=None, use_multiverse: bool = False,
         K: int = 8, sigma: float = 0.1, out_scale: float = 30.0):
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
    out_scale   : outcome sigmoid scale. The physics eval is in units of ~±10s
                  for decisive material swings; with unit scale its sigmoid is
                  saturated on every training position and the outcome gradient
                  vanishes. Scaling by ~30 puts typical scores inside the
                  sigmoid's informative band.
    """
    G, eps, c, roche, bonus, kgain, gamma, Rg, mat_gain, ld, cg, ig, eg, dr = _unpack(params)
    # mref is a FIXED unit scale (not trained) — keeps the tidal index well
    # conditioned.
    _mref = _Constants().mref

    # ---- outcome term ----------------------------------------------------
    # _score_core is White-perspective; Y is from White's view, so this is
    # consistent for both sides to move. (Outcome term is static — no parent.)
    S = out_scale * jax.vmap(lambda m: _score_core(m, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain, ld, cg, ig, eg, dr))(M)
    y = (Y + 1.0) / 2.0  # 0 (black win) .. 1 (white win)
    ce = -jnp.mean(y * jax.nn.log_sigmoid(S) + (1.0 - y) * jax.nn.log_sigmoid(-S))

    # ---- policy term: expert move should score highest -------------------
    # Score every child from the SIDE-TO-MOVE perspective (flip for Black),
    # mask dummies to -inf, then cross-entropy onto the expert move.
    # With use_multiverse, each child is scored under the Layer-2 Bayesian
    # average over K posterior realizations of the physics (the "Multiverse"),
    # which makes the physics signal discriminative between sibling moves.
    _const = _Constants(G=G, eps=eps, c=c, roche=roche, bonus=bonus,
                        kgain=kgain, gamma=gamma, Rg=Rg, mref=_mref,
                        mat_gain=mat_gain, lambda_delta=ld,
                        com_gain=cg, inertia_gain=ig,
                        entropy_gain=eg, lambda_drift=dr)

    def _policy_row(child_m, msk, turn, row_key, parent):
        if use_multiverse:
            Kc = child_m.shape[0]
            # one posterior seed per child, derived from the row seed
            child_keys = jax.random.split(row_key, Kc)
            white = jax.vmap(
                lambda mm, ck: multiverse_score_white(mm, _const, ck, K=K, sigma=sigma, parent=parent)
            )(child_m, child_keys)
        else:
            white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain, ld, cg, ig, eg, dr, parent))(child_m)
        # _score_core is WHITE-perspective, so White-to-move wants the HIGHEST
        # white score and Black-to-move wants the LOWEST -> flip for Black ONLY.
        side = jnp.where(turn > 0.0, -white, white)
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        return jax.nn.log_softmax(side / tau)

    if turns is None:
        turns = jnp.zeros((moves_m.shape[0],), dtype=jnp.float32)

    if use_multiverse and key is not None:
        n = moves_m.shape[0]
        row_keys = jax.random.split(key, n)  # one posterior seed per position
    else:
        row_keys = jnp.zeros((moves_m.shape[0], 2), dtype=jnp.uint32)

    # Parent mass vector per position (the position from which the children are
    # generated) — supplies the move-sensitivity (delta) terms their reference.
    parents = M

    logp = jax.vmap(_policy_row)(moves_m, mask, turns, row_keys, parents)   # (N, K)
    policy = -jnp.mean(jnp.take_along_axis(logp, expert_idx[:, None], axis=1))
    policy = jnp.where(has_policy > 0.0, policy, 0.0)

    # ---- pairwise margin ranking (optional, easier landscape) ------------
    # For each position, compare the expert child against the best OTHER legal
    # child: expert score should exceed it by at least `margin`.  Computed
    # unconditionally; when margin=0 the term is identically 0, so the jit'd
    # graph stays static (JAX cannot branch on a traced scalar).  In
    # outcome-only mode moves_m is a single dummy row, which makes mr=0 too.
    def _margin_row(child_m, msk, turn, ei, parent):
        white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain, ld, cg, ig, eg, dr, parent))(child_m)
        side = jnp.where(turn > 0.0, -white, white)
        side = jnp.where(msk > 0.5, side, -jnp.inf)
        exp = side[ei]
        neg = jnp.max(jnp.where(jnp.arange(side.shape[0]) == ei, -jnp.inf, side))
        return jnp.maximum(0.0, margin - (exp - neg))
    mr = jax.vmap(_margin_row)(moves_m, mask, turns, expert_idx, parents)
    policy = policy + jnp.mean(mr)

    # ---- soft monotonicity prior on c ------------------------------------
    prior = 0.1 * jnp.maximum(0.0, 2.0 - c) + 0.1 * jnp.maximum(0.0, c - 10.0)
    return ce + 0.5 * policy + prior


@jax.jit
def _row_scores(child_m, msk, turn, p, parent):
    G, eps, c, roche, bonus, kgain, gamma, Rg, mat_gain, ld, cg, ig, eg, dr = _unpack(p)
    _mref = _Constants().mref
    # Thread the parent mass vector so the measured ranking matches the kernel
    # the SEARCH uses (the move-sensitivity delta terms are active in play).
    white = jax.vmap(lambda mm: _score_core(mm, G, eps, c, roche, bonus, kgain, gamma, Rg, _mref, mat_gain, ld, cg, ig, eg, dr, parent))(child_m)
    side = jnp.where(turn > 0.0, -white, white)
    side = jnp.where(msk > 0.5, side, -jnp.inf)
    return side


def policy_metrics(constants, M, Y, turns, moves_m, mask, expert_idx,
                   is_capture_m=None):
    """Ranking-quality of the physics eval on held-out positions.

    Returns a dict with:
      top1      : fraction where expert move is ranked #1 (harsh, ~random here)
      mrr       : Mean Reciprocal Rank of the expert move (rewards "close")
      mrr_capture : MRR restricted to positions whose expert move is a capture
      mrr_quiet   : MRR restricted to positions whose expert move is quiet
    Vectorised (double-vmap), cheap enough to call on the val split.

    `is_capture_m` (N,) optional float mask: 1 where the expert move is a
    capture. When supplied, the capture/quiet MRRs are honest splits; without
    it they fall back to the overall MRR (legacy behaviour).
    """
    p = _to_params_vec(constants)
    M = jnp.asarray(M, dtype=jnp.float32)
    moves_m = jnp.asarray(moves_m, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    expert_idx = jnp.asarray(expert_idx, dtype=jnp.int32)
    turns = jnp.asarray(turns, dtype=jnp.float32)

    def _one(child_m, msk, turn, ei, parent):
        side = _row_scores(child_m, msk, turn, p, parent)
        # rank of expert: 1 + number of children scored strictly higher
        better = jnp.sum(side > side[ei])
        rank = better + 1
        return rank, (side[ei] == jnp.max(side))  # rank, is_top1

    rank, top1 = jax.vmap(_one)(moves_m, mask, turns, expert_idx, M)
    top1 = float(jnp.mean(top1.astype(jnp.float32)))
    mrr = float(jnp.mean(1.0 / rank.astype(jnp.float32)))
    out = {"top1": top1, "mrr": mrr, "mrr_capture": mrr, "mrr_quiet": mrr}
    if is_capture_m is not None:
        ic = jnp.asarray(is_capture_m, dtype=jnp.float32)
        inv = 1.0 / rank.astype(jnp.float32)
        n_cap = jnp.maximum(1.0, jnp.sum(ic))
        n_q = jnp.maximum(1.0, jnp.sum(1.0 - ic))
        out["mrr_capture"] = float(jnp.sum(ic * inv) / n_cap)
        out["mrr_quiet"] = float(jnp.sum((1.0 - ic) * inv) / n_q)
    return out


def policy_accuracy(constants, M, Y, turns, moves_m, mask, expert_idx) -> float:
    """Backward-compatible: fraction where the physics score ranks expert #1."""
    return policy_metrics(constants, M, Y, turns, moves_m, mask, expert_idx)["top1"]


def _to_params_vec(constants):
    """Pack a Constants into the traced param vector (shared leaf order)."""
    from ..core.constants import leaves_to_array
    return leaves_to_array(constants)
