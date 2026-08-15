"""Evaluation: physics (tidal disruption) + a soft, differentiable tactical layer.

chess:  the score tells the search which move keeps your King bound and the
        enemy King disrupted.
physics: Eval = +eta_enemy - eta_self + bonus(your force on enemy king)
                - penalty(enemy force on your king) + global field-energy edge.

        Disruption is SOURCE-ATTRIBUTED: a king is disrupted by the OPPONENT's
        masses, not by its own. This is the physically correct reading (a body's
        own gravity binds it; the enemy's gravity tears it) and it removes the
        old perverse incentive where capturing an enemy piece looked BAD because
        it lowered the total force on the enemy king.

        `roche` is the learned disruption threshold: the force-sigmoid flips
        around it, so gradient descent learns the tidal limit instead of us
        hand-picking it.

All heavy math is pure JAX. Training imports ``_score_core`` as a stable private
alias of the shared traced body so ``jax.grad`` can move every trainable leaf.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gravity import force_field, potential_field, _COORDS, _DIST2, _DIST
from .tidal import tidal_tensor_at, eig2x2
from .constants import Constants

_MAX_MOVES = 218  # theoretical max legal moves in chess


def _king_idx(masses, sign: float):
    """Integer (traced) index of the king of the given color (sign +1 white, -1 black).

    Uses an atol=5.0 closeness test so accretion-shifted King masses (e.g.
    1002.4, 1008.0) are still matched. Returns a valid integer index suitable
    for JAX advanced indexing; if no King is found it returns 0 (a1) as a safe
    default. Callers that need a hard guarantee should check `_king_found` first.
    """
    mask = jnp.isclose(jnp.abs(masses), 1000.0, atol=5.0) & (jnp.sign(masses) == sign)
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _king_found(masses, sign: float) -> bool:
    """True iff a King of the given color is actually present (no silent a1)."""
    mask = jnp.isclose(jnp.abs(masses), 1000.0, atol=5.0) & (jnp.sign(masses) == sign)
    return jnp.any(mask)


def _eta(U, king_sq, king_mass, Rg: float = 1.0, mref: float = 3.5) -> float:
    """Tidal-disruption index at a king from the supplied potential field U.

    eta = Rg_eff^3 * lambda1 / mref^2: the dominant tidal eigenvalue (tearing)
    scaled by the king's spatial extent (Rg) and a reference tidal-stress
    scale (mref).  G is intentionally ABSENT: lambda1 is already proportional
    to G (the tidal tensor is the Hessian of the potential), so including G in
    the denominator would cancel it and make eta independent of the field
    strength — physically the disruption criterion is G-independent, which is
    correct.  We divide by mref^2 (a minor-piece-scale constant), NOT by
    Mking^2: the King's own 1000-mass self-gravity would otherwise shrink eta
    to ~1e-6 and the Roche limit would never be reachable.

    The king's radius of gyration Rg_eff scales as the cube root of its mass
    (constant-density self-gravitating body), so a King that has accreted mass
    is more spatially extended and correspondingly easier to tidally disrupt.
    This is the physics-native "overextended piece is fragile" coupling (the
    missing C13 hook): accretion grows the King, and growth raises eta.

    Larger eta = closer to disruption.
    """
    A = tidal_tensor_at(U, king_sq)
    lam1, _ = eig2x2(A)
    Rg_eff = Rg * (jnp.abs(king_mass) / 1000.0) ** (1.0 / 3.0)
    return (Rg_eff**3) * lam1 / (mref**2 + 1e-9)


def _eta_pair(masses, G, eps, c, Rg, mref):
    """Return (eta_b, eta_w): tidal stress on enemy / own King from White's view.

    eta_b = stress YOUR masses exert on the BLACK king  (good for White)
    eta_w = stress THEIR masses exert on the WHITE king (bad for White)
    Both neutralized to 0 if a King is missing (corrupted board).
    """
    abs_m = jnp.abs(masses)
    white_m = jnp.where(masses > 0.0, abs_m, 0.0)
    black_m = jnp.where(masses < 0.0, abs_m, 0.0)
    U_w = potential_field(white_m, eps, G, c)
    U_b = potential_field(black_m, eps, G, c)
    wk = _king_idx(masses, 1.0)
    bk = _king_idx(masses, -1.0)
    kings_ok = _king_found(masses, 1.0) & _king_found(masses, -1.0)
    wk_m = jnp.abs(masses[wk])
    bk_m = jnp.abs(masses[bk])
    eta_b = jnp.where(kings_ok, _eta(U_w, bk, bk_m, Rg, mref), 0.0)
    eta_w = jnp.where(kings_ok, _eta(U_b, wk, wk_m, Rg, mref), 0.0)
    return eta_b, eta_w


class EvalTerms(NamedTuple):
    tidal_enemy: float
    tidal_self: float
    force_enemy_king: float
    force_own_king: float
    binding: float
    material: float
    delta_tidal: float
    delta_com: float
    delta_inertia: float
    delta_entropy: float
    total: float


def _score_terms_body(masses, G: float, eps: float, c: float, roche: float,
                bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 1.0,
                lambda_delta: float = 0.0, com_gain: float = 0.0,
                inertia_gain: float = 0.0, entropy_gain: float = 0.0,
                parent_masses=None):
    """Evaluation from White's perspective (positive = good for White).

    When `parent_masses` is supplied, the move-sensitivity (delta) terms are
    active: they measure what the MOVE *did* (child minus parent) rather than
    the static look of the child board. This is the only way to get a signal
    that discriminates between sibling moves (thesis G1b / roadmap core
    principle). When `parent_masses is None` the delta terms are zero, so the
    function degrades to a plain static evaluation (used at search leaves where
    no parent is available, and in tests).
    """
    abs_m = jnp.abs(masses)
    white_m = jnp.where(masses > 0.0, abs_m, 0.0)   # your masses
    black_m = jnp.where(masses < 0.0, abs_m, 0.0)   # enemy masses

    # Source-attributed force & potential: white's field vs black's field.
    F_w = force_field(white_m, eps, G, c)   # force everywhere due to YOUR masses
    F_b = force_field(black_m, eps, G, c)   # force everywhere due to ENEMY masses
    U_w = potential_field(white_m, eps, G, c)  # your potential (tears their king)
    U_b = potential_field(black_m, eps, G, c)  # their potential (tears your king)

    wk = _king_idx(masses, 1.0)
    bk = _king_idx(masses, -1.0)
    kings_ok = _king_found(masses, 1.0) & _king_found(masses, -1.0)

    # Disruption is caused by the OPPONENT's masses only.
    eta_w = _eta(U_b, wk, jnp.abs(masses[wk]), Rg, mref)   # enemy masses stressing your king : bad for White
    eta_b = _eta(U_w, bk, jnp.abs(masses[bk]), Rg, mref)   # your masses stressing enemy king : good for White

    # Force-based disruption, flipped around the learned Roche threshold.
    bonus_b = bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_w[bk] + 1e-9) - roche))
    pen_w = -bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_b[wk] + 1e-9) - roche))

    # If either King is missing (corrupted board), the disruption/force terms
    # are meaningless — neutralize them instead of silently scoring at a1.
    eta_w = jnp.where(kings_ok, eta_w, 0.0)
    eta_b = jnp.where(kings_ok, eta_b, 0.0)
    bonus_b = jnp.where(kings_ok, bonus_b, 0.0)
    pen_w = jnp.where(kings_ok, pen_w, 0.0)

    # Gravitational BINDING ENERGY edge (army-internal cohesion):
    #   U_bind_white = dot(white_co, U_wc) / 2
    # where white_co EXCLUDES the white king (and U_wc its source field).
    # U_w is negative potential, so a well-coordinated/centralized army has a
    # more negative binding energy. We reward OUR pieces being more bound than
    # THEIRS: gamma * (bind_b - bind_w).
    #
    # The previous global_edge = gamma * (sum|F_white|² - sum|F_black|²) PENALISED
    # centre play (a centre pawn radiates ~50% more total field energy than an edge
    # pawn), which is what made the engine obsessively open with h4/h3 — the
    # measured h-file bias. Binding energy is piece-position sensitive in the
    # CORRECT direction: central, coordinated pieces bind more strongly.
    #
    # The KING is excluded from this term (both as source and as receiver).
    # Physically the king is the tidal DETECTOR — its own 1000-mass gravity is
    # handled by the disruption/force terms, not by the cohesion edge. Keeping
    # king-piece pairs here leaks the king's 1000^2 self-energy back into every
    # positional relationship: any piece moving away from its own king paid a
    # ~100+ point binding penalty (measured: 1.e4 = -155 purely from binding),
    # so the engine huddled its army around the king and treated flank pushes
    # (which barely disturb the king's well) as least-bad moves — the "outward
    # comet" flank-pawn bias, with the causality inverted from what it looks
    # like. U_w/U_b are already computed above; the king's source field is
    # subtracted analytically (U_king_src = +1000 * gate(d)/r), so this adds
    # only vector ops, not extra field passes.
    gwk = jax.nn.sigmoid(c - _DIST[:, wk])
    gbk = jax.nn.sigmoid(c - _DIST[:, bk])
    rwk = jnp.sqrt(_DIST2[:, wk] + eps * eps)
    rbk = jnp.sqrt(_DIST2[:, bk] + eps * eps)
    king_m = 1000.0
    U_wc = jnp.where(kings_ok, U_w + king_m * gwk / rwk, U_w)
    U_bc = jnp.where(kings_ok, U_b + king_m * gbk / rbk, U_b)
    white_co = white_m.at[wk].set(0.0)
    black_co = black_m.at[bk].set(0.0)
    # potential_field includes each body's potential at its own square. Remove
    # that diagonal term; otherwise the king's 1000^2 self-energy overwhelms
    # every positional relationship and creates unstable flank preferences.
    self_scale = G * jax.nn.sigmoid(c) / jnp.sqrt(eps * eps)
    bind_w = (jnp.dot(white_co, U_wc) + self_scale * jnp.dot(white_co, white_co)) / 2.0
    bind_b = (jnp.dot(black_co, U_bc) + self_scale * jnp.dot(black_co, black_co)) / 2.0
    global_edge = gamma * (bind_b - bind_w)

    # Gravitational material edge: you command more mass -> stronger field.
    # Rewards captures / winning material; gives the search a clean, varied
    # gradient instead of the flat equilibrium of the disruption term alone.
    material = mat_gain * (jnp.sum(white_m) - jnp.sum(black_m))

    delta_eta_term = jnp.array(0.0, dtype=masses.dtype)
    com_term = jnp.array(0.0, dtype=masses.dtype)
    inertia_term = jnp.array(0.0, dtype=masses.dtype)
    entropy_term = jnp.array(0.0, dtype=masses.dtype)

    # ── Move-sensitivity (delta) terms ──────────────────────────────────────
    # Each is (child_quantity - parent_quantity); none of these fire when there
    # is no parent (static eval / search leaf). They are the real anti-flatness
    # signal: they vary between sibling moves because the move *changed* the
    # field, not because the resulting position happens to look good.
    if parent_masses is not None:
        # G1b — Δη: the tidal-disruption rate. Reward increasing the enemy
        # King's tidal stress more than your own. dη/dt, physically.
        p_eta_b, p_eta_w = _eta_pair(parent_masses, G, eps, c, Rg, mref)
        delta_eta = (eta_b - p_eta_b) - (eta_w - p_eta_w)
        delta_eta_term = lambda_delta * delta_eta

        # T2.1 — Center-of-mass advance delta. Reward shifting our ARMY's mass
        # centroid (kings excluded — the king is the protected detector, not a
        # weapon; including it let a 1000-mass king rank advance swamp every
        # other signal and marched the king into the enemy camp) toward the
        # enemy / away from home more than they do. Mass-weighting stays plain
        # (rank average over the army): an earlier "attack-axis" variant that
        # multiplied each piece's mass by exp(-dfile^2/2) re-weighted lateral
        # moves by up to ~54x and made castling appear catastrophically bad
        # (the kingside rook "became" supermassive when it left the h-file).
        p_abs = jnp.abs(parent_masses)
        p_w = jnp.where(parent_masses > 0.0, p_abs, 0.0)
        p_b = jnp.where(parent_masses < 0.0, p_abs, 0.0)
        p_wk_sq = _king_idx(parent_masses, 1.0)
        p_bk_sq = _king_idx(parent_masses, -1.0)
        p_w = p_w.at[p_wk_sq].set(0.0)
        p_b = p_b.at[p_bk_sq].set(0.0)
        w_total = jnp.sum(white_co) + 1e-9
        b_total = jnp.sum(black_co) + 1e-9
        com_rank_w = jnp.dot(white_co, _COORDS[:, 1]) / w_total
        com_rank_b = jnp.dot(black_co, _COORDS[:, 1]) / b_total
        p_wt = jnp.sum(p_w) + 1e-9
        p_bt = jnp.sum(p_b) + 1e-9
        p_com_w = jnp.dot(p_w, _COORDS[:, 1]) / p_wt
        p_com_b = jnp.dot(p_b, _COORDS[:, 1]) / p_bt
        com_delta = (com_rank_w - p_com_w) - (p_com_b - com_rank_b)
        com_term = com_gain * com_delta

        # T2.2 — Attack moment-of-inertia delta. The tidal tearing of the
        # enemy king acts along the king-king axis, so lateral distance enters
        # softened (k_ax = 0.2): a piece that advances along the enemy king's
        # column tightens the attack (rewarded), a lateral shuffle mostly
        # doesn't. Masses are NOT re-weighted by the axis — only the distance
        # measure is anisotropic (the mass-weighted Σ stays a plain "moment").
        def _ax_dist2(king_sq):
            d = _COORDS - _COORDS[king_sq]
            return d[:, 1] ** 2 + 0.2 * d[:, 0] ** 2

        dist2_to_bk = _ax_dist2(bk)
        dist2_to_wk = _ax_dist2(wk)
        I_attack_w = jnp.dot(white_co, dist2_to_bk)
        I_attack_b = jnp.dot(black_co, dist2_to_wk)
        p_I_attack_w = jnp.dot(p_w, dist2_to_bk)
        p_I_attack_b = jnp.dot(p_b, dist2_to_wk)
        # smaller I_attack_w is better; smaller I_attack_b is worse for us
        inertia_delta = (p_I_attack_w - I_attack_w) - (I_attack_b - p_I_attack_b)
        inertia_term = inertia_gain * inertia_delta / 100.0

        # T2.4 — Entropy (coordination) delta. Reward OUR army (kings excluded)
        # becoming more concentrated (lower entropy) and THEIRS more scattered.
        p_w_entropy = _shannon_entropy(p_w)
        p_b_entropy = _shannon_entropy(p_b)
        entropy_delta = (p_w_entropy - _shannon_entropy(white_co)) \
            + (_shannon_entropy(black_co) - p_b_entropy)
        entropy_term = entropy_gain * entropy_delta

    total = (eta_b - eta_w + bonus_b + pen_w + global_edge + material
             + delta_eta_term + com_term + inertia_term + entropy_term)
    return EvalTerms(eta_b, -eta_w, bonus_b, pen_w, global_edge, material,
                     delta_eta_term, com_term, inertia_term, entropy_term, total)


def _score_body(masses, G: float, eps: float, c: float, roche: float,
                bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 1.0,
                lambda_delta: float = 0.0, com_gain: float = 0.0,
                inertia_gain: float = 0.0, entropy_gain: float = 0.0,
                parent_masses=None):
    return _score_terms_body(masses, G, eps, c, roche, bonus, kgain, gamma,
                             Rg, mref, mat_gain, lambda_delta, com_gain,
                             inertia_gain, entropy_gain, parent_masses).total


@jax.jit
def _score_core_static(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                       mref, mat_gain, lambda_delta=0.0, com_gain=0.0,
                       inertia_gain=0.0, entropy_gain=0.0, parent_masses=None):
    return _score_terms_body(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                             mref, mat_gain, lambda_delta, com_gain,
                             inertia_gain, entropy_gain, parent_masses).total


def _shannon_entropy(m):
    """Shannon entropy of a (non-negative) mass distribution, normalized to ~[0,1]."""
    total = jnp.sum(m) + 1e-9
    p = m / total
    log_p = jnp.where(p > 1e-9, jnp.log(p + 1e-9), 0.0)
    H = -jnp.dot(p, log_p)
    return H / jnp.log(64.0)


def score_white(masses: "jnp.ndarray", constants, parent=None) -> float:
    """Evaluate from White's perspective. Pass `parent` (a mass vector) to
    activate the move-sensitivity (delta) terms."""
    # Jitted inference path: `_score_core_static` reuses the shared traced body
    # (the training alias `_score_core` stays a plain traceable function). A
    # `None` parent is a concrete value under jit, so the static branch fires
    # exactly as before; an array parent activates the delta terms. `batch_score`
    # still vmaps this fine.
    return _score_core_static(
        masses, constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg,
        constants.mref, constants.mat_gain, constants.lambda_delta,
        constants.com_gain, constants.inertia_gain, constants.entropy_gain,
        parent,
    )


# Stable private import used by the trainer.
_score_core = _score_body


def score_white_terms(masses: "jnp.ndarray", constants, parent=None) -> EvalTerms:
    """Return weighted contributions that sum exactly to ``score_white``."""
    return _score_terms_body(
        masses, constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg,
        constants.mref, constants.mat_gain, constants.lambda_delta,
        constants.com_gain, constants.inertia_gain, constants.entropy_gain,
        parent,
    )


def evaluate(board, constants) -> float:
    """Score from the perspective of the side to move."""
    masses = board.mass_vector()
    s = float(score_white(masses, constants))
    return s if board.turn == 0 else -s


def batch_score(masses_list, turns, constants, pad: int = _MAX_MOVES,
                parents=None):
    """Pad a list of child mass vectors to `pad` (the 218-max), vmap-evaluate.

    Returns (N,) side-to-move scores with dummy slots masked to -inf so they
    never win a node.

    If `parents` is given (a (N,64) array of the parent mass vectors), the
    move-sensitivity (delta) terms are active for each child.
    """
    n = len(masses_list)
    buf = jnp.zeros((pad, 64), dtype=jnp.float32)
    if n:
        stacked = jnp.stack([jnp.asarray(m, dtype=jnp.float32) for m in masses_list])
        buf = buf.at[:n].set(stacked)
    turns_buf = jnp.array([*(turns if n else [0]), *([0] * (pad - n))], dtype=jnp.int32)
    if parents is not None:
        parent_buf = jnp.zeros((pad, 64), dtype=jnp.float32)
        parent_buf = parent_buf.at[:n].set(jnp.asarray(parents, dtype=jnp.float32))
        white = jax.vmap(score_white, in_axes=(0, None, 0))(buf, constants, parent_buf)
    else:
        white = jax.vmap(score_white, in_axes=(0, None))(buf, constants)  # (pad,)
    side = jnp.where(turns_buf == 0, white, -white)
    mask = jnp.concatenate([jnp.ones(n), jnp.zeros(pad - n)])
    return jnp.where(mask > 0.5, side, -jnp.inf)


# ── Layer 2: "the Multiverse" ────────────────────────────────────────────────
# Bayesian model average over a posterior of physics constants. A move that is
# good across many possible universes is boosted; one that is only good under a
# single fragile regime is suppressed. This is the project's differentiator and
# the mechanism that makes the physics signal discriminative between sibling
# moves (the single-realization Layer-1 score is nearly flat across them).
import jax.random as _jr


def _perturb_constants(base: "Constants", key, sigma: float = 0.1) -> "Constants":
    """Sample one realization of the physics from the posterior around `base`."""
    k1, k2, k3 = _jr.split(key, 3)
    return Constants(
        G=base.G * (1.0 + sigma * _jr.normal(k1)),
        eps=base.eps * (1.0 + sigma * _jr.normal(k2)),
        c=jnp.clip(base.c * (1.0 + sigma * _jr.normal(k3)), 1.0, 10.0),
        roche=base.roche, bonus=base.bonus, kgain=base.kgain,
        gamma=base.gamma, Rg=base.Rg, mref=base.mref, mat_gain=base.mat_gain,
        lambda_delta=base.lambda_delta, com_gain=base.com_gain,
        inertia_gain=base.inertia_gain, entropy_gain=base.entropy_gain,
    )


def multiverse_score_white(masses: "jnp.ndarray", constants: "Constants",
                           key, K: int = 8, sigma: float = 0.1,
                           parent=None) -> "jnp.ndarray":
    """Layer-2 evaluation: mean of `score_white` over K posterior realizations.

    Returns a traced scalar (jit/vmap-safe). `key` should be unique per call
    site (fold it in at the caller) so different rows sample different universes.
    `parent` (mass vector) is threaded into each realization so the
    move-sensitivity (delta) terms stay active under the Bayesian average.
    """
    keys = _jr.split(key, K)
    scores = jnp.stack([
        score_white(masses, _perturb_constants(constants, k, sigma), parent)
        for k in keys
    ])
    return jnp.mean(scores)


def score_white_layer2(masses: "jnp.ndarray", constants: "Constants",
                       key, K: int = 8, sigma: float = 0.1,
                       parent=None) -> "jnp.ndarray":
    """Convenience: Layer-1 by default, Layer-2 when `use_multiverse` is set.

    Search/inference entry point. Kept thin so callers don't care which layer.
    """
    return multiverse_score_white(masses, constants, key, K=K, sigma=sigma,
                                  parent=parent)
