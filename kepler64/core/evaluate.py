"""Evaluation: the universe's physics decides the score. No chess tables, no
books — every term below is a real gravitational quantity of the mass lattice.

chess:  the score tells the search which move keeps your King bound and tears
        the enemy King apart, while commanding more matter than the enemy.
physics: Eval = +eta_enemy - eta_self                          (tidal stress)
                + bonus(your ARMY force on enemy king)         (disruption gauge)
                - penalty(their ARMY force on your king)
                + gamma * (their binding - your binding)       (cohesion edge)
                + mat_gain * (your mass - their mass)          (matter edge)
                + move-sensitivity deltas (deta, CoM flux, inertia, entropy)
                + drift (Verlet-projected d-eta over the horizon)

        Disruption is SOURCE-ATTRIBUTED: a king is disrupted by the OPPONENT'S
        masses, not by its own.

        The KING IS THE DETECTOR, NOT A FIELD SOURCE: every gauge that judges
        attack/danger (eta, force, drift) is sourced from the ARMY masses only.
        The kings are 1000-mass bodies (~96% of one side's total mass); if they
        sourced the attacking field, every king-facing-king configuration would
        read as a huge mutual attack and the real piece threats (~1e-3 gauge
        units) would drown. The king's own gravity is kept where it belongs:
        the binding/cohesion term and the disruption threshold.

All heavy math is pure JAX. Training imports ``_score_core`` as a stable private
alias of the shared traced body so ``jax.grad`` can move every trainable leaf.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gravity import force_field, potential_field, _COORDS
from .tidal import tidal_tensor_at, eig2x2
from .constants import Constants

_MAX_MOVES = 218  # theoretical max legal moves in chess


def _king_idx(masses, sign: float):
    """Traced index of the king of the given color (sign +1 white, -1 black).

    The king is the unique supermassive body (~1000); accretion and Lorentz
    boosts only GROW |m|, so the king is the square with the LARGEST |m| among
    supermassive squares (|m| >= 500). The old isclose(|m|, 1000, atol=5) test
    silently lost kings that had accreted a queen (|m| = 1007.2 > 1005) and
    zeroed every disruption term. Returns 0 (a1) only when no king exists.
    """
    mask = (jnp.abs(masses) >= 500.0) & (jnp.sign(masses) == sign)
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _king_found(masses, sign: float) -> bool:
    """True iff a King of the given color is actually present (no silent a1)."""
    mask = (jnp.abs(masses) >= 500.0) & (jnp.sign(masses) == sign)
    return jnp.any(mask)


def _army_split(masses):
    """(white_army, black_army, wk, bk, kings_ok).

    Army = every mass EXCEPT the kings. The returned vectors still carry the
    king squares' mass at 0, so the army mass total and the per-square layout
    are both exact.
    """
    abs_m = jnp.abs(masses)
    white_m = jnp.where(masses > 0.0, abs_m, 0.0)
    black_m = jnp.where(masses < 0.0, abs_m, 0.0)
    wk = _king_idx(masses, 1.0)
    bk = _king_idx(masses, -1.0)
    kings_ok = _king_found(masses, 1.0) & _king_found(masses, -1.0)
    return (white_m.at[wk].set(0.0), black_m.at[bk].set(0.0), wk, bk, kings_ok)


def _eta(U, king_sq, king_mass, Rg: float = 1.0, mref: float = 3.5) -> float:
    """Tidal-disruption index at a king from the supplied potential field U.

    eta = Rg_eff^3 * lambda1 / mref^2: the dominant tidal eigenvalue (tearing)
    scaled by the king's spatial extent (Rg) and a reference tidal-stress
    scale (mref). G is intentionally ABSENT: lambda1 is already proportional
    to G (the tidal tensor is the Hessian of the potential), so including G in
    the denominator would cancel it and make eta independent of the field
    strength — physically the disruption criterion is G-independent.

    The king's radius of gyration Rg_eff scales as the cube root of its mass
    (constant-density self-gravitating body), so a King that has accreted mass
    is more spatially extended and correspondingly easier to tidally disrupt.

    Larger eta = closer to disruption.
    """
    A = tidal_tensor_at(U, king_sq)
    lam1, _ = eig2x2(A)
    # cbrt floor: dummy/padded child rows (all-zero masses) land a "king" at
    # a1 with zero mass. Without the floor, d/dx x^(1/3) is infinite at 0 and
    # the jnp.where(kings_ok, ...) mask only hides the VALUE — the NaN
    # gradient still flows (0 * NaN = NaN), poisoning training. At real king
    # masses (>=500) the floor never activates.
    Rg_eff = Rg * jnp.maximum(jnp.abs(king_mass) / 1000.0, 1e-9) ** (1.0 / 3.0)
    return (Rg_eff**3) * lam1 / (mref**2 + 1e-9)


def _eta_pair(masses, G, eps, c, Rg, mref):
    """Return (eta_b, eta_w): tidal stress on enemy / own King from White's view.

    eta_b = stress YOUR ARMY exerts on the BLACK king  (good for White)
    eta_w = stress THEIR ARMY exerts on the WHITE king (bad for White)
    Both neutralized to 0 if a King is missing (corrupted board).
    """
    white_co, black_co, wk, bk, kings_ok = _army_split(masses)
    U_w = potential_field(white_co, eps, G, c)
    U_b = potential_field(black_co, eps, G, c)
    wk_m = jnp.abs(masses[wk])
    bk_m = jnp.abs(masses[bk])
    eta_b = jnp.where(kings_ok, _eta(U_w, bk, bk_m, Rg, mref), 0.0)
    eta_w = jnp.where(kings_ok, _eta(U_b, wk, wk_m, Rg, mref), 0.0)
    return eta_b, eta_w


def _tidal_tensor_at_pos(attacker_m, pos, G, eps, c):
    """Analytical Hessian (tidal tensor) of the gated Plummer potential at a
    continuous position `pos` — the continuous analogue of
    tidal_tensor_at()'s finite differences, so it resolves arbitrarily small
    King drifts instead of snapping to a lattice square.

    Both distance measures are floored with the Plummer softening eps: the
    King's projected coordinate can land EXACTLY on a lattice square, making
    the raw sqrt(sum(d^2)) zero at that row. sqrt has an unbounded backward
    gradient at 0, and on degenerate (dummy/padded, all-zero) rows that NaN
    poisoning spreads through the whole eval gradient. The floor is
    physically consistent — eps IS the softening length — and numerically
    inert everywhere except that one exact-coincidence row.
    """
    d = _COORDS - pos                                          # (64, 2)
    s2 = jnp.sum(d**2, axis=-1) + eps**2                        # (64,) softened
    dist = jnp.sqrt(s2)                                          # reach gate dist
    s = dist
    s3 = s2 * s                               # s^3
    s5 = s3 * s2                              # s^5
    gate = jax.nn.sigmoid(c - dist)
    w = G * jnp.abs(attacker_m) * gate        # (64,) gated weighted masses
    tr = jnp.sum(w / s3)
    xx = jnp.sum(w * d[:, 0] ** 2 / s5)
    xy = jnp.sum(w * d[:, 0] * d[:, 1] / s5)
    yy = jnp.sum(w * d[:, 1] ** 2 / s5)
    return jnp.array([[tr - 3.0 * xx, -3.0 * xy],
                      [-3.0 * xy, tr - 3.0 * yy]])


def _eta_at_pos(attacker_m, pos, king_mass, G, eps, c, Rg, mref):
    """Tidal-disruption index at a continuous position."""
    A = _tidal_tensor_at_pos(attacker_m, pos, G, eps, c)
    lam1, _ = eig2x2(A)
    # Same cbrt floor as _eta: dummy rows hit zero king mass; the infinite
    # backward gradient of x^(1/3) at 0 must never reach the optimizer.
    Rg_eff = Rg * jnp.maximum(jnp.abs(king_mass) / 1000.0, 1e-9) ** (1.0 / 3.0)
    return (Rg_eff**3) * lam1 / (mref**2 + 1e-9)


def _eta_drift(attacker_m, king_sq, king_mass, G, eps, c, Rg, mref,
               steps=4, dt=0.1):
    """Projected tidal-stress change (impending collapse) for one King.

    A short energy-conserving Leapfrog rollout advances the King's continuous
    coordinate under the ATTACKER's field (the field that tears it), and the
    tidal index at the projected location minus the current one is the net d-eta
    over the horizon. Positive = the King is drifting into higher stress.
    """
    am = jnp.abs(attacker_m)
    pos = _COORDS[king_sq]
    vel = jnp.zeros(2, dtype=pos.dtype)

    def _step(carry, _):
        p, v = carry
        d = _COORDS - p
        r2 = jnp.sum(d**2, axis=-1) + eps**2
        a = G * jnp.einsum("i,ij->j", am / (r2 * jnp.sqrt(r2)), d)
        v = v + a * dt
        p = p + v * dt
        return (p, v), p

    (p_end, _), _ = jax.lax.scan(_step, (pos, vel), None, length=steps)
    eta_now = _eta_at_pos(attacker_m, pos, king_mass, G, eps, c, Rg, mref)
    eta_end = _eta_at_pos(attacker_m, p_end, king_mass, G, eps, c, Rg, mref)
    return eta_end - eta_now


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
    drift: float
    total: float


def _score_terms_body(masses, G: float, eps: float, c: float, roche: float,
                bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 2.0,
                lambda_delta: float = 0.0, com_gain: float = 0.0,
                inertia_gain: float = 0.0, entropy_gain: float = 0.0,
                lambda_drift: float = 0.0, parent_masses=None):
    """Evaluation from White's perspective (positive = good for White).

    When `parent_masses` is supplied, the move-sensitivity (delta) terms are
    active: they measure what the MOVE *did* (child minus parent) rather than
    the static look of the child board — the signal that discriminates between
    sibling moves (thesis G1b / roadmap core principle). With
    `parent_masses is None` the delta terms are zero, so the function degrades
    to a plain static evaluation (search leaves, tests, standalone scoring).
    """
    abs_m = jnp.abs(masses)
    white_m = jnp.where(masses > 0.0, abs_m, 0.0)   # your masses (all)
    black_m = jnp.where(masses < 0.0, abs_m, 0.0)   # enemy masses (all)

    # Army split: kings are detectors, not field sources (see module docstring).
    white_co, black_co, wk, bk, kings_ok = _army_split(masses)

    # Source-attributed ARMY field: your army tears their king.
    U_w = potential_field(white_co, eps, G, c)   # your army potential
    U_b = potential_field(black_co, eps, G, c)   # their army potential
    F_w = force_field(white_co, eps, G, c)       # your army force everywhere
    F_b = force_field(black_co, eps, G, c)       # their army force everywhere

    eta_w = _eta(U_b, wk, jnp.abs(masses[wk]), Rg, mref)   # bad for White
    eta_b = _eta(U_w, bk, jnp.abs(masses[bk]), Rg, mref)   # good for White

    # Disruption GAUGE: the army force magnitude at the king, flipped around the
    # learned Roche threshold. Army-sourced, so in the opening both gauges are
    # tiny (near-neutral) and they rise only when a real piece attack lands on
    # a king. The old king-sourced gauge read ~1.0 everywhere (enemy king
    # pulling across the board), sat on the sigmoid midpoint, and contributed a
    # constant +/-150 that drowned every other signal.
    bonus_b = bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_w[bk] + 1e-9) - roche))
    pen_w = -bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_b[wk] + 1e-9) - roche))

    # Verlet tidal-drift (impending collapse): project each King forward under
    # the OPPONENT'S ARMY field and read the change in tidal stress.
    drift_b = _eta_drift(white_co, bk, jnp.abs(masses[bk]), G, eps, c, Rg, mref)
    drift_w = _eta_drift(black_co, wk, jnp.abs(masses[wk]), G, eps, c, Rg, mref)

    # If either King is missing (corrupted board), the disruption/force terms
    # are meaningless — neutralize them instead of silently scoring at a1.
    eta_w = jnp.where(kings_ok, eta_w, 0.0)
    eta_b = jnp.where(kings_ok, eta_b, 0.0)
    bonus_b = jnp.where(kings_ok, bonus_b, 0.0)
    pen_w = jnp.where(kings_ok, pen_w, 0.0)
    drift_b = jnp.where(kings_ok, drift_b, 0.0)
    drift_w = jnp.where(kings_ok, drift_w, 0.0)

    # Gravitational BINDING ENERGY edge (army-internal cohesion):
    #   U_bind = sum_i m_i * U_army(i) / 2   over each side's ARMY,
    # the true pairwise binding energy of the army (each pair counted once, self
    # pairs are constant and subtract out of the difference; kings excluded as
    # sources AND receivers — the king's 1000-mass well is not "coordination").
    # More negative = more bound/cohesive. We reward OUR pieces being more
    # bound than THEIRS: gamma * (bind_b - bind_w).
    bind_w = jnp.dot(white_co, U_w) / 2.0
    bind_b = jnp.dot(black_co, U_b) / 2.0
    global_edge = gamma * (bind_b - bind_w)

    # Gravitational material edge: you command more matter -> stronger field.
    # KING MASSES CANCELED: both kings weigh 1000, so the army-mass difference
    # equals the full-mass difference but is immune to accretion/Lorentz noise
    # on the kings' 1000-scale bodies. With mat_gain=2 a captured rook moves
    # the score by ~10 — decisively above the positional noise floor (~1-3),
    # which is the fix for the "give away free rooks" behaviour.
    material = mat_gain * (jnp.sum(white_co) - jnp.sum(black_co))

    delta_eta_term = jnp.array(0.0, dtype=masses.dtype)
    com_term = jnp.array(0.0, dtype=masses.dtype)
    inertia_term = jnp.array(0.0, dtype=masses.dtype)
    entropy_term = jnp.array(0.0, dtype=masses.dtype)

    # ── Move-sensitivity (delta) terms ──────────────────────────────────────
    # Each is (child_quantity - parent_quantity); none fire without a parent.
    if parent_masses is not None:
        # G1b — Δη: the tidal-disruption rate. Reward increasing the enemy
        # King's tidal stress more than your own. dη/dt, physically.
        p_eta_b, p_eta_w = _eta_pair(parent_masses, G, eps, c, Rg, mref)
        delta_eta = (eta_b - p_eta_b) - (eta_w - p_eta_w)
        delta_eta_term = lambda_delta * delta_eta

        # T2.1 — Momentum flux (mass-weighted CoM advance, DIFFERENCE form).
        #   P_rank = Σ m_i * rank_i  (army only). White's forward flux is
        #   F_w = P_w(child)-P_w(parent) (toward +y); Black's forward flux is
        #   F_b = -(P_b(child)-P_b(parent)) (Black advances toward -y). The
        #   momentum advantage is F_w - F_b = (P_w - p_P_w) + (P_b - p_P_b).
        # SIGN NOTE: this is a WHITE-perspective score, so EVERY term must be
        # positive iff it is good for White — including when the last move was
        # Black's. The previous (p_P_b - P_b) form was flipped: a Black advance
        # (P_b decreasing) read as positive for White, which corrupted Black's
        # reply pick at the search leaves.
        # WHY difference form: the previous mean-rank form divided by the army
        # total with a 1e-9 floor. When one army was nearly captured the
        # fraction blew up (a lone pawn "advanced" scored ~1400) and when the
        # enemy army was EMPTY the parent/child means were equal and the whole
        # term canceled — the measured bug where a free-rook capture scored
        # com_delta = 0 while a suicidal edge-rook charge scored +240. Mass
        # sums have no denominator at all; they are exact and bounded
        # (<= 39 * 7 = 273), so this term can no longer dwarf a real capture.
        p_w_full = jnp.where(parent_masses > 0.0, jnp.abs(parent_masses), 0.0)
        p_b_full = jnp.where(parent_masses < 0.0, jnp.abs(parent_masses), 0.0)
        p_wk_sq = _king_idx(parent_masses, 1.0)
        p_bk_sq = _king_idx(parent_masses, -1.0)
        p_w = p_w_full.at[p_wk_sq].set(0.0)
        p_b = p_b_full.at[p_bk_sq].set(0.0)
        P_w = jnp.dot(white_co, _COORDS[:, 1])
        P_b = jnp.dot(black_co, _COORDS[:, 1])
        p_P_w = jnp.dot(p_w, _COORDS[:, 1])
        p_P_b = jnp.dot(p_b, _COORDS[:, 1])
        com_delta = ((P_w - p_P_w) + (P_b - p_P_b)) / 10.0
        com_term = com_gain * com_delta

        # T2.2 — Attack moment-of-inertia delta. The tidal tearing of the
        # enemy king acts along the king axis, so lateral distance enters
        # softened (k_ax = 0.2). Masses are NOT re-weighted by the axis — only
        # the distance measure is anisotropic. Divided by 100 to land in the
        # same ~unit band as the other delta terms.
        # SIGN NOTE (WHITE perspective): closing OUR pieces toward the enemy
        # king (I_attack_w decreasing) is good for us; closing THEIR pieces
        # toward our king (I_attack_b decreasing) is bad. The old
        # -(I_attack_b - p_I_attack_b) form flipped the second half, so a Black
        # piece converging on the White king read as a White advantage.
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
        inertia_delta = (p_I_attack_w - I_attack_w) + (I_attack_b - p_I_attack_b)
        inertia_term = inertia_gain * inertia_delta / 100.0

        # T2.4 — Entropy (coordination) delta. Reward OUR army (kings excluded)
        # becoming more concentrated (lower entropy) and THEIRS more scattered.
        p_w_entropy = _shannon_entropy(p_w)
        p_b_entropy = _shannon_entropy(p_b)
        entropy_delta = (p_w_entropy - _shannon_entropy(white_co)) \
            + (_shannon_entropy(black_co) - p_b_entropy)
        entropy_term = entropy_gain * entropy_delta

    drift_term = lambda_drift * (drift_b - drift_w)

    total = (eta_b - eta_w + bonus_b + pen_w + global_edge + material
             + delta_eta_term + com_term + inertia_term + entropy_term
             + drift_term)
    return EvalTerms(eta_b, -eta_w, bonus_b, pen_w, global_edge, material,
                     delta_eta_term, com_term, inertia_term, entropy_term,
                     drift_term, total)


def _score_body(masses, G: float, eps: float, c: float, roche: float,
                bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 2.0,
                lambda_delta: float = 0.0, com_gain: float = 0.0,
                inertia_gain: float = 0.0, entropy_gain: float = 0.0,
                lambda_drift: float = 0.0, parent_masses=None):
    return _score_terms_body(masses, G, eps, c, roche, bonus, kgain, gamma,
                             Rg, mref, mat_gain, lambda_delta, com_gain,
                             inertia_gain, entropy_gain, lambda_drift,
                             parent_masses).total


@jax.jit
def _score_core_static(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                       mref, mat_gain, lambda_delta=0.0, com_gain=0.0,
                       inertia_gain=0.0, entropy_gain=0.0, lambda_drift=0.0,
                       parent_masses=None):
    return _score_terms_body(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                             mref, mat_gain, lambda_delta, com_gain,
                             inertia_gain, entropy_gain, lambda_drift,
                             parent_masses).total


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
    return _score_core_static(
        masses, constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg,
        constants.mref, constants.mat_gain, constants.lambda_delta,
        constants.com_gain, constants.inertia_gain, constants.entropy_gain,
        constants.lambda_drift, parent,
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
        constants.lambda_drift, parent,
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
    """Sample one realization of the physics from the posterior around `base`.

    Every trainable leaf is perturbed multiplicatively (log-normal-ish, so
    positive constants stay positive and the perturbation is scale-free). The
    reach gate `c` is clipped back to its physical prior [1, 10]. `mref` is a
    fixed unit scale (not a trainable weight), so it is held constant.
    """
    keys = _jr.split(key, 14)
    kG, ke, kc, kr, kb, kkg, kg, kR, kmg, kld, kcg, kig, keg, kdr = keys
    return Constants(
        G=base.G * (1.0 + sigma * _jr.normal(kG)),
        eps=base.eps * (1.0 + sigma * _jr.normal(ke)),
        c=jnp.clip(base.c * (1.0 + sigma * _jr.normal(kc)), 1.0, 10.0),
        roche=base.roche * (1.0 + sigma * _jr.normal(kr)),
        bonus=base.bonus * (1.0 + sigma * _jr.normal(kb)),
        kgain=base.kgain * (1.0 + sigma * _jr.normal(kkg)),
        gamma=base.gamma * (1.0 + sigma * _jr.normal(kg)),
        Rg=base.Rg * (1.0 + sigma * _jr.normal(kR)),
        mref=base.mref,
        mat_gain=base.mat_gain * (1.0 + sigma * _jr.normal(kmg)),
        lambda_delta=base.lambda_delta * (1.0 + sigma * _jr.normal(kld)),
        com_gain=base.com_gain * (1.0 + sigma * _jr.normal(kcg)),
        inertia_gain=base.inertia_gain * (1.0 + sigma * _jr.normal(kig)),
        entropy_gain=base.entropy_gain * (1.0 + sigma * _jr.normal(keg)),
        lambda_drift=base.lambda_drift * (1.0 + sigma * _jr.normal(kdr)),
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

    def _one(k):
        return score_white(masses, _perturb_constants(constants, k, sigma), parent)

    return jnp.mean(jax.vmap(_one)(keys))


def score_white_layer2(masses: "jnp.ndarray", constants: "Constants",
                       key, K: int = 8, sigma: float = 0.1,
                       parent=None) -> "jnp.ndarray":
    """Convenience: Layer-1 by default, Layer-2 when `use_multiverse` is set."""
    return multiverse_score_white(masses, constants, key, K=K, sigma=sigma,
                                  parent=parent)
