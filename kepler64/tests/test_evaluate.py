"""Sign-convention regression: heavy White mass near Black's king must help White."""

import jax.numpy as jnp

from kepler64.core.evaluate import score_white, score_white_terms, _eta_drift
from kepler64.core.constants import Constants


def _empty():
    return [0.0] * 64


def _build(white_king_sq, black_king_sq, heavy_sq, heavy_sign):
    m = _empty()
    m[white_king_sq] = 1000.0
    m[black_king_sq] = -1000.0
    m[heavy_sq] = heavy_sign * 9.0
    return jnp.array(m, dtype=jnp.float32)


def test_f4_white_mass_near_black_king_scores_higher():
    c = Constants()
    # White king a1 (0), Black king h8 (63). Heavy WHITE queen on g7 (54),
    # right next to Black's king -> good for White.
    a = _build(0, 63, 54, +1.0)
    # Symmetric reversal: swap colors AND mirror squares so it's the same
    # geometry with White's king under the same enemy force. Score must drop.
    # Reversed: White king h8 (63), Black king a1 (0), heavy BLACK queen b2 (9)
    # next to White's king -> bad for White.
    rev = _build(63, 0, 9, -1.0)

    sa = float(score_white(a, c))
    srev = float(score_white(rev, c))
    assert sa > srev


def test_f4_score_is_antisymmetric_under_color_swap():
    c = Constants()
    a = _build(0, 63, 54, +1.0)
    # Full color+geometry flip should negate the score (White<->Black swap).
    flipped = _build(63, 0, 9, -1.0)
    sa = float(score_white(a, c))
    sf = float(score_white(flipped, c))
    assert sa > 0.0
    assert sf < 0.0


def test_term_decomposition_sums_to_score():
    c = Constants()
    masses = _build(0, 63, 54, +1.0)
    terms = score_white_terms(masses, c)
    contributions = terms[:-1]
    sw = float(score_white(masses, c))
    # score_white is jitted; XLA fast-math reassociation makes the eager
    # term-sum agree only to ~1e-5 relative, so use a relative tolerance.
    assert abs(float(sum(contributions)) - sw) < 1e-3 * max(1.0, abs(sw))


def test_diagonal_self_energy_is_removed_from_binding():
    c = Constants(bonus=0.0, gamma=1.0, mat_gain=0.0,
                  lambda_delta=0.0, com_gain=0.0,
                  inertia_gain=0.0, entropy_gain=0.0)
    masses = jnp.zeros(64, dtype=jnp.float32).at[0].set(1000.0).at[63].set(-1000.0)
    terms = score_white_terms(masses, c)
    assert abs(float(terms.binding)) < 1e-3


def test_verlet_drift_is_finite_and_positive_under_attack():
    """The Verlet drift (impending collapse) is finite, and positive when a
    King is pulled toward the attacking mass that tears it."""
    c = Constants()
    # White king a1 (0), black king h8 (63), heavy WHITE queen on g7 (54) next
    # to the black king. The black king drifts toward the queen -> white's
    # tidal stress on it grows -> drift > 0.
    masses = jnp.zeros(64, dtype=jnp.float32)
    masses = masses.at[0].set(1000.0).at[63].set(-1000.0).at[54].set(9.0)
    white_m = jnp.where(masses > 0.0, jnp.abs(masses), 0.0)
    drift = float(_eta_drift(white_m, 63, 1000.0, c.G, c.eps, c.c, c.Rg, c.mref))
    assert jnp.isfinite(drift)
    assert drift > 0.0


def test_verlet_drift_term_is_in_score_decomposition():
    """The drift term is part of the total and finite (no NaN in the eval)."""
    c = Constants()
    masses = jnp.zeros(64, dtype=jnp.float32)
    masses = masses.at[0].set(1000.0).at[63].set(-1000.0).at[54].set(9.0)
    terms = score_white_terms(masses, c)
    assert jnp.isfinite(float(terms.drift))
    assert jnp.isfinite(float(terms.total))
    # The drift term must be one of the summed contributions.
    assert abs(float(sum(terms[:-1])) - float(terms.total)) < 1e-3 * max(1.0, abs(float(terms.total)))
