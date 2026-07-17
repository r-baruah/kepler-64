"""Sign-convention regression: heavy White mass near Black's king must help White."""

import jax.numpy as jnp

from kepler64.core.evaluate import score_white
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
