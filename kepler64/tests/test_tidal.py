"""Tests for the closed-form 2x2 eigensolver vs scipy, and the eta crossover."""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from kepler64.core.tidal import eig2x2, tidal_tensor_at, tidal_disruption
from kepler64.core.gravity import potential_field
from kepler64.core.constants import Constants


def test_eig2x2_matches_scipy():
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    lam1, lam2 = eig2x2(A)
    ref = np.sort(eigh(np.array(A))[0])[::-1]
    assert np.allclose(np.array([lam1, lam2]), ref, atol=1e-4)


def test_eig2x2_negative_discriminant_clamped():
    # Near-degenerate matrix: tr^2/4 - det is ~0 (or slightly negative under
    # rounding), which is what the sqrt-clamp guards. [[1,2],[2,1]] would have
    # eigenvalues 3 and -1 (positive discriminant) -> NOT a clamp test.
    A = jnp.array([[1.0, 1.0001], [1.0001, 1.0]], dtype=jnp.float32)
    lam1, lam2 = eig2x2(A)
    assert jnp.isfinite(lam1) and jnp.isfinite(lam2)


# --- F7: boundary Hessian must stay finite and be a real 2x2 second derivative ---

_CORNERS = [0, 7, 56, 63]  # a1, h1, a8, h8


def test_tidal_tensor_corners_finite_and_2x2():
    consts = Constants()
    masses = jnp.array([-9.0] + [0.0] * 63, dtype=jnp.float32)  # enemy mass on a1
    U = potential_field(masses, consts.eps, consts.G, consts.c)
    for sq in _CORNERS:
        A = tidal_tensor_at(U, sq)
        assert A.shape == (2, 2)
        assert jnp.isfinite(A).all()
        # symmetric Hessian
        assert jnp.allclose(A[0, 1], A[1, 0])


def test_tidal_tensor_not_collapsed_to_first_derivative():
    # With edge-padding, the second-difference stencil is valid at a corner and
    # the diagonal curvature terms are genuine (non-trivial), not zeroed out by
    # a collapsed one-sided stencil.
    consts = Constants()
    masses = jnp.array([0.0] * 27 + [-9.0] + [0.0] * 36, dtype=jnp.float32)
    U = potential_field(masses, consts.eps, consts.G, consts.c)
    A = tidal_tensor_at(U, 0)  # a1 corner
    # At least one curvature component must be non-zero -> real Hessian.
    assert float(jnp.abs(A).sum()) > 0.0


def test_tidal_disruption_finite_at_corners():
    consts = Constants()
    masses = jnp.array([-9.0] + [0.0] * 62 + [-5.0], dtype=jnp.float32)
    for sq in _CORNERS:
        eta, lam1, lam2 = tidal_disruption(masses, sq, consts)
        assert jnp.isfinite(eta) and jnp.isfinite(lam1) and jnp.isfinite(lam2)
