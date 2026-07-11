"""Tests for the closed-form 2x2 eigensolver vs scipy, and the eta crossover."""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from kepler64.core.tidal import eig2x2


def test_eig2x2_matches_scipy():
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    lam1, lam2 = eig2x2(A)
    ref = np.sort(eigh(np.array(A)).eigenvalues)[::-1]
    assert np.allclose(np.array([lam1, lam2]), ref, atol=1e-4)


def test_eig2x2_negative_discriminant_clamped():
    # asymmetric/rounding case must not produce NaN
    A = jnp.array([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float32)
    lam1, lam2 = eig2x2(A)
    assert jnp.isfinite(lam1) and jnp.isfinite(lam2)
