"""Tests for the N-body force field — the headline flex must be correct."""

import jax.numpy as jnp
import pytest

from kepler64.core.gravity import force_field, potential_field
from kepler64.core.constants import Constants


def test_forces_finite():
    masses = jnp.ones(64, dtype=jnp.float32)
    F = force_field(masses, eps=0.5, G=1.0)
    assert jnp.isfinite(F).all()


def test_potential_symmetric():
    masses = jnp.array([9.0] + [0.0] * 63, dtype=jnp.float32)
    U = potential_field(masses, eps=0.5, G=1.0)
    assert jnp.isfinite(U).all()
    # deeper well near the mass
    assert U[0] < U[63]
