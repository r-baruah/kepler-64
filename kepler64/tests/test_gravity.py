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


# --- F22: gravity physics laws ---

def _two_body(sq_i, m_i, sq_j, m_j):
    m = jnp.zeros(64, dtype=jnp.float32)
    m = m.at[sq_i].set(m_i)
    m = m.at[sq_j].set(m_j)
    return m


def test_newtons_third_law():
    # Two masses only: force on i (from j) == -force on j (from i) when the
    # masses are equal, and generally F_i = -F_j scaled by mass ratio. Use
    # equal masses so the accelerations are exact negatives.
    i, j = 0, 9  # a1 and b2 (diagonal neighbours)
    masses = _two_body(i, 5.0, j, 5.0)
    F = force_field(masses, eps=0.5, G=1.0, c=10.0)
    assert jnp.allclose(F[i], -F[j], atol=1e-4)


def test_force_direction_lies_on_the_i_j_axis():
    # A single source at j; the force at i must lie along the i-j line, i.e. the
    # vector F[i] is (anti)parallel to (r_i - r_j). j at sq 9 (x=1,y=1), i at
    # sq 0 (x=0,y=0): displacement is diagonal, so |F_x| == |F_y| and both share
    # the same sign (points consistently along the diagonal).
    i, j = 0, 9
    masses = _two_body(i, 0.0, j, 9.0)
    F = force_field(masses, eps=0.5, G=1.0, c=10.0)
    assert jnp.allclose(F[i, 0], F[i, 1], atol=1e-5)
    assert F[i, 0] != 0.0


def test_force_direction_sign_flips_with_geometry():
    # Moving the source to the opposite side flips the force direction.
    i = 9  # b2 (x=1,y=1)
    left = _two_body(i, 0.0, 0, 9.0)   # source at a1 (x=0,y=0), lower-left
    right = _two_body(i, 0.0, 18, 9.0)  # source at c3 (x=2,y=2), upper-right
    Fl = force_field(left, eps=0.5, G=1.0, c=10.0)
    Fr = force_field(right, eps=0.5, G=1.0, c=10.0)
    assert Fl[i, 0] * Fr[i, 0] < 0.0
    assert Fl[i, 1] * Fr[i, 1] < 0.0


def test_force_finite_with_coincident_softening():
    # Every square occupied -> zero-distance self terms; Plummer softening (eps)
    # must keep everything finite (no division by zero).
    masses = jnp.ones(64, dtype=jnp.float32) * 3.0
    F = force_field(masses, eps=0.5, G=1.0, c=10.0)
    U = potential_field(masses, eps=0.5, G=1.0, c=10.0)
    assert jnp.isfinite(F).all()
    assert jnp.isfinite(U).all()


def test_force_finite_tiny_softening():
    masses = jnp.ones(64, dtype=jnp.float32)
    F = force_field(masses, eps=1e-3, G=1.0, c=10.0)
    assert jnp.isfinite(F).all()
