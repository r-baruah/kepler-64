"""c-prior gradient tests: the prior must be differentiable and c traceable."""

import dataclasses

import jax
import jax.numpy as jnp

from kepler64.core.constants import Constants


def _c_prior_loss(c_val):
    """Loss whose gradient wrt c should be non-zero when c is out of [2, 10]."""
    consts = dataclasses.replace(Constants(), c=c_val)
    # Negate: c_prior is a reward (<=0); minimise the penalty magnitude.
    return -consts.c_prior()


def test_c_prior_grad_nonzero_and_finite_when_out_of_range():
    # c below the lower bound (2.0) -> active jnp.maximum branch -> real grad.
    g = jax.grad(_c_prior_loss)(jnp.float32(1.0))
    assert jnp.isfinite(g)
    assert g != 0.0


def test_c_prior_grad_nonzero_above_upper_bound():
    g = jax.grad(_c_prior_loss)(jnp.float32(12.0))
    assert jnp.isfinite(g)
    assert g != 0.0


def test_c_can_be_traced_jax_value():
    # c passed as a jnp float must flow through jax.grad without error.
    c_traced = jnp.float32(3.0)
    loss = _c_prior_loss(c_traced)
    assert jnp.isfinite(loss)
    g = jax.grad(_c_prior_loss)(c_traced)
    assert jnp.isfinite(g)


def test_untrained_delta_terms_default_to_neutral():
    c = Constants()
    assert c.lambda_delta == 0.0
    assert c.com_gain == 0.0
    assert c.inertia_gain == 0.0
    assert c.entropy_gain == 0.0
