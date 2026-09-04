"""Tests for Phase 2 backend optimizations: fused policy/margin loss and multiverse methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import chess

from kepler64 import (
    RocheEngine,
    Constants,
    apply_capture,
    uncertainty_field,
    stokes_flow,
    observe_update,
    multiverse_score_white,
)
from kepler64.training.loss import loss, _unpack
from kepler64.core.evaluate import _score_core


def test_fused_loss_scalar_and_grad():
    params = Constants().to_array()
    N = 4
    K = 8
    rng = np.random.default_rng(42)

    M = jnp.asarray(rng.standard_normal((N, 64)), dtype=jnp.float32)
    Y = jnp.asarray(rng.choice([-1.0, 0.0, 1.0], size=(N,)), dtype=jnp.float32)
    moves_m = jnp.asarray(rng.standard_normal((N, K, 64)), dtype=jnp.float32)
    expert_idx = jnp.asarray(rng.integers(0, K, size=(N,)), dtype=jnp.int32)
    mask = jnp.ones((N, K), dtype=jnp.float32)
    turns = jnp.zeros((N,), dtype=jnp.float32)

    val, grad = jax.value_and_grad(loss)(
        params, M, Y, moves_m, expert_idx, has_policy=1.0, mask=mask, turns=turns, margin=0.5
    )

    assert jnp.isfinite(val)
    assert grad.shape == params.shape
    assert jnp.all(jnp.isfinite(grad))


def test_multiverse_convenience_methods():
    engine = RocheEngine(load_trained=False)

    # 1. Uncertainty field and Stokes flow
    samples = jnp.ones((5, 64), dtype=jnp.float32)
    # inject variance on square 0
    samples = samples.at[0, 0].set(2.0)
    u_field = engine.uncertainty_field(samples)
    assert u_field.shape == (64,)
    assert u_field[0] > 0.0
    assert u_field[1] == 0.0

    flow = engine.stokes_flow(eval_samples=samples)
    assert flow.shape == (64, 2)
    assert jnp.all(jnp.isfinite(flow))

    # Test flow passing uncertainty directly
    flow_u = engine.stokes_flow(uncertainty=u_field)
    assert jnp.allclose(flow, flow_u)

    # 2. Accretion on capture
    masses = jnp.zeros((64,), dtype=jnp.float32).at[0].set(3.0).at[1].set(5.0)
    m_new, rg_new = engine.accrete_capture(masses, captor_sq=0, captured_sq=1, eta_acc=0.8, Rg_old=1.0)
    # captor gets 3.0 + 0.8 * 5.0 = 7.0
    assert float(m_new[0]) == pytest.approx(7.0)
    # captured square is cleared
    assert float(m_new[1]) == pytest.approx(0.0)
    # Rg grows by (7.0 / 3.0) ** (1/3)
    expected_rg = (7.0 / 3.0) ** (1.0 / 3.0)
    assert float(rg_new) == pytest.approx(expected_rg)


def test_roche_engine_play_with_observer():
    board = chess.Board()
    engine = RocheEngine(load_trained=False)
    initial_G = float(engine.constants.G)

    # Play 1 move with observe=True
    move = engine.play(board, depth=1, observe=True)
    assert move is not None
    assert move in board.legal_moves

    # Observer should maintain valid constants
    assert float(engine.constants.G) > 0.0
    arr = engine.constants.to_array()
    assert np.all(np.isfinite(arr))
