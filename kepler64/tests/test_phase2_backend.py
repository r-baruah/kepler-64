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


def test_engine_warmup():
    engine = RocheEngine(load_trained=False)
    # warmup should return engine instance and run without error
    ret = engine.warmup()
    assert ret is engine


def test_fastboard_legal_moves_and_king_tracking():
    from kepler64.core.fastboard import FastBoard

    fens = [
        chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]
    for fen in fens:
        b = chess.Board(fen)
        fb = FastBoard.from_chess(b)

        # O(1) king square lookup check
        assert fb.king_sq(True) == b.king(chess.WHITE)
        assert fb.king_sq(False) == b.king(chess.BLACK)

        # Move generation equivalence
        chess_moves = set((m.from_square, m.to_square, m.promotion or 0) for m in b.legal_moves)
        fb_moves = set(fb.legal_moves())
        assert fb_moves == chess_moves, f"Move mismatch on {fen}"


def test_parallel_head_to_head_matches():
    from scripts.credibility_gate import head_to_head
    from kepler64.core.constants import Constants

    c = Constants()
    saved = []

    def _on_game_end(hist, match):
        saved.append(match["games"])

    res = head_to_head(c, c, games=2, move_ms=10.0, max_plies=4, workers=2, on_game_end=_on_game_end)
    assert res["games"] == 2
    assert len(res["history"]) == 2
    assert len(saved) == 2


