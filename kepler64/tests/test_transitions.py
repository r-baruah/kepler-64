import chess
import jax.numpy as jnp
import pytest

from kepler64.core.fastboard import FastBoard
from kepler64.core.transitions import child_mass_vector


def test_capture_accretion_increases_capturer_magnitude():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    fb = FastBoard.from_chess(board)
    parent = fb.mass_vector()
    move = (chess.E4, chess.D5, 0)
    child = child_mass_vector(fb, move, parent)
    assert float(child[chess.D5]) == pytest.approx(1.8)


def test_prior_accretion_survives_quiet_move():
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    fb = FastBoard.from_chess(board)
    parent = fb.mass_vector().at[chess.E4].set(2.5)
    child = child_mass_vector(fb, (chess.E4, chess.E5, 0), parent)
    assert float(child[chess.E5]) == 2.5
    assert float(child[chess.E4]) == 0.0


def test_black_capture_accretion_keeps_negative_sign():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1")
    fb = FastBoard.from_chess(board)
    child = child_mass_vector(fb, (chess.D5, chess.E4, 0), fb.mass_vector())
    assert float(child[chess.E4]) == pytest.approx(-1.8)


def test_en_passant_captures_the_passed_pawn():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    fb = FastBoard.from_chess(board)
    parent = fb.mass_vector()
    child = child_mass_vector(fb, (chess.E5, chess.D6, 0), parent)
    assert float(child[chess.D6]) == pytest.approx(1.8)
    assert float(child[chess.D5]) == 0.0


def test_promotion_quiet_push_uses_piece_mass():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    fb = FastBoard.from_chess(board)
    child = child_mass_vector(fb, (chess.A7, chess.A8, 5), fb.mass_vector())
    assert float(child[chess.A8]) == pytest.approx(9.0)
    assert float(child[chess.A7]) == 0.0


def test_promotion_capture_accrues_captured_rook_mass():
    board = chess.Board("r3k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    fb = FastBoard.from_chess(board)
    child = child_mass_vector(fb, (chess.A7, chess.A8, 5), fb.mass_vector())
    assert float(child[chess.A8]) == pytest.approx(9.0 + 0.8 * 5.0)


def test_castling_preserves_rook_accretion():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    fb = FastBoard.from_chess(board)
    parent = fb.mass_vector().at[chess.H1].set(5.7)
    child = child_mass_vector(fb, (chess.E1, chess.G1, 0), parent)
    assert float(child[chess.G1]) == pytest.approx(1000.0)
    assert float(child[chess.H1]) == 0.0
    assert float(child[chess.E1]) == 0.0
    assert float(child[chess.F1]) == pytest.approx(5.7)
