"""Regression tests for core correctness fixes."""
import chess
import numpy as np
import pytest

from kepler64.core.fastboard import FastBoard
from kepler64.core.transitions import child_mass_vector

_PROMO = {2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK, 5: chess.QUEEN}
_BASE = {2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0}


def _find(fb, f, t, pr):
    for m in fb.legal_moves():
        if m == (f, t, pr):
            return m
    raise AssertionError("move not legal")


@pytest.mark.parametrize("pr", [2, 3, 4, 5])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("white", [True, False])
def test_promotion_colour_fen_and_mass_sign(pr, capture, white):
    if white:
        f, t = 48, (57 if capture else 56)
        fen = "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1" if capture else "7k/P7/8/8/8/8/8/4K3 w - - 0 1"
    else:
        f, t = 8, (1 if capture else 0)
        fen = "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1" if capture else "4k3/8/8/8/8/8/p7/7K b - - 0 1"
    cb = chess.Board(fen)
    fb = FastBoard.from_chess(cb)
    mv = _find(fb, f, t, pr)
    child = fb.apply(mv)
    cb.push(chess.Move(f, t, _PROMO[pr]))
    assert child.to_chess().board_fen() == cb.board_fen()
    parent = np.asarray(fb.mass_vector(4.0), dtype=np.float32)
    cm = child_mass_vector(fb, mv, parent, child_board=child, c_lorentz=4.0)
    sign = 1.0 if white else -1.0
    assert np.sign(cm[t]) == sign
    assert abs(cm[t]) >= _BASE[pr] - 1e-4


_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "4k3/8/8/8/8/8/p7/4K2R b K - 0 1",
    "r3k2r/8/8/8/8/5q2/8/R3K2R w KQkq - 0 1",
]


def _mvset(fb):
    return {(f, t, p) for f, t, p in fb.legal_moves()}


def _cmset(cb):
    return {(m.from_square, m.to_square, m.promotion or 0) for m in cb.legal_moves}


@pytest.mark.parametrize("fen", _FENS)
def test_legal_moves_match_python_chess_depth2(fen):
    cb = chess.Board(fen)
    fb = FastBoard.from_chess(cb)
    assert _mvset(fb) == _cmset(cb)
    for m in list(cb.legal_moves):
        child = fb.apply((m.from_square, m.to_square, m.promotion or 0))
        cb.push(m)
        assert child.to_chess().board_fen() == cb.board_fen()
        assert _mvset(child) == _cmset(cb), (fen, m)
        cb.pop()
