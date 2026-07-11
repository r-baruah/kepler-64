"""Tests that the pure-JAX board agrees with python-chess.

This is the proof that removing python-chess from the hot path does not change
the game. Run after the v1.5 pure-JAX board lands.
"""

import pytest

pytest.importorskip("chess")

from kepler64.core.board import Board


def test_from_chess_roundtrip():
    import chess

    b = chess.Board()
    kb = Board.from_chess(b)
    assert kb.turn == 0
    assert int(kb.pieces[0]) == chess.ROOK  # a1 rook


def test_legal_moves_match():
    import chess

    b = chess.Board()
    kb = Board.from_chess(b)
    assert len(kb.legal_moves()) == len(list(b.legal_moves))
