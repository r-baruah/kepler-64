"""Kepler-64 — the Roche Engine.

A chess engine whose evaluation is a differentiable N-body gravitational
potential. It does not understand chess; it understands orbital mechanics and
structural collapse.

  chess:  a position is evaluated by physics, not by heuristics.
  physics: pieces are masses; the enemy King "collapses" past its Roche/tidal
           limit when the tidal Jacobian at its coordinate exceeds a learned
           binding threshold.
"""

from .core.constants import Constants
from .core.board import Board
from .core.evaluate import evaluate

__all__ = ["RocheEngine", "Constants", "Board", "evaluate"]


class RocheEngine:
    """Instantiate a universe.

    Holds the (learnable) physical constants of this chess universe and exposes
    the evaluation interface used by the search tree.
    """

    def __init__(self, constants: Constants | None = None, seed_image=None):
        self.constants = constants or Constants()
        if seed_image is not None:
            from .core.image_seed import seed_from_image

            self.constants = seed_from_image(seed_image, self.constants)

    def evaluate(self, board: Board) -> float:
        """Score a position from the perspective of the side to move.

        Positive = good for the side to move; negative = its King is being
        tidally disrupted.
        """
        return evaluate(board, self.constants)

    def play(self, board, depth: int = 3):
        """Search + return the best move (as a python-chess Move)."""
        import chess

        from .core.fastboard import FastBoard
        from .search.minimax import best_move

        fb = board if isinstance(board, FastBoard) else FastBoard.from_chess(board)
        mv = best_move(self, fb, depth)
        if mv is None:
            return None
        f, t, promo = mv
        return chess.Move(f, t, chess.PieceType(promo) if promo else None)
