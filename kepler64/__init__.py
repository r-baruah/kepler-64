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

    def __init__(self, constants: Constants | None = None, seed_image=None,
                 load_trained: bool = True):
        # Constants resolution order:
        #   1. an explicitly supplied Constants (tests, ablations, matches)
        #   2. the persisted trained artifact (trained_constants.json) if it
        #      exists — the trained universe the pipeline last accepted
        #   3. the pristine hand-set universe
        if constants is None and load_trained:
            from .core.constants import load_constants
            constants = load_constants() or Constants()
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

    def play(self, board, depth: int = 3, search_time_ms: float | None = None,
             max_depth: int = 8, use_multiverse: bool = True,
             multiverse_seed: int | None = None):
        """Search + return the best move (as a python-chess Move).

        `depth` caps iterative deepening (default 3). Pass `search_time_ms` to
        search with a wall-clock budget instead of a fixed depth: the engine
        deepens until the budget is exhausted and returns the best move of the
        last COMPLETED iteration (never an unfinished deeper search).

        `use_multiverse` (default True) runs the Layer-2 posterior mean on the
        root's near-tie candidates, so the multiverse breaks ties the search
        cannot; it is deterministic (fixed posterior seed). Pass
        `multiverse_seed` to reseed the posterior draws — training uses this
        for exploration noise on self-play games without changing the physics.
        """
        import chess

        from .core.fastboard import FastBoard
        from .search.minimax import best_move, best_move_time, iterative_search

        fb = board if isinstance(board, FastBoard) else FastBoard.from_chess(board)
        mv_seed = 20260808 if multiverse_seed is None else multiverse_seed

        # Closed-orbit history: every position already on the line (board
        # identity only — pieces/turn/castling/ep — NOT the halfmove clocks,
        # which change on every ply and would defeat the comparison). A root
        # move that lands back on a seen board is a cycle with no net
        # momentum: the laws assign it the draw value.
        seen = None
        if isinstance(board, chess.Board) and board.move_stack:
            seen = set()
            b = board.copy()
            seen.add(" ".join(b.fen().split()[:4]))
            for _ in range(len(board.move_stack)):
                b.pop()
                seen.add(" ".join(b.fen().split()[:4]))

        if search_time_ms is not None:
            mv, _ = iterative_search(self, fb, max_depth=max_depth,
                                     time_ms=search_time_ms, seen=seen,
                                     use_multiverse=use_multiverse,
                                     multiverse_seed=mv_seed)
        else:
            mv, _ = iterative_search(self, fb, max_depth=depth, seen=seen,
                                     use_multiverse=use_multiverse,
                                     multiverse_seed=mv_seed)
        if mv is None:
            return None
        f, t, promo = mv
        return chess.Move(f, t, chess.PieceType(promo) if promo else None)
