"""Quasi-equilibrium opening book.

chess:  bias early play toward structurally sound positions.
physics: precomputed N-body equilibrium states where the net force on key
        coordinates is near zero (low-energy foundations). v1: best_move falls
        back to search; book lookup is a passthrough hook for v2.

Quasi-equilibrium opening book: not yet implemented (future work). `load_book`
is a documented stub that returns None; callers must handle a None book
gracefully (no crash, fall back to search).
"""

import pathlib

from ..core.board import Board


def load_book(path: str = str(pathlib.Path(__file__).parent / "openings.npz")):
    """Quasi-equilibrium opening book: not yet implemented (future work).

    Documented stub — returns None. Callers must treat a None book as "no book
    available" and fall back to search without crashing.
    """
    return None


def pick(board: Board, book=None):
    """Return a book move if known, else None (caller falls back to search).

    Gracefully handles book=None (stub) by returning None immediately.
    """
    if book is None:
        return None
    return None
