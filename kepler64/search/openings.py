"""Quasi-equilibrium opening book.

chess:  bias early play toward structurally sound positions.
physics: precomputed N-body equilibrium states where the net force on key
        coordinates is near zero (low-energy foundations). v1: best_move falls
        back to search; book lookup is a passthrough hook for v2.
"""

from ..core.board import Board


def load_book(path: str = "openings.npz"):
    """Return precomputed low-energy opening positions. (TODO: v2 train)"""
    return None


def pick(board: Board, book=None):
    """Return a book move if known, else None (caller falls back to search)."""
    return None
