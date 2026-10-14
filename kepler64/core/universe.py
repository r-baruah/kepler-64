"""Persistent universe state for a real game.

`FastBoard.from_chess` rebuilds masses from piece identities alone, so every
fresh root forgets two pieces of history the physics depends on:

  * accretion — mass a piece absorbed on earlier captures;
  * velocity  — the per-piece motion accumulator behind the Lorentz boost.

Inside one search tree both survive (`child_mass_vector` threads them), but a
game driven move-by-move from a `chess.Board` would reset them every ply.
`GameState` carries the board, its velocity and the accreted mass vector
across the real game line, so "a capture makes the taker heavier for the rest
of the game" is literally true in play, matches and replays.
"""

from __future__ import annotations

import chess
import numpy as np

from .fastboard import FastBoard
from .transitions import child_mass_vector


class GameState:
    """The universe of one game: python-chess board + physical state."""

    __slots__ = ("board", "fb", "masses", "c")

    def __init__(self, board: chess.Board | None = None, c: float | None = None):
        from .constants import Constants

        self.board = board.copy() if board is not None else chess.Board()
        self.c = float(c) if c is not None else float(Constants().c)
        self.fb = FastBoard.from_chess(self.board)
        self.masses = np.asarray(self.fb.mass_vector(self.c), dtype=np.float32)

    @classmethod
    def from_moves(cls, moves, start: chess.Board | None = None,
                   c: float | None = None) -> "GameState":
        state = cls(start, c=c)
        for mv in moves:
            state.push(mv)
        return state

    def push(self, move: chess.Move) -> np.ndarray:
        """Play `move` and carry accretion + velocity forward. Returns the
        parent mass vector (the state before the move)."""
        parent = self.masses
        mt = (move.from_square, move.to_square, move.promotion or 0)
        child_fb = self.fb.apply(mt)
        self.masses = np.asarray(
            child_mass_vector(self.fb, mt, parent, child_board=child_fb,
                              c_lorentz=self.c),
            dtype=np.float32)
        self.fb = child_fb
        self.board.push(move)
        return parent

    def excess(self) -> np.ndarray:
        """Per-square mass above the piece's rest mass (accretion + Lorentz),
        signed by colour like the mass vector."""
        rest = np.asarray(FastBoard.from_chess(self.board).mass_vector(self.c),
                          dtype=np.float32)
        return self.masses - rest

    def copy(self) -> "GameState":
        new = object.__new__(GameState)
        new.board = self.board.copy()
        new.c = self.c
        new.fb = FastBoard(self.fb.pieces.copy(), self.fb.turn, self.fb.castling,
                           self.fb.ep, self.fb.velocity.copy())
        new.masses = self.masses.copy()
        return new
