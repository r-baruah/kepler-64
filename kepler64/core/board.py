"""Board state and legal-move generation.

DESIGN (locked): the board is a *swappable* interface. v1 uses a python-chess
adapter so the pure-JAX physics kernel can be validated immediately; the
pure-JAX board (no python-chess in the hot path) is the v1.5 unlock that makes
the end-to-end sub-ms claim true.

chess:  a position is 64 squares of pieces + turn/castling/ep.
physics: the evaluation only needs the (64,) mass vector + the static (64,64)
         distance matrix, so the board's job is to produce that mass vector.
"""

from dataclasses import dataclass, field
import jax.numpy as jnp

try:
    import chess
except ImportError:  # python-chess only required for the v1 adapter / training
    chess = None

from .constants import PIECE_MASSES

# Square index convention: sq = rank*8 + file, rank 0 = rank 1 (white back rank).
SQ_NAMES = None  # populated lazily if chess is available


@dataclass
class Board:
    pieces: object  # (64,) int array of python-chess piece codes (0 = empty)
    turn: int = 0  # 0 = white to move, 1 = black
    castling: int = 0
    ep_square: int = -1
    halfmove: int = 0
    fullmove: int = 1
    _chess: object = field(default=None, repr=False)

    # ---- v1 adapter: build from a real python-chess board -----------------
    @classmethod
    def from_chess(cls, board) -> "Board":
        pieces = jnp.array(
            [board.piece_type_at(i) or 0 for i in range(64)], dtype=jnp.int32
        )
        return cls(
            pieces=pieces,
            turn=0 if board.turn else 1,
            castling=board.castling_xfen(),
            ep_square=board.ep_square or -1,
            halfmove=board.halfmove_clock,
            fullmove=board.fullmove_number,
            _chess=board,
        )

    def to_chess(self):
        return self._chess

    # ---- mass vector (what the physics kernel consumes) -------------------
    def mass_vector(self) -> "jnp.ndarray":
        """(64,) signed gravitational masses. Empty squares -> 0.

        White pieces are +mass, black pieces -mass. The sign identifies color
        for the kernel (gravity itself uses |mass|, so it stays attractive).
        """
        import chess as _chess

        masses = jnp.zeros(64, dtype=jnp.float32)
        for sq in range(64):
            pt = int(self.pieces[sq])
            if pt == 0:
                continue
            base = PIECE_MASSES[_chess.PIECE_SYMBOLS[pt].upper()]
            sign = 1.0 if self._color_at(sq) == 0 else -1.0
            masses = masses.at[sq].set(sign * base)
        return masses

    def _color_at(self, sq) -> int:
        if self._chess is not None:
            pc = self._chess.piece_at(sq)
            return 0 if pc and pc.color == __import__("chess").WHITE else 1
        return 0 if sq < 16 else 1

    # ---- legal moves -------------------------------------------------------
    def legal_moves(self):
        """v1: delegate to python-chess. v1.5: pure-JAX generator."""
        if self._chess is not None:
            return list(self._chess.legal_moves)
        raise NotImplementedError("pure-JAX legal move generation (v1.5)")

    def apply_move(self, move) -> "Board":
        """v1: apply on the python-chess board, re-wrap."""
        if self._chess is not None:
            nb = self._chess.copy()
            nb.push(move)
            return Board.from_chess(nb)
        raise NotImplementedError("pure-JAX move application (v1.5)")

    # ---- game state --------------------------------------------------------
    def is_game_over(self) -> bool:
        return self._chess is not None and self._chess.is_game_over()

    def is_checkmate(self) -> bool:
        return self._chess is not None and self._chess.is_checkmate()

    def is_capture(self, move) -> bool:
        return self._chess is not None and self._chess.is_capture(move)
