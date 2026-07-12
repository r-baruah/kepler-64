"""Pure-NumPy board + legal move generator — removes python-chess from the
search hot path (the v1 "tax").

chess:  full legal move generation (pseudo-legal + king-safety filter), castling,
        en passant, promotion. Verified against python-chess on random positions.
physics: board state is a (64,) int8 array with SIGN = colour, so mass_vector()
        stays a one-line lookup. Move application is a cheap array copy, not a
        python-chess Board clone. The differentiable gravity kernel is untouched.

State encoding: pieces[sq] in {-6..-1, 0, 1..6}; + = white, - = black,
abs = type (1=P,2=N,3=B,4=R,5=Q,6=K). Square index: a1=0 .. h8=63 (matches
python-chess).
"""

import numpy as np

import jax.numpy as jnp

from .constants import PIECE_MASSES

# Direction tables (dx, dy).
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_ROOK = (0, 1, 2, 3)
_BISHOP = (4, 5, 6, 7)
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# Pawn capture diagonals by colour: white attacks up-forward, black down-forward.
_PAWN_CAP = {True: [(1, 1), (-1, 1)], False: [(1, -1), (-1, -1)]}

# Precomputed rays: (sq, dir_idx) -> list of squares from sq outward (excl. sq).
_RAYS = {}
for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    for _di, (_dx, _dy) in enumerate(_DIRS):
        _lst = []
        _nf, _nr = _f + _dx, _r + _dy
        while 0 <= _nf < 8 and 0 <= _nr < 8:
            _lst.append(_nr * 8 + _nf)
            _nf += _dx
            _nr += _dy
        _RAYS[(_sq, _di)] = _lst

# Castling rights bitmask: WK=1, WQ=2, BK=4, BQ=8.
_WK, _WQ, _BK, _BQ = 1, 2, 4, 8

# Mass LUT indexed by abs(piece type): 0=empty,1=P,2=N,3=B,4=R,5=Q,6=K.
_MASS_LUT = np.array([0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 1000.0], dtype=np.float32)


class FastBoard:
    __slots__ = ("pieces", "turn", "castling", "ep")

    def __init__(self, pieces=None, turn=0, castling=0, ep=-1):
        self.pieces = pieces if pieces is not None else np.zeros(64, dtype=np.int8)
        self.turn = turn          # 0 = white to move, 1 = black
        self.castling = castling  # bitmask
        self.ep = ep              # en-passant target square, -1 if none

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_chess(cls, board):
        import chess

        pieces = np.zeros(64, dtype=np.int8)
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None:
                continue
            t = pc.piece_type  # 1..6
            pieces[sq] = t if pc.color == chess.WHITE else -t
        rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights |= _WK
        if board.has_queenside_castling_rights(chess.WHITE):
            rights |= _WQ
        if board.has_kingside_castling_rights(chess.BLACK):
            rights |= _BK
        if board.has_queenside_castling_rights(chess.BLACK):
            rights |= _BQ
        ep = board.ep_square if board.ep_square is not None else -1
        return cls(pieces, 0 if board.turn == chess.WHITE else 1, rights, ep)

    def to_chess(self):
        import chess

        b = chess.Board(None)
        b.clear()
        for sq in range(64):
            p = int(self.pieces[sq])
            if p == 0:
                continue
            color = chess.WHITE if p > 0 else chess.BLACK
            b.set_piece_at(sq, chess.Piece(abs(p), color))
        b.turn = chess.WHITE if self.turn == 0 else chess.BLACK
        if self.castling & _WK:
            b.castling_rights |= chess.BB_H1
        if self.castling & _WQ:
            b.castling_rights |= chess.BB_A1
        if self.castling & _BK:
            b.castling_rights |= chess.BB_H8
        if self.castling & _BQ:
            b.castling_rights |= chess.BB_A8
        b.ep_square = self.ep if self.ep >= 0 else None
        b.clear_stack()
        return b

    # ---- queries ---------------------------------------------------------
    def king_sq(self, white: bool) -> int:
        target = 6 if white else -6
        idx = np.nonzero(self.pieces == target)[0]
        return int(idx[0]) if idx.size else -1

    def square_attacked(self, sq: int, by_white: bool) -> bool:
        """Is `sq` attacked by side `by_white`? Pure ray-walk, no allocation."""
        board = self.pieces
        f, r = sq % 8, sq // 8
        # Pawn attacks.
        if by_white:
            for df in (-1, 1):
                nf, nr = f + df, r - 1
                if 0 <= nf < 8 and 0 <= nr < 8 and board[nr * 8 + nf] == 1:
                    return True
        else:
            for df in (-1, 1):
                nf, nr = f + df, r + 1
                if 0 <= nf < 8 and 0 <= nr < 8 and board[nr * 8 + nf] == -1:
                    return True
        # Knight.
        for dx, dy in _KNIGHT:
            nf, nr = f + dx, r + dy
            if 0 <= nf < 8 and 0 <= nr < 8:
                p = board[nr * 8 + nf]
                if p != 0 and (p > 0) == by_white and abs(p) == 2:
                    return True
        # King.
        for dx, dy in _KING:
            nf, nr = f + dx, r + dy
            if 0 <= nf < 8 and 0 <= nr < 8:
                p = board[nr * 8 + nf]
                if p != 0 and (p > 0) == by_white and abs(p) == 6:
                    return True
        # Sliding orth (R/Q).
        for di in _ROOK:
            for t in _RAYS[(sq, di)]:
                p = board[t]
                if p != 0:
                    if (p > 0) == by_white and abs(p) in (4, 5):
                        return True
                    break
        # Sliding diag (B/Q).
        for di in _BISHOP:
            for t in _RAYS[(sq, di)]:
                p = board[t]
                if p != 0:
                    if (p > 0) == by_white and abs(p) in (3, 5):
                        return True
                    break
        return False

    def in_check(self, white: bool) -> bool:
        return self.square_attacked(self.king_sq(white), not white)

    # ---- move generation -------------------------------------------------
    def _gen_pseudo(self):
        board = self.pieces
        white = self.turn == 0
        moves = []
        for sq in range(64):
            p = int(board[sq])
            if p == 0 or (p > 0) != white:
                continue
            pt = abs(p)
            f, r = sq % 8, sq // 8
            if pt == 1:
                self._pawn_moves(sq, f, r, white, moves)
            elif pt == 2:
                self._step_moves(sq, f, r, white, _KNIGHT, moves)
            elif pt == 3:
                self._slide_moves(sq, white, _BISHOP, moves)
            elif pt == 4:
                self._slide_moves(sq, white, _ROOK, moves)
            elif pt == 5:
                self._slide_moves(sq, white, (0, 1, 2, 3, 4, 5, 6, 7), moves)
            elif pt == 6:
                self._step_moves(sq, f, r, white, _KING, moves)
                self._castle_moves(sq, white, moves)
        return moves

    def _step_moves(self, sq, f, r, white, offsets, moves):
        board = self.pieces
        for dx, dy in offsets:
            nf, nr = f + dx, r + dy
            if 0 <= nf < 8 and 0 <= nr < 8:
                t = nr * 8 + nf
                tp = int(board[t])
                if tp == 0 or (tp > 0) != white:
                    moves.append((sq, t, 0))

    def _slide_moves(self, sq, white, dirs, moves):
        board = self.pieces
        for di in dirs:
            for t in _RAYS[(sq, di)]:
                tp = int(board[t])
                if tp == 0:
                    moves.append((sq, t, 0))
                else:
                    if (tp > 0) != white:
                        moves.append((sq, t, 0))
                    break

    def _pawn_moves(self, sq, f, r, white, moves):
        board = self.pieces
        fwd = 8 if white else -8
        start = (r == 1) if white else (r == 6)
        promo_rank = 7 if white else 0
        # Single push.
        t1 = sq + fwd
        if 0 <= t1 < 64 and int(board[t1]) == 0:
            if (t1 // 8) == promo_rank:
                for pr in (2, 3, 4, 5):
                    moves.append((sq, t1, pr))
            else:
                moves.append((sq, t1, 0))
                if start:
                    t2 = sq + 2 * fwd
                    if int(board[t2]) == 0:
                        moves.append((sq, t2, 0))
        # Captures + en passant.
        for dx, dy in _PAWN_CAP[white]:
            nf, nr = f + dx, r + dy
            if 0 <= nf < 8 and 0 <= nr < 8:
                t = nr * 8 + nf
                tp = int(board[t])
                if tp != 0 and (tp > 0) != white:
                    if (t // 8) == promo_rank:
                        for pr in (2, 3, 4, 5):
                            moves.append((sq, t, pr))
                    else:
                        moves.append((sq, t, 0))
                elif t == self.ep:
                    moves.append((sq, t, 0))

    def _castle_moves(self, sq, white, moves):
        board = self.pieces
        if white:
            if self.castling & _WK and int(board[5]) == 0 and int(board[6]) == 0 \
                    and int(board[7]) == 4 \
                    and not self.square_attacked(4, False) \
                    and not self.square_attacked(5, False) \
                    and not self.square_attacked(6, False):
                moves.append((4, 6, 0))
            if self.castling & _WQ and int(board[1]) == 0 and int(board[2]) == 0 \
                    and int(board[3]) == 0 and int(board[0]) == 4 \
                    and not self.square_attacked(4, False) \
                    and not self.square_attacked(3, False) \
                    and not self.square_attacked(2, False):
                moves.append((4, 2, 0))
        else:
            if self.castling & _BK and int(board[61]) == 0 and int(board[62]) == 0 \
                    and int(board[63]) == -4 \
                    and not self.square_attacked(60, True) \
                    and not self.square_attacked(61, True) \
                    and not self.square_attacked(62, True):
                moves.append((60, 62, 0))
            if self.castling & _BQ and int(board[57]) == 0 and int(board[58]) == 0 \
                    and int(board[59]) == 0 and int(board[56]) == -4 \
                    and not self.square_attacked(60, True) \
                    and not self.square_attacked(59, True) \
                    and not self.square_attacked(58, True):
                moves.append((60, 58, 0))

    def legal_moves(self):
        white = self.turn == 0
        legal = []
        for mv in self._gen_pseudo():
            nb = self.apply(mv)
            k = nb.king_sq(white)
            if k >= 0 and not nb.square_attacked(k, not white):
                legal.append(mv)
        return legal

    # ---- application -----------------------------------------------------
    def apply(self, move):
        f, t, promo = move
        nb = FastBoard(self.pieces.copy(), self.turn, self.castling, -1)
        board = nb.pieces
        p = int(board[f])
        white = p > 0
        pt = abs(p)

        # En-passant capture (pawn diagonal onto empty ep square).
        if pt == 1 and t == self.ep and int(board[t]) == 0:
            cap = t - (8 if white else -8)
            board[cap] = 0

        # Move / promote the piece.
        board[t] = promo if promo else p
        board[f] = 0

        # Castling rook.
        if pt == 6 and abs(t - f) == 2:
            if t == 6:       # white kingside
                board[5] = board[7]; board[7] = 0
            elif t == 2:     # white queenside
                board[3] = board[0]; board[0] = 0
            elif t == 62:    # black kingside
                board[61] = board[63]; board[63] = 0
            elif t == 58:    # black queenside
                board[59] = board[56]; board[56] = 0

        # En-passant target update.
        if pt == 1 and abs(t - f) == 16:
            nb.ep = (f + t) // 2
        else:
            nb.ep = -1

        # Castling-rights invalidation.
        rights = self.castling
        if f == 4:
            rights &= ~(_WK | _WQ)
        elif f == 60:
            rights &= ~(_BK | _BQ)
        if t == 0 or f == 0:
            rights &= ~_WQ
        if t == 7 or f == 7:
            rights &= ~_WK
        if t == 56 or f == 56:
            rights &= ~_BQ
        if t == 63 or f == 63:
            rights &= ~_BK
        nb.castling = rights

        nb.turn = 1 - self.turn
        return nb

    # ---- physics interface ----------------------------------------------
    def mass_vector(self) -> "jnp.ndarray":
        """(64,) signed gravitational masses: white +, black -. King = 1000.

        Fully vectorized LUT lookup (no per-square Python loop)."""
        lut = _MASS_LUT  # (7,) -> mass by abs piece type
        m = np.sign(self.pieces) * lut[np.abs(self.pieces.astype(np.int8))]
        return jnp.asarray(m)

    def is_game_over(self) -> bool:
        return len(self.legal_moves()) == 0

    def is_checkmate(self) -> bool:
        return self.is_game_over() and self.in_check(self.turn == 0)

    def is_capture(self, move) -> bool:
        f, t, _ = move
        if int(self.pieces[t]) != 0:
            return True
        # en passant
        return abs(self.pieces[f]) == 1 and t == self.ep
