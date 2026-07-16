"""Build supervised training examples from REAL chess data.

Sources (no invented heuristics - these are real games / puzzle solutions):
  * Lichess puzzles CSV : FEN + Moves = [opponent_setup, solution_1, ...].
    The position to solve is FEN-after-opponent_setup; the expert move is
    solution_1.  Every puzzle is a forced win for the side to move, so the
    outcome label is +1 for that side.
  * PGN games          : behavioural cloning.  At every ply the played move is
    the expert move; the outcome label is the final game result (White view).

Each example carries the position's mass vector, its outcome, the side to
move, and the mass vectors of every legal child move (padded) so the policy
term can rank the expert move above the alternatives through the gravity
kernel.  Captures accrete mass exactly like the search does.
"""

from __future__ import annotations

import chess
import numpy as np

from ..core.fastboard import FastBoard, _MASS_LUT

_MAX_MOVES = 64   # pad children to this (covers >99.9% of real positions)
_ACCR = 0.8       # accretion fraction (matches search/minimax.py)

# Promotion piece-type map: chess.Move.promotion uses python-chess piece types
# (1=P, 2=N, 3=B, 4=R, 5=Q, 6=K); FastBoard uses the same encoding but 0=none.

def _move_to_tuple(m) -> tuple:
    """Convert a chess.Move to a FastBoard tuple (from_sq, to_sq, promo)."""
    return (m.from_square, m.to_square, m.promotion if m.promotion else 0)


def _mass_np(pieces: np.ndarray) -> np.ndarray:
    """Pure-numpy mass vector — avoids JAX device round-trips in the data loop."""
    return np.sign(pieces).astype(np.float32) * _MASS_LUT[np.abs(pieces.astype(np.int8))]


def _accreted(child_masses: np.ndarray, parent_masses: np.ndarray, move) -> np.ndarray:
    """Apply accretion to a capture: capturing piece absorbs _ACCR fraction of captured mass."""
    to_sq = int(move[1])
    captured_mass = float(parent_masses[to_sq])  # mass of the piece being captured
    child_masses[to_sq] += _ACCR * captured_mass
    return child_masses


def _children(chess_board):
    """Return (pos_mass, child_masses (K,64), mask (K,), legal_moves).

    Uses pure-numpy mass_vector to avoid JAX device round-trips in the hot loop.
    """
    fb = FastBoard.from_chess(chess_board)
    parent = _mass_np(fb.pieces)          # (64,) float32, no JAX
    legal = list(fb.legal_moves())
    child_m = np.zeros((_MAX_MOVES, 64), dtype=np.float32)
    mask    = np.zeros((_MAX_MOVES,),    dtype=np.float32)
    for i, m in enumerate(legal):
        if i >= _MAX_MOVES:
            break
        child = fb.apply(m)
        mv = _mass_np(child.pieces)        # pure numpy — fast
        if fb.is_capture(m):
            mv = _accreted(mv, parent, m)
        child_m[i] = mv
        mask[i]    = 1.0
    return parent, child_m, mask, legal


def puzzle_examples(csv_path, limit=5000):
    """Yield training examples from a Lichess puzzles CSV (utf-8).

    Lichess CSV format:
      FEN   - position BEFORE the opponent's last move
      Moves - 'opponent_move  solver_move1  solver_move2 ...'

    We push parts[0] to reach the puzzle position, then parts[1] is the
    expert (solver) move.  Every puzzle is a forced win for the side to move
    after pushing parts[0], so the outcome label is +1 for that side.
    """
    import csv

    out = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if len(out) >= limit:
                break
            fen = row.get("FEN")
            moves_str = row.get("Moves")
            if not fen or not moves_str:
                continue
            parts = moves_str.split()
            if len(parts) < 2:          # need at least opponent_move + solution
                continue
            try:
                board = chess.Board(fen)
                setup  = chess.Move.from_uci(parts[0])   # opponent's last move
                expert = chess.Move.from_uci(parts[1])   # solver's move (what we train)
                if not board.is_legal(setup):
                    continue
                board.push(setup)                         # now it's the solver's turn
                if not board.is_legal(expert):
                    continue
            except Exception:
                continue
            solver = board.turn
            parent, child_m, mask, legal = _children(board)
            expert_tup = _move_to_tuple(expert)
            try:
                expert_idx = legal.index(expert_tup)
            except ValueError:
                continue
            if expert_idx >= _MAX_MOVES:
                continue
            out.append({
                "mass": parent,
                "outcome": 1.0 if solver == chess.WHITE else -1.0,
                "turn": 0.0 if solver == chess.WHITE else 1.0,
                "child_m": child_m,
                "mask": mask,
                "expert_idx": int(expert_idx),
            })
    return out


def game_examples(pgn_path, limit_games=10, limit_positions=20000):
    """Yield behavioural-cloning examples from a PGN (all plies of each game)."""
    import chess.pgn

    out = []
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while len(out) < limit_positions:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            res = game.headers.get("Result", "*")
            if res == "1-0":
                outcome = 1.0
            elif res == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0
            board = game.board()
            for mv in game.mainline_moves():
                if not board.is_legal(mv):
                    break
                parent, child_m, mask, legal = _children(board)
                mv_tup = _move_to_tuple(mv)
                try:
                    expert_idx = legal.index(mv_tup)
                except ValueError:
                    break
                if expert_idx >= _MAX_MOVES:
                    break
                out.append({
                    "mass": parent,
                    "outcome": outcome,
                    "turn": 0.0 if board.turn == chess.WHITE else 1.0,
                    "child_m": child_m,
                    "mask": mask,
                    "expert_idx": int(expert_idx),
                })
                board.push(mv)
            limit_games -= 1
            if limit_games <= 0:
                break
    return out


def to_arrays(examples):
    """Stack examples into padded numpy arrays."""
    n = len(examples)
    M = np.stack([e["mass"] for e in examples]).astype(np.float32)
    Y = np.array([e["outcome"] for e in examples], dtype=np.float32)
    turns = np.array([e["turn"] for e in examples], dtype=np.float32)
    moves_m = np.stack([e["child_m"] for e in examples]).astype(np.float32)
    mask = np.stack([e["mask"] for e in examples]).astype(np.float32)
    expert_idx = np.array([e["expert_idx"] for e in examples], dtype=np.int32)
    return M, Y, turns, moves_m, mask, expert_idx
