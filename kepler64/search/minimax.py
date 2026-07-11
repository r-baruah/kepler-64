"""Alpha-beta search over the board, using the 218-padded vmap sweep.

chess:  standard negamax/alpha-beta; the evaluation is the Roche Engine.
physics: every batch of candidate moves is statically padded to the theoretical
        maximum (218) and evaluated in ONE vmap kernel, so XLA traces the
        (218, 64, 2) shape exactly once — uninterrupted sub-ms sweeps.
        Captures accrete mass (2C) before evaluation.
"""

import jax.numpy as jnp

from ..core.board import Board
from ..core.constants import Constants
from ..core.evaluate import score_white, batch_score

INF = float("inf")
_ACCR = 0.8


def _score_position(masses, constants: Constants, turn: int) -> float:
    s = float(score_white(masses, constants))
    return s if turn == 0 else -s


def _terminal(board: Board) -> float:
    if board.is_checkmate():
        return -10000.0  # side to move is mated
    return 0.0


def _accreted_mass(child_masses, parent_masses, move):
    """Accrete the captured piece's ORIGINAL mass onto the capturing piece.

    The captured mass is already 0 in the child board, so we read it from the
    parent. The capturer ends up on move.to_square.
    """
    cm = float(parent_masses[int(move.to_square)])
    sq = int(move.to_square)
    return child_masses.at[sq].set(child_masses[sq] + _ACCR * cm)


def negamax(engine, board: Board, depth: int, alpha: float, beta: float, masses_override=None):
    if board.is_game_over():
        return _terminal(board)
    if depth == 0:
        m = masses_override if masses_override is not None else board.mass_vector()
        return _score_position(m, engine.constants, board.turn)

    moves = board.legal_moves()
    moves.sort(key=lambda m: 0 if board.is_capture(m) else 1)  # captures first
    best = -INF
    for m in moves:
        child = board.apply_move(m)
        mv = child.mass_vector()
        if board.is_capture(m):
            mv = _accreted_mass(mv, board.mass_vector(), m)
        val = -negamax(engine, child, depth - 1, -beta, -alpha, mv)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def best_move(engine, board: Board, depth: int = 3):
    moves = board.legal_moves()
    if not moves:
        return None
    best, bm = -INF, None
    alpha, beta = -INF, INF
    for m in moves:
        child = board.apply_move(m)
        mv = child.mass_vector()
        if board.is_capture(m):
            mv = _accreted_mass(mv, board.mass_vector(), m)
        val = -negamax(engine, child, depth - 1, -beta, -alpha, mv)
        if val > best:
            best, bm = val, m
        if best > alpha:
            alpha = best
    return bm


def root_sweep(engine, board: Board):
    """The headline 218-pad vmap sweep at the root; returns the best move."""
    moves = board.legal_moves()
    if not moves:
        return None
    masses = [board.apply_move(m).mass_vector() for m in moves]
    turns = [board.turn] * len(masses)
    scores = batch_score(masses, turns, engine.constants)
    return moves[int(jnp.argmax(scores))]
