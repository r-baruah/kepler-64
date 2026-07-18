"""Alpha-beta search over the board, using the pure-JAX-friendly FastBoard.

chess:  standard negamax/alpha-beta; the evaluation is the Roche Engine.
physics: every batch of candidate moves is statically padded to the theoretical
         maximum (218) and evaluated in ONE vmap kernel, so XLA traces the
         (218, 64, 2) shape exactly once — uninterrupted sub-ms sweeps.
         Captures accrete mass (2C) before evaluation. python-chess is used only
         to convert the public input/output (RocheEngine.play).
"""

import jax.numpy as jnp

from ..core.fastboard import FastBoard
from ..core.constants import Constants
from ..core.evaluate import score_white, batch_score, multiverse_score_white

INF = float("inf")
_MAX_MOVES = 218
_ACCR = 0.8


def _score_position(masses, constants: Constants, turn: int,
                     use_multiverse: bool = False, key=None, K: int = 8,
                     sigma: float = 0.1) -> float:
    if use_multiverse and key is not None:
        s = float(multiverse_score_white(masses, constants, key, K=K, sigma=sigma))
    else:
        s = float(score_white(masses, constants))
    return s if turn == 0 else -s


def _terminal(board: FastBoard) -> float:
    if board.is_checkmate():
        return -10000.0  # side to move is mated
    return 0.0


def _accreted_mass(child_masses, parent_masses, move):
    """Accrete the captured piece's ORIGINAL mass onto the capturing piece.

    The captured mass is already 0 in the child board, so we read it from the
    parent. The capturer ends up on move[1] (to_square).
    """
    sq = int(move[1])
    cm = float(parent_masses[sq])
    return child_masses.at[sq].set(child_masses[sq] + _ACCR * cm)


def negamax(engine, board: FastBoard, depth: int, alpha: float, beta: float, masses_override=None):
    if board.is_game_over():
        return _terminal(board)
    # Leaves: resolve forcing capture lines via the batched quiescence (one vmap
    # per node instead of one score_white per capture). This is what keeps the
    # search fast — the 218-pad sweep runs at each leaf in a single XLA call.
    if depth <= 1:
        return _quiesce(engine, board, alpha, beta, masses_override, qdepth=3)

    parent_mv = board.mass_vector()
    moves = board.legal_moves()
    moves.sort(key=lambda m: 0 if board.is_capture(m) else 1)  # captures first
    best = -INF
    for m in moves:
        child = board.apply(m)
        mv = child.mass_vector()
        if board.is_capture(m):
            mv = _accreted_mass(mv, parent_mv, m)
        val = -negamax(engine, child, depth - 1, -beta, -alpha, mv)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _quiesce(engine, board: FastBoard, alpha: float, beta: float, masses_override=None, qdepth: int = 4):
    """Quiescence search: resolve forcing capture lines so the eval isn't read
    at a volatile horizon (standard, and fits the theme — extend the gravity
    sweep along the forcing lines). All capture children of a node are scored in
    ONE `batch_score` vmap (the 218-pad sweep), so the cost per node is a single
    XLA call rather than one score_white per capture. Capped to avoid blow-ups.
    """
    if board.is_game_over():
        return _terminal(board)
    parent_mv = masses_override if masses_override is not None else board.mass_vector()
    stand = _score_position(parent_mv, engine.constants, board.turn)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    caps = [m for m in board.legal_moves() if board.is_capture(m)]
    if not caps:
        return alpha
    caps.sort(key=lambda m: 0 if board.is_capture(m) else 1)
    caps = caps[:10]

    # Score all capture children in ONE vmap sweep (side-to-move perspective).
    child_m = []
    for m in caps:
        child = board.apply(m)
        mv = child.mass_vector()
        if board.is_capture(m):
            mv = _accreted_mass(mv, parent_mv, m)
        child_m.append(mv)
    scores = batch_score(child_m, [board.turn] * len(child_m), engine.constants, pad=16)
    order = sorted(range(len(caps)), key=lambda i: -float(scores[i]))
    for i in order:
        if qdepth <= 1:
            val = -float(scores[i])
        else:
            child = board.apply(caps[i])
            val = -_quiesce(engine, child, -beta, -alpha, None, qdepth - 1)
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha


def best_move(engine, board: FastBoard, depth: int = 3):
    moves = board.legal_moves()
    if not moves:
        return None
    best, bm = -INF, None
    alpha, beta = -INF, INF
    parent_mv = board.mass_vector()
    for m in moves:
        child = board.apply(m)
        mv = child.mass_vector()
        if board.is_capture(m):
            mv = _accreted_mass(mv, parent_mv, m)
        val = -negamax(engine, child, depth - 1, -beta, -alpha, mv)
        if val > best:
            best, bm = val, m
        if best > alpha:
            alpha = best
    return bm


def root_sweep(engine, board: FastBoard):
    """The headline 218-pad vmap sweep at the root; returns the best move."""
    moves = board.legal_moves()
    if not moves:
        return None
    masses = [board.apply(m).mass_vector() for m in moves]
    turns = [board.turn] * len(masses)
    scores = batch_score(masses, turns, engine.constants)
    return moves[int(jnp.argmax(scores))]
