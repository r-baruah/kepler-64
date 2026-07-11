"""Offline data pipeline — python-chess used ONLY here.

chess:  turn games into (position, outcome) samples for training.
physics: the tensors feed the differentiable gravity kernel so gradient descent
        can learn G, eps, c, roche from real (or self-play) games.
"""

import jax.numpy as jnp

from ..core.board import Board


def self_play_positions(engine, n: int = 200, depth: int = 1, seed: int = 0):
    """Collect (mass_vector, outcome) samples by letting the engine play itself.

    outcome in {+1 win for white, -1 win for black, 0 draw} from White's view.
    Simplified stand-in for the Lichess pipeline; swap in load_lichess() later.
    """
    import random

    import chess

    random.seed(seed)
    out = []
    for _ in range(n):
        b = chess.Board()
        count = 0
        while not b.is_game_over() and count < 100:
            move = engine.play(Board.from_chess(b), depth=depth) if hasattr(engine, "play") else None
            if move is None:
                move = random.choice(list(b.legal_moves))
            b.push(move)
            count += 1
        oc = b.outcome()
        if oc is None:
            outcome = 0.0
        elif oc.winner is None:
            outcome = 0.0
        else:
            outcome = 1.0 if oc.winner == chess.WHITE else -1.0
        out.append((Board.from_chess(b).mass_vector(), outcome))
    return out


def to_tensor(board):
    return board.mass_vector()
