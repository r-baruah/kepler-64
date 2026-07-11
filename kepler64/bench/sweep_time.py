"""Benchmark: sub-ms evaluation sweep.

Proves the headline claim is reproducible, not asserted. Times the 218-padded
vmap sweep of all legal moves at the root over many random positions and
reports p50/p99 latency in milliseconds.
"""

import random
import time

import chess
import numpy as np

from ..core.board import Board
from ..core.evaluate import batch_score


def _random_boards(n: int = 20, plies: int = 10):
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(plies):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
        out.append(b)
    return out


def sweep_benchmark(engine, positions=None, repeats: int = 50, warmup: int = 3):
    if positions is None:
        positions = _random_boards()
    # warmup (JIT compile)
    for _ in range(warmup):
        b = random.choice(positions)
        board = Board.from_chess(b)
        moves = board.legal_moves()
        masses = [board.apply_move(m).mass_vector() for m in moves]
        batch_score(masses, [board.turn] * len(masses), engine.constants)

    times = []
    for _ in range(repeats):
        b = random.choice(positions)
        board = Board.from_chess(b)
        moves = board.legal_moves()
        masses = [board.apply_move(m).mass_vector() for m in moves]
        turns = [board.turn] * len(masses)
        t0 = time.perf_counter()
        batch_score(masses, turns, engine.constants)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(times.mean()),
        "n_positions": len(positions),
        "repeats": repeats,
    }
