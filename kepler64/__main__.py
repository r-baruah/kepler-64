"""Kepler-64 — runnable demo.

Plays a short engine game, renders a Glass Box frame, and runs the synchronized
candidate-sweep benchmark. Usage:  python -m kepler64
"""

import chess

from . import RocheEngine
from .core.board import Board
from .viz.glassbox import render_position
from .bench.sweep_time import sweep_benchmark


def main():
    engine = RocheEngine()
    print("Kepler-64 Roche Engine ready. Constants:", engine.constants)

    b = chess.Board()
    for ply in range(10):
        if b.is_game_over():
            break
        mv = engine.play(b)
        if mv is None:
            break
        san = b.san(mv)
        b.push(mv)
        print(f"ply {ply}: {san}  ->  {b.fen()}")

    render_position(Board.from_chess(b), engine.constants, "kepler64_frame.png")
    print("Rendered kepler64_frame.png")

    res = sweep_benchmark(engine, repeats=30)
    print("Sweep benchmark:", res)


if __name__ == "__main__":
    main()
