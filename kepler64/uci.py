"""Kepler-64 as a UCI engine — play the universe in any chess GUI.

chess:  speaks the UCI protocol on stdin/stdout (uci/isready/position/go/quit)
        so Arena, Cutechess, lichess-bot, or Stockfish-test rigs can use
        Kepler-64 as a normal engine.
physics: every move is still pure Roche physics (FastBoard -> gravity -> tidal
        eta -> gauges -> deltas -> negamax). UCI is a wire, not a law.

Scores are Kepler units, NOT centipawns (PRODUCT.md constraint). The `info
score cp` field carries int(score * 100) so GUIs can plot it; treat it as a
native-unit readout, not a Stockfish-scale evaluation.

Usage:
    python -m kepler64.uci            # stdio UCI loop
    kepler64-uci                      # same, via console script
"""

from __future__ import annotations

import sys

import chess

from . import RocheEngine
from .core.constants import Constants
from .core.fastboard import FastBoard
from .search.minimax import iterative_search

ENGINE_NAME = "Kepler-64"
ENGINE_AUTHOR = "Kepler-64 contributors"
DEFAULT_DEPTH = 3
MAX_DEPTH = 8
_MOVE_OVERHEAD_MS = 20


def parse_position(tokens: list[str], board: chess.Board) -> chess.Board:
    """Apply a tokenized `position ...` line. Returns the new board."""
    if not tokens or tokens[0] != "position":
        return board
    args = tokens[1:]
    if args and args[0] == "startpos":
        board = chess.Board()
        args = args[1:]
    elif args and args[0] == "fen":
        # fen <6 fields> [moves ...]
        if "moves" in args:
            mi = args.index("moves")
            board = chess.Board(" ".join(args[1:mi]))
            args = args[mi:]
        else:
            return chess.Board(" ".join(args[1:7]))
    if args and args[0] == "moves":
        for m in args[1:]:
            try:
                board.push_uci(m)
            except ValueError:
                break  # ponytail: stop at first illegal token, keep prior moves
    return board


def go_budget(tokens: list[str], turn_white: bool) -> tuple[int, float | None]:
    """Parse a tokenized `go ...` line -> (max_depth, time_ms|None)."""
    d = dict(zip(tokens[1::2], tokens[2::2])) if len(tokens) > 1 else {}
    if "depth" in d:
        return max(1, min(MAX_DEPTH, int(d["depth"]))), None
    if "movetime" in d:
        return MAX_DEPTH, max(10.0, float(d["movetime"]) - _MOVE_OVERHEAD_MS)
    if "wtime" in d or "btime" in d:
        side, inc = ("wtime", "winc") if turn_white else ("btime", "binc")
        left = float(d.get(side, 1000.0))
        bonus = float(d.get(inc, 0.0))
        mtg = float(d.get("movestogo", 30))
        return MAX_DEPTH, max(10.0, min(left / max(1.0, mtg) + bonus * 0.5, left / 4.0))
    if "infinite" in tokens:
        return MAX_DEPTH, None
    return DEFAULT_DEPTH, None


def search(board: chess.Board, engine: RocheEngine,
           max_depth: int, time_ms: float | None) -> tuple[str, int]:
    """Search and return (bestmove_uci, info_cp). `info_cp` is native*100."""
    fb = FastBoard.from_chess(board)
    seen = {" ".join(board.fen().split()[:4])} if board.move_stack else None
    mv, score = iterative_search(engine, fb, max_depth=max_depth,
                                 time_ms=time_ms, seen=seen,
                                 use_multiverse=True)
    if mv is None:
        return "0000", 0
    f, t, promo = mv
    uci = chess.Move(f, t, chess.PieceType(promo) if promo else None).uci()
    return uci, int(float(score if score is not None else 0.0) * 100)


def main(argv: list[str] | None = None) -> int:
    engine = RocheEngine()  # loads trained_constants.json when present, else pristine
    board = chess.Board()
    max_depth = DEFAULT_DEPTH
    out = sys.stdout
    for raw in sys.stdin:
        tokens = raw.strip().split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            out.write(f"id name {ENGINE_NAME}\nid author {ENGINE_AUTHOR}\n")
            out.write("option name Depth type spin default 3 min 1 max 8\n")
            out.write("option name Multiverse type check default true\nuciok\n")
        elif cmd == "isready":
            out.write("readyok\n")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "setoption":
            # setoption name <id> value <x> — only Depth honored.
            try:
                ni = tokens.index("name")
                name = tokens[ni + 1]
                val = tokens[tokens.index("value") + 1] if "value" in tokens else ""
                if name.lower() == "depth":
                    max_depth = max(1, min(MAX_DEPTH, int(val)))
            except (ValueError, IndexError):
                pass
        elif cmd == "position":
            board = parse_position(tokens, board)
        elif cmd == "go":
            depth, budget = go_budget(tokens, board.turn == chess.WHITE)
            depth = min(depth, max_depth) if "depth" in tokens else max_depth
            uci, cp = search(board, engine, depth, budget)
            out.write(f"info depth {depth} score cp {cp} pv {uci}\nbestmove {uci}\n")
        elif cmd == "stop":
            continue  # synchronous search: nothing to interrupt
        elif cmd == "quit":
            break
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
