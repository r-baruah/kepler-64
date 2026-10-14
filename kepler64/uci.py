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
from .core.universe import GameState
from .search.minimax import (INF, MATE, _MATE_BOUND, iterative_search,
                             seen_positions)

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


_GO_VALUED = {"depth", "movetime", "wtime", "btime", "winc",
              "binc", "movestogo", "nodes", "mate"}
_GO_FLAGS = {"ponder", "infinite"}


def parse_go(tokens: list[str]) -> tuple[dict[str, float], set[str]]:
    """Parse `go ...` into ({name: number}, {flags}). Valueless tokens (ponder,
    infinite) are flags; every other known token consumes the next number.
    Unknown tokens (e.g. searchmoves' move list) are skipped."""
    vals: dict[str, float] = {}
    flags: set[str] = set()
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in _GO_FLAGS:
            flags.add(t)
        elif t in _GO_VALUED and i + 1 < len(tokens):
            try:
                vals[t] = float(tokens[i + 1])
            except ValueError:
                pass
            i += 1
        i += 1
    return vals, flags


def go_budget(tokens: list[str], turn_white: bool,
              default_depth: int = DEFAULT_DEPTH) -> tuple[int, float | None]:
    """Parse a tokenized `go ...` line -> (max_depth, time_ms|None).

    A time budget (movetime / clocks) deepens up to MAX_DEPTH: the clock, not
    the Depth option, bounds the search. `default_depth` is used for a bare
    `go`."""
    d, flags = parse_go(tokens)
    if "depth" in d:
        return max(1, min(MAX_DEPTH, int(d["depth"]))), None
    if "movetime" in d:
        return MAX_DEPTH, max(10.0, d["movetime"] - _MOVE_OVERHEAD_MS)
    if "wtime" in d or "btime" in d:
        side, inc = ("wtime", "winc") if turn_white else ("btime", "binc")
        left = d.get(side, 1000.0)
        bonus = d.get(inc, 0.0)
        mtg = d.get("movestogo", 30.0)
        return MAX_DEPTH, max(10.0, min(left / max(1.0, mtg) + bonus * 0.5, left / 4.0))
    if "infinite" in flags or "ponder" in flags:
        return MAX_DEPTH, None
    return default_depth, None


def score_to_uci(score: float | None) -> str:
    """Native score -> UCI `score ...` fields. Mates (|s| >= MATE bound) become
    `mate N` (moves, negative when we are mated); infinities are clamped. Other
    scores are Kepler units * 100 as `cp` (native readout, not centipawns)."""
    if score is None or score != score:
        return "cp 0"
    if abs(score) >= _MATE_BOUND and abs(score) != INF:
        plies = max(1, int(round(MATE - abs(score))))
        n = (plies + 1) // 2
        return f"mate {n if score > 0 else -n}"
    if abs(score) == INF:
        score = (MATE if score > 0 else -MATE)
    return f"cp {int(max(-MATE, min(MATE, score)) * 100)}"


def search_full(board: chess.Board, engine: RocheEngine, max_depth: int,
                time_ms: float | None, use_multiverse: bool = True
                ) -> tuple[str, float | None, int]:
    """Search; return (bestmove_uci, native score, completed depth).

    The game is replayed from the GUI's move list through a GameState, so
    mass absorbed on earlier captures and piece velocity carry into the
    search, and every position already on the line counts for the
    closed-orbit (repetition) rule.
    """
    state = GameState.from_moves(board.move_stack, start=board.root(),
                                 c=float(engine.constants.c))
    stats: dict = {}
    mv, score = iterative_search(engine, state.fb, max_depth=max_depth,
                                 time_ms=time_ms, seen=seen_positions(board),
                                 use_multiverse=use_multiverse,
                                 root_masses=state.masses, stats=stats)
    if mv is None:
        return "0000", 0.0, 0
    f, t, promo = mv
    uci = chess.Move(f, t, chess.PieceType(promo) if promo else None).uci()
    return uci, score, int(stats.get("depth", max_depth))


def search(board: chess.Board, engine: RocheEngine,
           max_depth: int, time_ms: float | None,
           use_multiverse: bool = True) -> tuple[str, int]:
    """Search and return (bestmove_uci, info_cp). `info_cp` is native*100."""
    uci, score, _ = search_full(board, engine, max_depth, time_ms, use_multiverse)
    s = float(score if score is not None else 0.0)
    return uci, int(max(-MATE, min(MATE, s)) * 100)


def main(argv: list[str] | None = None) -> int:
    engine = RocheEngine()  # loads trained_constants.json when present, else pristine
    board = chess.Board()
    max_depth = DEFAULT_DEPTH
    use_multiverse = True
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
            engine.warmup()
            out.write("readyok\n")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "setoption":
            # setoption name <id> value <x> — Depth and Multiverse honored.
            try:
                ni = tokens.index("name")
                name = tokens[ni + 1]
                val = tokens[tokens.index("value") + 1] if "value" in tokens else ""
                if name.lower() == "depth":
                    max_depth = max(1, min(MAX_DEPTH, int(val)))
                elif name.lower() == "multiverse":
                    use_multiverse = val.strip().lower() not in ("false", "0", "off", "no")
            except (ValueError, IndexError):
                pass
        elif cmd == "position":
            board = parse_position(tokens, board)
        elif cmd == "go":
            depth, budget = go_budget(tokens, board.turn == chess.WHITE,
                                      default_depth=max_depth)
            if "depth" in parse_go(tokens)[0]:
                depth = min(depth, max_depth)  # explicit depth stays capped by the option
            uci, score, done = search_full(board, engine, depth, budget,
                                           use_multiverse=use_multiverse)
            out.write(f"info depth {done} score {score_to_uci(score)} pv {uci}\n"
                      f"bestmove {uci}\n")
        elif cmd == "stop":
            continue  # synchronous search: nothing to interrupt
        elif cmd == "quit":
            break
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
