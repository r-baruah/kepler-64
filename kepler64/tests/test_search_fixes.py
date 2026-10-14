"""Regression tests for the search/UCI fixes: mate distance, in-check
quiescence, time abort, TT mate adjustment, null-move guard, UCI parsing."""
import io
import time

import chess

from kepler64 import RocheEngine
from kepler64 import uci
from kepler64.core.constants import Constants
from kepler64.core.fastboard import FastBoard
from kepler64.search import minimax
from kepler64.search.minimax import (
    MATE, INF, iterative_search, _quiesce, _score_position, _has_non_pawn_material,
    _tt_score_in, _tt_score_out, seen_positions,
)

ENG = RocheEngine(constants=Constants(), load_trained=False)


def _uci(mv):
    f, t, p = mv
    return chess.Move(f, t, chess.PieceType(p) if p else None).uci()


def test_mate_in_one_scores_mate_distance():
    fb = FastBoard.from_chess(chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"))
    mv, sc = iterative_search(ENG, fb, max_depth=2)
    assert _uci(mv) == "d1d8"
    assert sc == MATE - 1


def test_prefers_mate_in_one_over_slower_mate():
    # KQ vs K: Qg7# is mate in 1, plenty of mate-in-2 moves also exist.
    fb = FastBoard.from_chess(chess.Board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"))
    mv, sc = iterative_search(ENG, fb, max_depth=3)
    assert _uci(mv) == "g1g7"
    assert sc == MATE - 1


def test_tt_mate_scores_are_ply_adjusted():
    assert _tt_score_out(_tt_score_in(MATE - 5, 3), 3) == MATE - 5
    assert _tt_score_out(_tt_score_in(-(MATE - 5), 3), 3) == -(MATE - 5)
    assert _tt_score_in(12.5, 4) == 12.5
    # A mate stored at one ply and probed at another shifts by the ply gap.
    stored = _tt_score_in(MATE - 7, 3)          # mate 4 plies below the node
    assert _tt_score_out(stored, 5) == MATE - 9


def test_in_check_quiescence_does_not_stand_pat():
    b = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")  # Qe2 checks the white king
    assert b.is_check()
    fb = FastBoard.from_chess(b)
    # A huge known stand-pat would be returned verbatim if stand-pat applied.
    val = _quiesce(ENG, fb, -INF, 0.0, qdepth=3, known_stand=1e6)
    assert val < 1e6
    # Not in check: the known stand-pat is honoured (fail-soft cutoff).
    quiet = FastBoard.from_chess(chess.Board())
    assert _quiesce(ENG, quiet, -INF, 0.0, qdepth=3, known_stand=1e6) == 1e6


def test_null_move_zugzwang_guard():
    kp = FastBoard.from_chess(chess.Board("8/8/4k3/8/4P3/4K3/8/8 w - - 0 1"))
    assert not _has_non_pawn_material(kp)
    assert _has_non_pawn_material(FastBoard.from_chess(chess.Board()))
    black = FastBoard.from_chess(chess.Board("8/8/4k3/8/4P3/4K3/8/8 b - - 0 1"))
    assert not _has_non_pawn_material(black)


def test_time_budget_aborts_and_returns_legal_move():
    b = chess.Board()
    fb = FastBoard.from_chess(b)
    t = time.perf_counter()
    mv, _ = iterative_search(ENG, fb, max_depth=8, time_ms=300)
    assert time.perf_counter() - t < 8.0
    assert chess.Move.from_uci(_uci(mv)) in b.legal_moves


def test_repeated_root_child_scores_draw_even_at_depth_one():
    b = chess.Board()
    for m in ("g1f3", "g8f6", "f3g1", "f6g8"):
        b.push_uci(m)
    seen = seen_positions(b)
    fb = FastBoard.from_chess(b)
    _, _, scored, _ = minimax._search_root_iteration(
        ENG, fb, fb.mass_vector(float(ENG.constants.c)),
        fb.legal_moves(), 1, -INF, INF, seen, None, minimax._SearchCtx(),
        time.perf_counter(), None)
    repeats = [v for m, v in scored if _uci(m) in ("g1f3",)]
    assert repeats == [0.0]


def test_history_is_capped():
    ctx = minimax._SearchCtx()
    ctx.history[5] = minimax._HISTORY_MAX
    ctx.age_history()
    assert ctx.history[5] == minimax._HISTORY_MAX // 2
    assert minimax._HISTORY_MAX < 20_000


# ── UCI ──────────────────────────────────────────────────────────────────────

def test_go_parser_valueless_tokens():
    vals, flags = uci.parse_go("go ponder wtime 1000 btime 2000 infinite".split())
    assert flags == {"ponder", "infinite"}
    assert vals == {"wtime": 1000.0, "btime": 2000.0}
    depth, budget = uci.go_budget("go infinite".split(), True)
    assert (depth, budget) == (uci.MAX_DEPTH, None)
    depth, budget = uci.go_budget("go wtime 60000 btime 60000".split(), True)
    assert depth == uci.MAX_DEPTH and budget and budget > 10
    assert uci.go_budget(["go"], True, default_depth=2) == (2, None)


def test_score_to_uci_mate_and_inf():
    assert uci.score_to_uci(MATE - 1) == "mate 1"
    assert uci.score_to_uci(MATE - 3) == "mate 2"
    assert uci.score_to_uci(-(MATE - 2)) == "mate -1"
    assert uci.score_to_uci(INF).startswith("cp ")
    assert uci.score_to_uci(-INF).startswith("cp ")
    assert uci.score_to_uci(1.5) == "cp 150"
    assert uci.score_to_uci(None) == "cp 0"


def _run_main(monkeypatch, script):
    out = io.StringIO()
    monkeypatch.setattr(uci.sys, "stdin", io.StringIO(script))
    monkeypatch.setattr(uci.sys, "stdout", out)
    uci.main()
    return out.getvalue()


def test_uci_movetime_deepens_beyond_depth_option(monkeypatch):
    calls = []

    def fake(board, engine, max_depth, time_ms, use_multiverse=True):
        calls.append((max_depth, time_ms, use_multiverse))
        return "e2e4", 0.0, 1

    monkeypatch.setattr(uci, "search_full", fake)
    _run_main(monkeypatch, "setoption name Depth value 1\ngo movetime 500\n"
                           "go\nquit\n")
    assert calls[0][0] == uci.MAX_DEPTH and calls[0][1] is not None
    assert calls[1] == (1, None, True)  # bare `go` honours the Depth option


def test_uci_multiverse_option_honoured(monkeypatch):
    calls = []

    def fake(board, engine, max_depth, time_ms, use_multiverse=True):
        calls.append(use_multiverse)
        return "e2e4", MATE - 1, 2

    monkeypatch.setattr(uci, "search_full", fake)
    out = _run_main(monkeypatch, "go depth 1\nsetoption name Multiverse value false\n"
                                 "go depth 1\nquit\n")
    assert calls == [True, False]
    assert "score mate 1" in out and "info depth 2" in out


def test_uci_live_movetime_reports_completed_depth(monkeypatch):
    out = _run_main(monkeypatch,
                    "position fen 7k/8/5K2/8/8/8/8/6Q1 w - - 0 1\n"
                    "setoption name Depth value 1\ngo movetime 1500\nquit\n")
    line = [l for l in out.splitlines() if l.startswith("info depth")][0]
    assert int(line.split()[2]) >= 2
    assert "bestmove g1g7" in out
