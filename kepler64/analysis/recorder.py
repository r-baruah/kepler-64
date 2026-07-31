"""Record engine decisions as portable, presentation-ready data."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import time

import chess
import chess.pgn

from ..core.board import Board
from ..core.evaluate import score_white, score_white_terms
from ..core.fastboard import FastBoard
from ..core.transitions import child_mass_vector
from ..search.minimax import INF, negamax


def _move_tuple(move):
    return (move.from_square, move.to_square, move.promotion or 0)


def _term_dict(terms):
    return {name: float(value) for name, value in zip(terms._fields, terms)}


def _position_eval(masses, constants, turn, parent=None):
    terms = score_white_terms(masses, constants, parent=parent)
    white = float(terms.total)
    return {
        "score_white": white,
        "score_side_to_move": white if turn == chess.WHITE else -white,
        "terms": _term_dict(terms),
    }


def analyze_position(engine, board: chess.Board, depth: int = 2,
                     candidate_limit: int | None = 8,
                     include_move_uci: str | None = None):
    """Analyze every legal root move with an exact full-window search."""
    fb = FastBoard.from_chess(board)
    parent = fb.mass_vector()
    candidates = []
    started = time.perf_counter()
    for move in fb.legal_moves():
        child = fb.apply(move)
        masses = child_mass_vector(fb, move, parent)
        stats = {"nodes": 0, "qnodes": 0, "cutoffs": 0}
        t0 = time.perf_counter()
        score = -negamax(engine, child, depth - 1, -INF, INF, masses,
                         parent_masses=parent, stats=stats)
        elapsed = time.perf_counter() - t0
        terms = score_white_terms(masses, engine.constants, parent=parent)
        white = float(terms.total)
        py_move = chess.Move(move[0], move[1], promotion=move[2] or None)
        candidates.append({
            "move_uci": py_move.uci(),
            "move_san": board.san(py_move),
            "search_score_mover": float(score),
            "immediate_score_white": white,
            "immediate_score_mover": white if board.turn == chess.WHITE else -white,
            "terms": _term_dict(terms),
            "depth": depth,
            "nodes": stats["nodes"],
            "qnodes": stats["qnodes"],
            "cutoffs": stats["cutoffs"],
            "elapsed_ms": elapsed * 1000.0,
        })
    candidates.sort(key=lambda row: row["search_score_mover"], reverse=True)
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
    shown = candidates if candidate_limit is None else candidates[:candidate_limit]
    if include_move_uci and not any(row["move_uci"] == include_move_uci for row in shown):
        played = next((row for row in candidates if row["move_uci"] == include_move_uci), None)
        if played is not None:
            shown = [*shown, played]
    return {
        "best_move_uci": candidates[0]["move_uci"] if candidates else None,
        "candidates": shown,
        "legal_move_count": len(candidates),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "depth": depth,
        "nodes": sum(row["nodes"] for row in candidates),
        "qnodes": sum(row["qnodes"] for row in candidates),
        "cutoffs": sum(row["cutoffs"] for row in candidates),
    }


def record_game(game: chess.pgn.Game, engine, depth: int = 2,
                candidate_limit: int | None = 8, max_plies: int | None = None):
    """Record a PGN game and Kepler's decision surface before every move."""
    board = game.board()
    initial_fen = board.fen()
    plies = []
    moves = list(game.mainline_moves())
    if max_plies is not None:
        moves = moves[:max_plies]

    for ply, played in enumerate(moves, 1):
        fen_before = board.fen()
        fb = FastBoard.from_chess(board)
        parent = fb.mass_vector()
        before = _position_eval(parent, engine.constants, board.turn)
        analysis = analyze_position(engine, board, depth, candidate_limit,
                                    include_move_uci=played.uci())
        san = board.san(played)
        flags = {
            "is_capture": board.is_capture(played),
            "is_castling": board.is_castling(played),
            "is_en_passant": board.is_en_passant(played),
        }
        played_tuple = _move_tuple(played)
        played_masses = child_mass_vector(fb, played_tuple, parent)
        transition = _position_eval(played_masses, engine.constants,
                                    board.turn, parent=parent)
        selected = next((row for row in analysis["candidates"]
                         if row["move_uci"] == played.uci()), None)
        board.push(played)
        after_mass = FastBoard.from_chess(board).mass_vector()
        after = _position_eval(after_mass, engine.constants, board.turn)
        shift_white = after["score_white"] - before["score_white"]
        mover = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
        plies.append({
            "ply": ply,
            "move_number": (ply + 1) // 2,
            "mover": "white" if mover == chess.WHITE else "black",
            "move_uci": played.uci(),
            "move_san": san,
            "fen_before": fen_before,
            "fen_after": board.fen(),
            "eval_before": before,
            "eval_after": after,
            "eval_shift_white": shift_white,
            "eval_shift_mover": shift_white if mover == chess.WHITE else -shift_white,
            "played_move_evaluation": transition,
            "selected_candidate_rank": selected["rank"] if selected else None,
            "analysis": analysis,
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "promotion": chess.piece_name(played.promotion) if played.promotion else None,
            **flags,
        })

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "Kepler-64",
            "depth": depth,
            "constants": asdict(engine.constants),
        },
        "game": dict(game.headers),
        "initial_fen": initial_fen,
        "result": game.headers.get("Result", board.result()),
        "plies": plies,
    }


def record_selfplay(engine, depth: int = 2, max_plies: int = 40,
                    candidate_limit: int | None = 8):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update({"Event": "Kepler-64 self-play", "White": "Kepler-64",
                         "Black": "Kepler-64", "Result": "*"})
    node = game
    for _ in range(max_plies):
        if board.is_game_over():
            break
        analysis = analyze_position(engine, board, depth, candidate_limit=None)
        move = chess.Move.from_uci(analysis["best_move_uci"])
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result()
    return record_game(game, engine, depth, candidate_limit, max_plies)
