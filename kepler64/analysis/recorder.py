"""Record engine decisions as portable, presentation-ready data."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import time

import chess
import chess.pgn

import jax.numpy as jnp
import numpy as np

from ..core.evaluate import king_physics, score_white_terms
from ..core.transitions import child_mass_vector
from ..core.universe import GameState
from ..search.minimax import INF, negamax


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


_PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


def _as_state(engine, board):
    if isinstance(board, GameState):
        return board
    return GameState(board, c=float(engine.constants.c))


def _signed(v):
    return [round(float(x), 4) for x in np.asarray(v, dtype=np.float32)]


def analyze_position(engine, board, depth: int = 2,
                     candidate_limit: int | None = 8,
                     include_move_uci: str | None = None):
    """Analyze every legal root move with an exact full-window search.

    `board` is a chess.Board (fresh universe at rest mass) or a GameState
    (the game's carried accretion + velocity — what the engine plays with).
    """
    state = _as_state(engine, board)
    board = state.board
    fb = state.fb
    parent = jnp.asarray(state.masses, dtype=jnp.float32)
    c_lor = float(engine.constants.c)
    candidates = []
    started = time.perf_counter()
    for move in fb.legal_moves():
        child = fb.apply(move)
        masses = child_mass_vector(fb, move, parent, child_board=child,
                                   c_lorentz=c_lor)
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
    full = {
        "best_move_uci": candidates[0]["move_uci"] if candidates else None,
        "candidates": candidates,
        "legal_move_count": len(candidates),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "depth": depth,
        "nodes": sum(row["nodes"] for row in candidates),
        "qnodes": sum(row["qnodes"] for row in candidates),
        "cutoffs": sum(row["cutoffs"] for row in candidates),
    }
    return _trim_analysis(full, candidate_limit, include_move_uci)


def _trim_analysis(analysis, candidate_limit, include_move_uci=None):
    """Keep the top `candidate_limit` rows, plus the played move if it ranked lower."""
    candidates = analysis["candidates"]
    shown = candidates if candidate_limit is None else candidates[:candidate_limit]
    if include_move_uci and not any(row["move_uci"] == include_move_uci for row in shown):
        played = next((row for row in candidates if row["move_uci"] == include_move_uci), None)
        if played is not None:
            shown = [*shown, played]
    return {**analysis, "candidates": shown}


def _capture_event(board: chess.Board, move: chess.Move, parent, child):
    """Accretion bookkeeping for a capture, read from the carried masses."""
    if not board.is_capture(move):
        return None
    victim_sq = move.to_square
    if board.is_en_passant(move):
        victim_sq = move.to_square + (-8 if board.turn == chess.WHITE else 8)
    captor = board.piece_at(move.from_square)
    victim = board.piece_at(victim_sq)
    return {
        "captor_piece": _PIECE_NAMES[captor.piece_type],
        "captor_color": "white" if captor.color == chess.WHITE else "black",
        "victim_piece": _PIECE_NAMES[victim.piece_type],
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "captor_mass_before": abs(float(parent[move.from_square])),
        "victim_mass": abs(float(parent[victim_sq])),
        "captor_mass_after": abs(float(child[move.to_square])),
    }


def _highlights(plies):
    """Game-level narration anchors."""
    collapse = [row["ply"] for row in plies
                if any(k.get("collapsing") for k in row["physics"]["kings"].values())]
    swings = [(abs(row["eval_shift_white"]), row["ply"]) for row in plies]
    heaviest = None
    if plies:
        last = plies[-1]["physics"]
        masses = np.abs(np.asarray(last["masses_after"], dtype=np.float32))
        kings = {k["square"] for k in last["kings"].values()}
        army = [(float(m), sq) for sq, m in enumerate(masses) if sq not in kings and m > 0]
        if army:
            m, sq = max(army)
            heaviest = {"square": chess.square_name(sq), "mass": m}
    return {
        "collapse_plies": collapse,
        "first_collapse_ply": collapse[0] if collapse else None,
        "biggest_swing_ply": max(swings)[1] if swings else None,
        "captures": sum(1 for row in plies if row["physics"]["capture"]),
        "heaviest_piece_final": heaviest,
    }


def record_game(game: chess.pgn.Game, engine, depth: int = 2,
                candidate_limit: int | None = 8, max_plies: int | None = None,
                analyses=None):
    """Record a PGN game and Kepler's decision surface before every move.

    The game is replayed through a GameState, so accretion and Lorentz mass
    carry forward exactly as they do when the engine plays. `analyses` (one
    full, untrimmed analysis per ply) skips re-searching plies already
    analyzed — self-play passes the analyses it chose its moves from.
    """
    state = GameState(game.board(), c=float(engine.constants.c))
    board = state.board
    initial_fen = board.fen()
    plies = []
    moves = list(game.mainline_moves())
    if max_plies is not None:
        moves = moves[:max_plies]

    for ply, played in enumerate(moves, 1):
        fen_before = board.fen()
        parent = state.masses.copy()
        before = _position_eval(parent, engine.constants, board.turn)
        if analyses is not None and ply - 1 < len(analyses):
            analysis = _trim_analysis(analyses[ply - 1], candidate_limit,
                                      include_move_uci=played.uci())
        else:
            analysis = analyze_position(engine, state, depth, candidate_limit,
                                        include_move_uci=played.uci())
        san = board.san(played)
        flags = {
            "is_capture": board.is_capture(played),
            "is_castling": board.is_castling(played),
            "is_en_passant": board.is_en_passant(played),
        }
        selected = next((row for row in analysis["candidates"]
                         if row["move_uci"] == played.uci()), None)
        mover = board.turn
        pre_push = board.copy(stack=False)
        state.push(played)
        child = state.masses
        capture = _capture_event(pre_push, played, parent, child)
        transition = _position_eval(child, engine.constants, mover, parent=parent)
        after = _position_eval(child, engine.constants, board.turn)
        shift_white = after["score_white"] - before["score_white"]
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
            "physics": {
                "masses_after": _signed(child),
                "excess_after": _signed(state.excess()),
                "kings": king_physics(child, engine.constants),
                "capture": capture,
            },
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "promotion": chess.piece_name(played.promotion) if played.promotion else None,
            **flags,
        })

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "Kepler-64",
            "depth": depth,
            "constants": asdict(engine.constants),
        },
        "game": dict(game.headers),
        "initial_fen": initial_fen,
        "result": game.headers.get("Result", board.result()),
        "highlights": _highlights(plies),
        "plies": plies,
    }


def record_selfplay(engine, depth: int = 2, max_plies: int = 40,
                    candidate_limit: int | None = 8):
    state = GameState(c=float(engine.constants.c))
    board = state.board
    game = chess.pgn.Game()
    game.headers.update({"Event": "Kepler-64 self-play", "White": "Kepler-64",
                         "Black": "Kepler-64", "Result": "*"})
    node = game
    analyses = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        analysis = analyze_position(engine, state, depth, candidate_limit=None)
        analyses.append(analysis)
        # The move is the real engine's choice (closed-orbit rule, multiverse
        # tie-break), not just the top analysis row.
        move = engine.play(state, depth=depth) or chess.Move.from_uci(
            analysis["best_move_uci"])
        node = node.add_variation(move)
        state.push(move)
    game.headers["Result"] = board.result()
    return record_game(game, engine, depth, candidate_limit, max_plies,
                       analyses=analyses)
