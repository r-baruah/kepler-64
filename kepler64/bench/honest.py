"""Honest head-to-head harness: what actually decided each game.

Three things the older match loops (bench/selfmatch, credibility_gate) did
not separate, and a skeptical reader will ask about:

1. **How a game ended.** Checkmate on the board, a draw by rule, or the ply
   cap followed by mass adjudication (`_terminal_mass_edge`). Adjudicated
   results are counted separately, never folded into "wins".
2. **Whether the games were different.** Both engines are deterministic at a
   fixed depth, so N games from the start position can be the same game
   repeated. Each game here opens from a seeded random opening, and the
   report counts distinct move sequences.
3. **Whether the physics does anything beyond counting material.**
   `material_only()` builds constants under which every field term vanishes
   (G = 0 and every positional gain 0); the score is then exactly
   `mat_gain * (army mass difference)` with the same search. Physics vs
   material-only is the ablation.

Games are played through `GameState`, so accretion and velocity carry across
the game exactly as the physics claims.
"""

from __future__ import annotations

from dataclasses import replace
import random

import chess
import chess.pgn

from ..core.constants import Constants
from ..core.evaluate import king_physics
from ..core.universe import GameState


def material_only(base: Constants | None = None) -> Constants:
    """Same universe with every field term switched off.

    G = 0 zeroes the potential and force fields, which zeroes the tidal index
    (eta), drift, binding, wave loss and horizon overlap. The two disruption
    gauges become the same constant and cancel. The positional delta gains
    are zeroed too. What is left is mat_gain * (white army - black army).
    """
    base = base or Constants()
    return replace(base, G=0.0, gamma=0.0, lambda_delta=0.0, com_gain=0.0,
                   inertia_gain=0.0, entropy_gain=0.0, lambda_drift=0.0,
                   lambda_gw=0.0, lambda_sch=0.0)


def random_openings(n: int, plies: int = 4, seed: int = 0) -> list[list[str]]:
    """`n` distinct random opening lines of `plies` half-moves (UCI)."""
    rng = random.Random(seed)
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        board = chess.Board()
        line = []
        for _ in range(plies):
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            if not moves:
                break
            mv = rng.choice(moves)
            board.push(mv)
            line.append(mv.uci())
        key = " ".join(line)
        if key not in seen and not board.is_game_over():
            seen.add(key)
            out.append(line)
    return out


def _mass_edge(board: chess.Board) -> float:
    from ..training.selfplay import _terminal_mass_edge
    return _terminal_mass_edge(board)


def play_game(eng_white, eng_black, opening: list[str] | None = None, *,
              depth: int | None = 2, move_ms: float | None = None,
              max_plies: int | None = 200, observer: Constants | None = None) -> dict:
    """Play one game. `max_plies=None` plays to a rule-based ending.

    Returns the result, how it was reached ("checkmate", "stalemate",
    "repetition", "fifty_moves", "insufficient_material", "ply_cap"), the
    mass-adjudicated result when the cap was hit, and the first ply at which
    either king's Roche gauge crossed 0.5 under `observer` constants.
    """
    observer = observer or eng_white.constants
    # One game, one light speed for the carried velocity boost: the
    # observer's c. Engines with a different c search from these masses with
    # their own c; the velocity boost is small, so the mismatch is minor.
    state = GameState(c=float(observer.c))
    for uci in opening or []:
        state.push(chess.Move.from_uci(uci))
    book_plies = len(state.board.move_stack)
    collapse = {"white": None, "black": None}
    plies = 0
    while not state.board.is_game_over(claim_draw=True):
        if max_plies is not None and plies >= max_plies:
            break
        eng = eng_white if state.board.turn == chess.WHITE else eng_black
        if move_ms is not None:
            mv = eng.play(state, search_time_ms=move_ms, use_multiverse=False)
        else:
            mv = eng.play(state, depth=depth, use_multiverse=False)
        if mv is None:
            break
        state.push(mv)
        plies += 1
        kp = king_physics(state.masses, observer)
        for side in ("white", "black"):
            if collapse[side] is None and kp.get(side, {}).get("collapsing"):
                collapse[side] = book_plies + plies

    board = state.board
    outcome = board.outcome(claim_draw=True)
    adjudicated = None
    if outcome is not None:
        termination = {
            chess.Termination.CHECKMATE: "checkmate",
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
            chess.Termination.FIFTY_MOVES: "fifty_moves",
            chess.Termination.SEVENTYFIVE_MOVES: "fifty_moves",
            chess.Termination.THREEFOLD_REPETITION: "repetition",
            chess.Termination.FIVEFOLD_REPETITION: "repetition",
        }.get(outcome.termination, outcome.termination.name.lower())
        result = outcome.result()
    else:
        termination = "ply_cap"
        result = "*"
        edge = _mass_edge(board)
        adjudicated = "1-0" if edge > 0 else "0-1" if edge < 0 else "1/2-1/2"

    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = result
    return {
        "result": result,
        "termination": termination,
        "adjudicated": adjudicated,
        "plies": book_plies + plies,
        "opening": list(opening or []),
        "moves": " ".join(m.uci() for m in board.move_stack),
        "first_collapse_ply": collapse,
        "pgn": str(game),
    }


def run_match(constants_a: Constants, constants_b: Constants, *, games: int = 20,
              depth: int | None = 2, move_ms: float | None = None,
              max_plies: int | None = 200, opening_plies: int = 4,
              seed: int = 0, observer: Constants | None = None,
              on_game=None) -> dict:
    """A vs B with colours alternating on each opening (pairs of games)."""
    from .. import RocheEngine

    eng_a = RocheEngine(constants_a, load_trained=False)
    eng_b = RocheEngine(constants_b, load_trained=False)
    pairs = (games + 1) // 2
    openings = random_openings(pairs, opening_plies, seed) if opening_plies else [[]] * pairs
    records = []
    for g in range(games):
        opening = openings[(g // 2) % len(openings)] if openings else []
        a_white = g % 2 == 0
        white, black = (eng_a, eng_b) if a_white else (eng_b, eng_a)
        rec = play_game(white, black, opening, depth=depth, move_ms=move_ms,
                        max_plies=max_plies, observer=observer or constants_a)
        rec["game"] = g
        rec["a_color"] = "white" if a_white else "black"
        records.append(rec)
        if on_game:
            on_game(rec)
    return {"records": records, "summary": summarize(records)}


def _score_for_a(result: str, a_color: str) -> str | None:
    if result == "1/2-1/2":
        return "draw"
    if result not in ("1-0", "0-1"):
        return None
    white_won = result == "1-0"
    return "win" if white_won == (a_color == "white") else "loss"


def summarize(records: list[dict]) -> dict:
    """Split decided-on-the-board from adjudicated, and count distinct games."""
    board = {"win": 0, "draw": 0, "loss": 0}
    adjudicated = {"win": 0, "draw": 0, "loss": 0}
    terminations: dict[str, int] = {}
    collapse_before_end = 0
    for rec in records:
        terminations[rec["termination"]] = terminations.get(rec["termination"], 0) + 1
        res = _score_for_a(rec["result"], rec["a_color"])
        if res is not None:
            board[res] += 1
        else:
            adj = _score_for_a(rec["adjudicated"] or "1/2-1/2", rec["a_color"])
            adjudicated[adj] += 1
        if any(v is not None for v in rec["first_collapse_ply"].values()):
            collapse_before_end += 1
    n = len(records)
    decided = board["win"] + board["loss"] + board["draw"]
    return {
        "games": n,
        "distinct_games": len({rec["moves"] for rec in records}),
        "decided_on_board": board,
        "adjudicated_at_cap": adjudicated,
        "terminations": terminations,
        "board_score_a": (board["win"] + 0.5 * board["draw"]) / decided if decided else None,
        "games_with_roche_collapse": collapse_before_end,
    }
