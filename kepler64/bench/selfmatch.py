"""Honest strength measurement: the trained universe vs the one it came from.

chess:  plays a fixed-budget head-to-head match between a CANDIDATE engine
        (some constants) and a BASELINE engine (other constants), alternating
        colors so the first-move advantage cancels. Reports wins / draws /
        losses and an Elo estimate. No external engines needed — the improvement
        is measured RELATIVE to the prior universe, which is exactly what the
        training loop's accept/reject gate needs.
physics: the two engines differ ONLY in their physical constants. Every other
        law (board, legality, search) is identical, so a win-rate edge is
        attributable to the evaluation physics alone — the clean ablation.

This is the Elo axis the training pipeline climbs. It never consults a book or
a hand-written evaluation: the match is pure search physics vs search physics.
"""

from __future__ import annotations

import math

import chess
import numpy as np

from ..core.constants import Constants


def play_match(candidate: Constants, baseline: Constants, *,
               games: int = 16, move_ms: float = 120.0,
               max_plies: int = 120, seed: int = 0) -> dict:
    """Head-to-head `candidate` vs `baseline`. Returns
    {"wins", "losses", "draws", "elo", "games", "score"}.

    `elo` is the candidate's Elo advantage over the baseline computed from the
    match score via the standard Elo curve (score = 1/(1+10^(-elo/400))).
    """
    from .. import RocheEngine

    rng = __import__("numpy").random.default_rng(seed)
    cand = RocheEngine(candidate)
    base = RocheEngine(baseline)
    wins = losses = draws = 0
    for g in range(games):
        board = chess.Board()
        # Alternate colors: game parity makes the White advantage cancel out.
        cand_white = (g % 2 == 0)
        plies = 0
        while not board.is_game_over() and plies < max_plies:
            cand_turn = (board.turn == chess.WHITE) == cand_white
            eng = cand if cand_turn else base
            mv = eng.play(board, search_time_ms=move_ms)
            if mv is None:
                break
            board.push(mv)
            plies += 1
        oc = board.outcome()
        if oc is None or oc.winner is None:
            draws += 1
        else:
            cand_won = (oc.winner == chess.WHITE) == cand_white
            if cand_won:
                wins += 1
            else:
                losses += 1
    elo, score = _elo_from_score(wins, losses, draws)
    return {"wins": wins, "losses": losses, "draws": draws, "games": games,
            "score": score, "elo": elo}


def _elo_from_score(wins: int, losses: int, draws: int):
    """Elo from a (w, l, d) record. Returns (elo, score) or (0.0, 0.5) if empty."""
    n = wins + losses + draws
    if n == 0:
        return 0.0, 0.5
    score = (wins + 0.5 * draws) / n
    if score <= 0.0:
        return -600.0, score
    if score >= 1.0:
        return 600.0, score
    return -400.0 * math.log2(1.0 / score - 1.0), score
