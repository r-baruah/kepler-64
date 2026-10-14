"""Single home for the Elo-from-score formula (score = 1/(1+10^(-elo/400)))."""

from __future__ import annotations

import math


def elo_diff(score: float) -> float:
    """Elo difference implied by an expected score strictly inside (0, 1)."""
    return 400.0 * math.log10(score / (1.0 - score))


def score_of(wins: int, losses: int, draws: int) -> float:
    """Match score in [0, 1] (draws count half); 0.5 for an empty record."""
    n = wins + losses + draws
    return 0.5 if n == 0 else (wins + 0.5 * draws) / n


def elo_from_record(wins: int, losses: int, draws: int, bound: float = 600.0):
    """(elo, score) from a W/L/D record; +-`bound` at 0%/100%, 0.0 if empty."""
    if wins + losses + draws == 0:
        return 0.0, 0.5
    score = score_of(wins, losses, draws)
    if score <= 0.0:
        return -bound, score
    if score >= 1.0:
        return bound, score
    return elo_diff(score), score


def elo_proxy(wins: int, draws: int, losses: int, clamp: float = 0.01) -> float:
    """Elo difference with the score clamped to [clamp, 1-clamp] (never inf)."""
    s = min(max(score_of(wins, losses, draws), clamp), 1.0 - clamp)
    return elo_diff(s)
