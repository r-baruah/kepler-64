"""Tests for the UCI match harness logic (no external engine needed).

`run_match` requires a real UCI opponent binary; everything below it —
Elo estimation and game adjudication — is verified here with a mock.
"""

import numpy as np
import pytest

from ..core.constants import Constants
from ..match.uci_harness import elo_estimate, play_game


def test_elo_estimate_math():
    # A 50% score against an equal-strength opponent implies the same Elo.
    assert elo_estimate(1500, 1, 1, 0) == pytest.approx(1500, abs=1e-6)
    # 75% score against a 1500-Elo opponent implies ~+190 Elo.
    est = elo_estimate(1500, 3, 1, 0)
    assert 1680 < est < 1700
    # All draws (no information) -> None.
    assert elo_estimate(1500, 0, 0, 5) is None
    # No games at all -> None.
    assert elo_estimate(1500, 0, 0, 0) is None


def test_play_game_against_mock_opponent():
    """Full game loop with a deterministic-ish mock: legal moves only,
    bounded plies, and a valid result in {-1, 0, +1}."""
    import chess
    import chess.engine

    class MockOpponent:
        def __init__(self, seed=7):
            self.rng = np.random.default_rng(seed)

        def play(self, board, limit):
            mv = self.rng.choice(list(board.legal_moves))
            return chess.engine.PlayResult(mv, None)

    from .. import RocheEngine

    ke = RocheEngine(constants=Constants(), load_trained=False)
    res = play_game(ke, MockOpponent(), kepler_white=True, depth=1,
                    opp_limit=None, max_plies=24)
    assert res in (-1, 0, 1)
