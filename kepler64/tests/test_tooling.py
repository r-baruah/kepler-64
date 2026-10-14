"""Fast tests for tooling helpers: Elo maths, the xla helper, and glassbox rendering."""
import math
import os

import chess

import pytest

from kepler64.bench.elo import elo_diff, elo_from_record, elo_proxy, score_of


def test_elo_even_score_is_zero():
    assert elo_diff(0.5) == pytest.approx(0.0)
    assert elo_from_record(3, 3, 2) == (pytest.approx(0.0), 0.5)


def test_elo_matches_standard_curve():
    # 64% -> ~ +100 Elo on the standard logistic curve (base 10, 400 scale).
    elo, score = elo_from_record(64, 36, 0)
    assert score == pytest.approx(0.64)
    assert elo == pytest.approx(400 * math.log10(0.64 / 0.36))
    assert 90 < elo < 110
    assert 1 / (1 + 10 ** (-elo / 400)) == pytest.approx(score)


def test_elo_from_record_edges():
    assert elo_from_record(0, 0, 0) == (0.0, 0.5)
    assert elo_from_record(5, 0, 0) == (600.0, 1.0)
    assert elo_from_record(0, 5, 0) == (-600.0, 0.0)
    assert score_of(1, 1, 2) == 0.5


def test_elo_proxy_clamped_and_antisymmetric():
    assert elo_proxy(8, 0, 0) == pytest.approx(400 * math.log10(0.99 / 0.01))
    assert elo_proxy(0, 0, 0) == pytest.approx(0.0)
    assert elo_proxy(5, 2, 1) == pytest.approx(-elo_proxy(1, 2, 5))


def test_tune_xla_respects_existing_flags(monkeypatch):
    from kepler64.training.xla_env import tune_xla
    monkeypatch.setenv("XLA_FLAGS", "--custom")
    tune_xla()
    assert os.environ["XLA_FLAGS"] == "--custom"
    monkeypatch.delenv("XLA_FLAGS")
    monkeypatch.setenv("KEPLER64_NO_XLA_TUNE", "1")
    tune_xla()
    assert "XLA_FLAGS" not in os.environ


def test_render_position_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from kepler64.core.board import Board
    from kepler64.core.constants import Constants
    from kepler64.viz.glassbox import render_position

    out = render_position(Board.from_chess(chess.Board()), Constants(), str(tmp_path / "f.png"),
                          move_no=1, turn="white")
    assert out == str(tmp_path / "f.png")
    data = (tmp_path / "f.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 1000
