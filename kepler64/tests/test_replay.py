import chess
import chess.pgn
from pathlib import Path

from kepler64 import Constants, RocheEngine
from kepler64.analysis import record_game
from kepler64.viz.replay import _display_candidates, render_html


def test_short_game_records_candidate_analysis():
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for uci in ("e2e4", "e7e5"):
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    replay = record_game(game, RocheEngine(Constants()), depth=1,
                         candidate_limit=4, max_plies=2)
    assert len(replay["plies"]) == 2
    assert replay["plies"][0]["move_uci"] == "e2e4"
    assert replay["plies"][0]["analysis"]["candidates"]
    assert replay["plies"][0]["played_move_evaluation"]["terms"]["total"] \
        == replay["plies"][0]["played_move_evaluation"]["score_white"]


def test_display_candidates_retains_low_rank_selected_move():
    row = {
        "move_uci": "h2h4",
        "analysis": {"candidates": [
            {"move_uci": f"a2a{i}", "rank": i} for i in range(1, 10)
        ] + [{"move_uci": "h2h4", "rank": 10}]},
    }
    shown = _display_candidates(row, limit=8)
    assert len(shown) == 8
    assert shown[-1]["move_uci"] == "h2h4"


def test_render_html_contains_ephemeris_cta_and_semantic_readout(tmp_path):
    game = chess.pgn.Game()
    game.add_variation(chess.Move.from_uci("e2e4"))
    replay = record_game(game, RocheEngine(Constants()), depth=1,
                         candidate_limit=4, max_plies=1)
    frame = tmp_path / "frame.png"
    summary = tmp_path / "summary.png"
    gif = tmp_path / "replay.gif"
    for path in (frame, summary, gif):
        path.write_bytes(b"image")
    output = tmp_path / "index.html"
    render_html(replay, [str(frame)], gif, summary, output)
    document = output.read_text(encoding="utf-8")
    assert "What if gravity could play chess?" in document
    assert "View source on GitHub" in document
    assert "Candidate search" in document
    assert "aria-live=\"polite\"" in document


def test_render_html_handles_empty_replay(tmp_path):
    replay = {"game": {}, "result": "*", "plies": []}
    output = tmp_path / "index.html"
    render_html(replay, [], tmp_path / "missing.gif",
                tmp_path / "missing.png", output)
    document = output.read_text(encoding="utf-8")
    assert "No analyzed plies are available" in document
    assert "No animation was generated" in document
