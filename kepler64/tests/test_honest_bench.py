"""Honesty harness: material-only ablation and result bookkeeping."""

import chess

from kepler64 import Constants, RocheEngine
from kepler64.bench.honest import material_only, play_game, random_openings, summarize
from kepler64.core.evaluate import score_white_terms
from kepler64.core.fastboard import FastBoard


def test_material_only_scores_exactly_material():
    c = material_only(Constants())
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    m = FastBoard.from_chess(board).mass_vector()
    terms = score_white_terms(m, c, parent=m)
    assert abs(float(terms.total) - float(terms.material)) < 1e-4


def test_random_openings_are_distinct_and_seeded():
    a = random_openings(6, plies=4, seed=3)
    assert len({" ".join(x) for x in a}) == 6
    assert a == random_openings(6, plies=4, seed=3)


def test_summary_separates_adjudication_from_board_results():
    recs = [
        {"result": "1-0", "a_color": "white", "termination": "checkmate",
         "adjudicated": None, "moves": "a", "first_collapse_ply": {"white": None, "black": 30}},
        {"result": "*", "a_color": "black", "termination": "ply_cap",
         "adjudicated": "0-1", "moves": "b", "first_collapse_ply": {"white": None, "black": None}},
        {"result": "*", "a_color": "white", "termination": "ply_cap",
         "adjudicated": "0-1", "moves": "b", "first_collapse_ply": {"white": None, "black": None}},
    ]
    s = summarize(recs)
    assert s["decided_on_board"] == {"win": 1, "draw": 0, "loss": 0}
    assert s["adjudicated_at_cap"] == {"win": 1, "draw": 0, "loss": 1}
    assert s["distinct_games"] == 2
    assert s["games_with_roche_collapse"] == 1


def test_play_game_reports_cap_and_adjudication():
    eng = RocheEngine(Constants(), load_trained=False)
    rec = play_game(eng, eng, ["e2e4", "e7e5"], depth=1, max_plies=2)
    assert rec["termination"] == "ply_cap"
    assert rec["adjudicated"] in ("1-0", "0-1", "1/2-1/2")
    assert rec["plies"] == 4
