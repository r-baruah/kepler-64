"""UCI wire: parsing + a live depth-1 search returns a legal move."""
import chess

from .. import RocheEngine
from ..core.constants import Constants
from ..uci import go_budget, parse_position, search


def test_position_with_moves():
    b = parse_position("position startpos moves e2e4 e7e5".split(), chess.Board())
    assert b.move_stack and b.peek().uci() == "e7e5"


def test_go_depth_parse():
    depth, budget = go_budget("go depth 2".split(), True)
    assert (depth, budget) == (2, None)


def test_go_movetime_parse():
    depth, budget = go_budget("go movetime 500".split(), True)
    assert depth == 8 and budget == 480.0


def test_search_returns_legal_move():
    eng = RocheEngine(constants=Constants(), load_trained=False)
    uci, _ = search(chess.Board(), eng, max_depth=1, time_ms=None)
    assert chess.Move.from_uci(uci) in chess.Board().legal_moves
