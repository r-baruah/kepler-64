"""UCI match harness — measure Kepler-64 against real engines.

chess:  plays the RocheEngine against any UCI opponent (Stockfish, Patricia,
        Lc0/Maia, ...) via python-chess, alternating sides, and reports a win
        rate plus an Elo estimate relative to the opponent's known strength.
physics: this is pure measurement — no engine internals are touched. The point
        is to locate Kepler-64 on the Elo axis before/after training.

Opponent spec:
  * plain UCI:  path string, e.g. ".../stockfish.exe"
  * lc0 + net:  {"exe": ".../lc0.exe", "weights": ".../maia-1900.pb.gz"}
"""

import math

import chess
import chess.engine

from .. import RocheEngine
from ..core.constants import Constants


def _launch(opp):
    if isinstance(opp, dict):
        return chess.engine.SimpleEngine.popen_uci([opp["exe"], "--weights=" + opp["weights"]])
    return chess.engine.SimpleEngine.popen_uci(opp)


def _kepler_move(ke, board, depth):
    return ke.play(board, depth=depth)


def play_game(ke, opponent, kepler_white, depth, opp_limit, max_plies=120):
    """Play one game. Returns +1 (Kepler wins), -1 (opp wins), 0 (draw)."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        kepler_to_move = (board.turn == chess.WHITE) if kepler_white else (board.turn == chess.BLACK)
        if kepler_to_move:
            mv = _kepler_move(ke, board, depth)
        else:
            mv = opponent.play(board, opp_limit).move
        if mv is None:
            break
        board.push(mv)
        plies += 1
    oc = board.outcome()
    if oc is None:
        return 0
    if oc.winner is None:
        return 0
    kepler_won = (oc.winner == chess.WHITE) == kepler_white
    return 1 if kepler_won else -1


def elo_estimate(opp_elo, wins, losses, draws):
    """Elo of Kepler-64 implied by the match score vs a known-Elo opponent.

    Solves  score = 1/(1+10**((opp_elo - elo)/400))  for elo.
    Returns None when there is no decisive game (score undefined).
    """
    n = wins + losses + draws
    if n == 0 or (wins == 0 and losses == 0):
        return None
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        # Bound only; exact Elo is unbounded at the extremes.
        return None
    return opp_elo - 400 * math.log10((1.0 / score) - 1.0)


def run_match(opp_spec, opp_elo, games=6, depth=2, opp_limit=None,
              constants=None, seed=0):
    ke = RocheEngine(constants or Constants())
    opponent = _launch(opp_spec)
    if opp_limit is None:
        opp_limit = chess.engine.Limit(time=0.2)
    wins = losses = draws = 0
    for g in range(games):
        kepler_white = (g % 2 == 0)
        res = play_game(ke, opponent, kepler_white, depth, opp_limit)
        wins += res == 1
        losses += res == -1
        draws += res == 0
        print(f"  game {g} (Kepler {'W' if kepler_white else 'B'}): "
              f"{'win' if res==1 else ('loss' if res==-1 else 'draw')}")
    opponent.quit()
    est = elo_estimate(opp_elo, wins, losses, draws)
    print(f"\nKepler-64 vs {opp_elo}-Elo opponent: W {wins} / L {losses} / D {draws}")
    if est is not None:
        print(f"  => implied Kepler-64 Elo ~ {est:.0f}")
    else:
        print("  => score too extreme to bound Elo; Kepler-64 is well below opponent.")
    return wins, losses, draws, est
