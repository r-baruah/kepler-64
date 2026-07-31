"""Curriculum training data from a real UCI engine (Stockfish).

chess:  Stockfish's `Skill Level` option emulates a range of playing strengths
        without any internet/data download. We play self-play games at a chosen
        skill, then harvest (position, expert-move) pairs plus game outcomes.
        Ramp the skill level up as training progresses (the user's curriculum:
        start ~1000-1200, increase toward expert).
physics: the harvested moves are the SUPERVISORY TARGET. We never invent chess
        rules — we only let gradient descent reshape the physical constants so
        the gravity eval agrees with strong play.

build_policy_batch() returns (moves_m (N,K,64), expert_idx (N,), turns (N,),
mask (N,K)) ready for training/loss.py (mask zeroes dummy child slots).
"""

import chess
import chess.engine
import jax.numpy as jnp

from ..core.fastboard import FastBoard
from ..core.transitions import child_mass_vector


def _mv_uci(mv):
    """FastBoard move tuple (from, to, promo) -> UCI string."""
    f, t, pr = mv
    s = chess.square_name(f) + chess.square_name(t)
    if pr:
        s += "pnbrqk"[pr - 1]
    return s


def play_selfplay(engine_path, skill_level=1, n_games=20, max_plies=80,
                  limit=None, sample_every=3):
    """Play Stockfish-vs-Stockfish at `skill_level`.

    Returns `games`: list of (rows, outcome) where
      rows   = list of (fen, expert_uci, turn) sampled every `sample_every` plies
      outcome = {-1,0,+1} from White's view
    """
    if limit is None:
        limit = chess.engine.Limit(time=0.05)
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"Skill Level": skill_level})

    games = []
    for _ in range(n_games):
        board = chess.Board()
        rows = []
        plies = 0
        while not board.is_game_over() and plies < max_plies:
            mv = eng.play(board, limit).move
            if mv is None:
                break
            if plies % sample_every == 0:
                rows.append((board.fen(), mv.uci(), 0 if board.turn == chess.WHITE else 1))
            board.push(mv)
            plies += 1
        oc = board.outcome()
        if oc is None or oc.winner is None:
            outcome = 0.0
        else:
            outcome = 1.0 if oc.winner == chess.WHITE else -1.0
        games.append((rows, outcome))
    eng.quit()
    return games


def build_policy_batch(games, max_children: int = 40):
    """Materialize training tensors from `games` (see play_selfplay)."""
    moves_list, expert_idx, turns, Y, mask_list = [], [], [], [], []
    for rows, outcome in games:
        for fen, expert_uci, turn in rows:
            fb = FastBoard.from_chess(chess.Board(fen))
            legal = fb.legal_moves()
            child_m = []
            eidx = -1
            for i, m in enumerate(legal):
                child_m.append(child_mass_vector(fb, m, fb.mass_vector()))
                if _mv_uci(m) == expert_uci:
                    eidx = i
            if eidx < 0 or not child_m:
                continue
            # truncate if too many children
            if len(child_m) > max_children:
                if eidx >= max_children:
                    continue
                child_m = child_m[:max_children]
            k = len(child_m)
            # pad to max_children with zero mass vectors (masked out in loss)
            while len(child_m) < max_children:
                child_m.append(jnp.zeros(64, dtype=jnp.float32))
            mask = [1.0] * k + [0.0] * (max_children - k)
            moves_list.append(jnp.stack(child_m).astype(jnp.float32))
            expert_idx.append(eidx)
            turns.append(turn)
            Y.append(outcome)
            mask_list.append(jnp.array(mask, dtype=jnp.float32))
    return (jnp.stack(moves_list).astype(jnp.float32),
            jnp.array(expert_idx, dtype=jnp.int32),
            jnp.array(turns, dtype=jnp.int32),
            jnp.array(Y, dtype=jnp.float32),
            jnp.array(mask_list, dtype=jnp.float32))
