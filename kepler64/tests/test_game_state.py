"""Game-line physics: accretion and light speed carry across real moves."""

import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from kepler64 import Constants, RocheEngine
from kepler64.analysis import record_game
from kepler64.core.constants import (
    TRAINABLE_LEAVES, leaves_to_array, load_constants,
)
from kepler64.core.evaluate import king_physics
from kepler64.core.fastboard import FastBoard
from kepler64.core.universe import GameState
from kepler64.training.loss import _unpack

ROOT = Path(__file__).resolve().parents[2]


def _game(ucis):
    game = chess.pgn.Game()
    node = game
    for uci in ucis:
        node = node.add_variation(chess.Move.from_uci(uci))
    return game


def test_accretion_persists_across_the_game_line():
    # 1.e4 d5 2.exd5 Nf6 3.Nc3 — the d5 pawn ate a pawn and keeps that mass
    # through two later quiet moves.
    state = GameState.from_moves(
        [chess.Move.from_uci(u) for u in ("e2e4", "d7d5", "e4d5", "g8f6", "b1c3")])
    d5 = chess.D5
    rest = float(FastBoard.from_chess(state.board).mass_vector(state.c)[d5])
    assert abs(float(state.masses[d5])) > abs(rest) + 0.5
    # A fresh board built from the same position has forgotten it.
    assert abs(rest) < 1.5


def test_recorder_uses_the_learned_light_speed():
    # A rook that slid 7 squares carries velocity; its Lorentz mass depends on
    # c. The recorder must use the engine's c, not the 6.0 fallback.
    game = _game(["h2h4", "a7a6", "h1h3", "a6a5", "h3a3"])
    slow = Constants(c=2.0)
    replay = record_game(game, RocheEngine(slow), depth=1, candidate_limit=2)
    expected = GameState.from_moves(list(game.mainline_moves()), c=2.0).masses
    got = np.asarray(replay["plies"][-1]["physics"]["masses_after"], dtype=np.float32)
    assert np.allclose(got, expected, atol=1e-3)
    default_c = GameState.from_moves(list(game.mainline_moves())).masses
    assert not np.allclose(got, default_c, atol=1e-3)


def test_replay_carries_capture_events_and_highlights():
    game = _game(["e2e4", "d7d5", "e4d5", "d8d5"])
    replay = record_game(game, RocheEngine(Constants()), depth=1, candidate_limit=2)
    assert replay["schema_version"] == 2
    cap = replay["plies"][2]["physics"]["capture"]
    assert cap["captor_piece"] == "pawn" and cap["victim_piece"] == "pawn"
    assert cap["captor_mass_after"] > cap["captor_mass_before"]
    queen_cap = replay["plies"][3]["physics"]["capture"]
    # The queen swallows the pawn that had already accreted: victim > 1.
    assert queen_cap["victim_mass"] > 1.5
    assert replay["highlights"]["captures"] == 2
    kings = replay["plies"][0]["physics"]["kings"]
    assert set(kings) == {"white", "black"}


def test_play_accepts_game_state_without_mutating_it():
    state = GameState.from_moves([chess.Move.from_uci("e2e4")])
    before = state.masses.copy()
    move = RocheEngine(Constants()).play(state, depth=1, use_multiverse=False)
    assert move in state.board.legal_moves
    assert np.array_equal(state.masses, before)


def test_king_physics_quiet_at_the_start():
    masses = FastBoard.from_chess(chess.Board()).mass_vector()
    kp = king_physics(masses, Constants())
    for side in ("white", "black"):
        assert 0.0 <= kp[side]["gauge"] <= 1.0
        assert not kp[side]["collapsing"]


def test_loss_mat_gain_floor_matches_leaf_bound():
    arr = leaves_to_array(Constants(mat_gain=1.0))
    unpacked = _unpack(arr)
    assert float(unpacked[TRAINABLE_LEAVES.index("mat_gain")]) == 1.5


def test_trained_constants_with_meta_load():
    path = ROOT / "kepler64" / "training" / "trained_constants.json"
    assert "_meta" in json.loads(path.read_text())
    assert load_constants(path) is not None


def test_normalized_policy_loss_is_finite_on_a_flat_row():
    # Every child identical -> zero score spread. Normalization must not
    # divide by zero; the loss is the uniform cross-entropy log(n).
    import jax.numpy as jnp
    from kepler64.training.loss import loss

    parent = np.asarray(FastBoard.from_chess(chess.Board()).mass_vector(), dtype=np.float32)
    n = 5
    moves_m = np.stack([parent] * n)[None]
    mask = np.ones((1, n), dtype=np.float32)
    val = loss(leaves_to_array(Constants()), jnp.asarray(parent[None]), jnp.zeros(1),
               jnp.asarray(moves_m), jnp.zeros(1, dtype=jnp.int32), jnp.asarray(1.0),
               mask=jnp.asarray(mask), turns=jnp.zeros(1), tau=1.0,
               normalize_scores=True)
    assert np.isfinite(float(val))
    assert abs(float(val) - np.log(n)) < 0.2


def test_uci_search_replays_the_move_list():
    from kepler64.uci import parse_position, search

    board = parse_position("position startpos moves e2e4 d7d5 e4d5 g8f6".split(),
                           chess.Board())
    uci, _ = search(board, RocheEngine(Constants(), load_trained=False), 1, None)
    assert chess.Move.from_uci(uci) in board.legal_moves
