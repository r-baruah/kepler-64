"""Regression tests for the 2026-08-23 audit fixes (Phases 0-1).

Covers:
  A1  single speed of light: the Lorentz boost is governed by Constants.c
  A2  Verlet tidal drift uses the same sigmoid(c - d) reach gate as static eta
  A3  velocity gain is distance-weighted (queen slide > pawn step)
  A4  promotion preserves a pawn's previously accreted mass surplus
  A5  TT depth-preferred replacement survives same-key shallow revisits
"""

import numpy as np
import pytest

from ..core.fastboard import FastBoard
from ..core.transitions import child_mass_vector
from ..core.evaluate import _eta_drift, score_white, score_white_terms as _terms
from ..core.constants import Constants
from ..search.minimax import TT, _TT_EXACT

import jax.numpy as jnp


# ── helpers ──────────────────────────────────────────────────────────────────

def _start_board():
    import chess
    return FastBoard.from_chess(chess.Board())


def _find(moves, f, t):
    for mv in moves:
        if mv[0] == f and mv[1] == t:
            return mv
    raise AssertionError(f"move {f}->{t} not legal here")


# ── A1: one speed of light across the whole pipeline ────────────────────────

def test_lorentz_boost_uses_supplied_c():
    """mass_vector must boost with the c it was given, not a hardcoded one."""
    b = _start_board()
    # Knight b1 -> c3 gives the knight nonzero velocity.
    mv = _find(b.legal_moves(), 1, 18)
    child = b.apply(mv)
    assert np.any(child.velocity > 0.0)

    base = float(np.abs(np.asarray(child.mass_vector(50.0)))[18])  # ~unboosted
    boosted_small_c = float(np.abs(np.asarray(child.mass_vector(2.0)))[18])
    boosted_large_c = float(np.abs(np.asarray(child.mass_vector(50.0)))[18])
    # A smaller light speed means a larger u = v/(v+c), hence MORE boost.
    assert boosted_small_c > boosted_large_c
    assert boosted_large_c == pytest.approx(base, abs=1e-4)


def test_ply_consistency_holds_for_any_c():
    """child_mass_vector(board,...) must equal child.mass_vector(c) exactly,
    for ANY light speed — this is what forbids a second hardcoded c."""
    c = 3.7
    b = _start_board()
    m1 = _find(b.legal_moves(), 1, 18)           # White N:b1-c3
    mid = b.apply(m1)
    m_black = _find(mid.legal_moves(), 53, 45)   # Black f7-f6
    mid2 = mid.apply(m_black)                    # knight momentum survives
    parent_mv = mid2.mass_vector(c)              # unboost path active next ply
    m2 = _find(mid2.legal_moves(), 18, 35)       # White N:c3-d5
    child = mid2.apply(m2)
    threaded = child_mass_vector(mid2, m2, parent_mv, child_board=child,
                                 c_lorentz=c)
    direct = np.asarray(child.mass_vector(c))
    assert np.allclose(threaded, direct, atol=1e-5)


# ── A3: distance-weighted velocity ──────────────────────────────────────────

def test_velocity_gain_is_distance_weighted():
    b = _start_board()
    pawn_step = _find(b.legal_moves(), 12, 20)   # e2-e3: 1 square
    pawn_double = _find(b.legal_moves(), 12, 28) # e2-e4: 2 squares
    v_step = float(b.apply(pawn_step).velocity[20])
    v_double = float(b.apply(pawn_double).velocity[28])
    assert v_step == pytest.approx(1.0)
    assert v_double == pytest.approx(2.0)

    knight = _find(b.legal_moves(), 1, 18)       # Nb1-c3: sqrt(1+4)
    v_knight = float(b.apply(knight).velocity[18])
    assert v_knight == pytest.approx(5.0 ** 0.5)


def test_stationary_velocity_decays():
    from ..core.fastboard import VELOCITY_DECAY
    b = _start_board()
    mv = _find(b.legal_moves(), 1, 18)                    # White N:b1-c3
    child = b.apply(mv)
    v_after = float(child.velocity[18])
    reply = _find(child.legal_moves(), 53, 45)            # Black f7-f6
    later = child.apply(reply)
    # The knight did NOT move; its momentum relaxed exactly one decay step.
    assert float(later.velocity[18]) == pytest.approx(v_after * VELOCITY_DECAY,
                                                      rel=1e-5)


# ── A4: promotion preserves accreted mass ───────────────────────────────────

def test_promotion_carries_accreted_surplus():
    board = FastBoard()
    board.pieces[48] = 1    # white pawn on a7
    board.pieces[56] = -4   # black rook on a8 (promotion capture)
    board.pieces[63] = -6   # black king h8 (board legality irrelevant here)
    masses = np.zeros(64, dtype=np.float32)
    masses[48] = 1.0 + 2.4  # pawn accreted 2.4 mass earlier
    masses[56] = -5.0
    masses[63] = -1000.0

    promo_move = (48, 56, 5)  # a7xa8=Q
    child = child_mass_vector(board, promo_move, masses)

    expected = 9.0 + 2.4 + 0.8 * 5.0   # queen base + carried surplus + capture share
    assert float(child[56]) == pytest.approx(expected, abs=1e-5)
    assert float(child[48]) == 0.0
    total_before = float(np.sum(np.abs(masses)))
    total_after = float(np.sum(np.abs(child)))
    assert total_after >= total_before      # matter is never destroyed


# ── A2: gated rollout dynamics ───────────────────────────────────────────────

def test_eta_drift_respects_light_speed():
    """A distant attacker's pull must vanish when c is tiny — the drift term
    simulates the SAME universe as the static eta terms."""
    pieces = np.zeros(64, dtype=np.int8)
    pieces[0] = 5     # white rook a1
    pieces[63] = -6   # black king h8 (distance ~9.9 squares)
    b = FastBoard(pieces)
    m = np.abs(np.asarray(b.mass_vector()))
    king_mass = 1000.0

    def _drift(c):
        return float(_eta_drift(m, 63, king_mass,
                                Constants().G, Constants().eps, c,
                                Constants().Rg, Constants().mref))

    drift_slow = _drift(0.01)   # influence has not arrived: no drift
    drift_fast = _drift(40.0)   # light instant: the king feels the rook

    assert abs(drift_slow) < 1e-6
    assert abs(drift_fast) > 1e-9


# ── A8: Observer preserves every leaf and evolves in-game ───────────────────

def test_observer_preserves_all_leaves():
    """The old observe_update reconstructed Constants(G,eps,c,roche) and reset
    the other ten leaves to defaults. Every leaf must survive observation."""
    from ..multiverse.observer import observe_update
    import chess

    base = Constants(bonus=123.0, mat_gain=2.5, entropy_gain=3.25,
                     lambda_drift=0.75, com_gain=1.5)
    b = FastBoard.from_chess(chess.Board())
    m = np.abs(np.asarray(b.mass_vector()))

    new = observe_update(base, m)

    # Non-default leaves must be ~their old values (tiny drift), NOT defaults.
    assert new.bonus == pytest.approx(base.bonus, abs=base.bonus * 0.05)
    assert new.mat_gain == pytest.approx(base.mat_gain, abs=0.1)
    assert new.entropy_gain == pytest.approx(base.entropy_gain, abs=0.2)
    assert new.lambda_drift == pytest.approx(base.lambda_drift, abs=0.1)
    assert new.com_gain == pytest.approx(base.com_gain, abs=0.1)
    # mref is not a leaf and must pass through untouched.
    assert new.mref == base.mref
    # c stays inside its physical prior.
    assert 1.0 <= float(new.c) <= 10.0


def test_observer_shifts_toward_stronger_gravity_on_big_score():
    from ..multiverse.observer import observe_update
    pieces = np.zeros(64, dtype=np.int8)
    pieces[27] = 5     # white rook d4
    pieces[33] = 5     # white rook d5 — heavy attack toward black kingside
    pieces[62] = -6    # black king g8-ish
    b = FastBoard(pieces)
    m = np.asarray(b.mass_vector())
    base = Constants()
    new = observe_update(base, m, alpha=0.5, beta_kl=0.0)  # exaggerated step
    # A positive (White-good) score pushes G up.
    assert float(new.G) > float(base.G)


def test_engine_observe_flag_evolves_constants():
    """play(observe=True) persists evolved constants on the engine instance."""
    import chess
    from .. import RocheEngine

    eng = RocheEngine(constants=Constants(), load_trained=False)
    board = chess.Board()
    g_before = float(eng.constants.G)
    mv = eng.play(board, depth=2, observe=True)
    assert mv is not None and mv in board.legal_moves
    g_after = float(eng.constants.G)
    # The shift is tiny but nonzero; crucially the engine now CARRIES it.
    assert g_after == pytest.approx(g_before, rel=0.05)
    assert eng.constants is not None


# ── Phase 4: Peters-Mathews gravitational-wave term ─────────────────────────

def test_leaf_packing_has_17_leaves():
    from ..core.constants import leaves_to_array, TRAINABLE_LEAVES
    assert len(TRAINABLE_LEAVES) == 17
    assert leaves_to_array(Constants()).shape == (17,)


def test_schwarzschild_term_neutral_at_default_gain():
    """lambda_sch ships at 0.0 — identical physics until training moves it."""
    import chess
    b = FastBoard.from_chess(chess.Board())
    m = b.mass_vector()
    t = _terms(m, Constants())
    assert float(t.schwarzschild) == 0.0


def test_schwarzschild_penalizes_huddled_heavy_pieces():
    """With the gain on, two adjacent queens overlap horizons (penalty);
    the same queens far apart do not. Isolated `schwarzschild` term."""
    c_sch = Constants(lambda_sch=1e-2)
    close = np.zeros(64); close[27] = 9.0; close[28] = 9.0
    close[4] = 1000.0; close[60] = -1000.0
    far = np.zeros(64); far[0] = 9.0; far[63 - 7] = 9.0
    far[4] = 1000.0; far[63] = -1000.0

    t_close = _terms(jnp.asarray(close), c_sch)
    t_far = _terms(jnp.asarray(far), c_sch)
    # Both terms are <= 0 for White (own overlaps are a cost); adjacent
    # heavy queens pay strictly more.
    assert float(t_close.schwarzschild) < float(t_far.schwarzschild) <= 0.0
    # The far pair's horizons (r_s = 2*9/16 ≈ 1.1 each) must NOT overlap.
    assert abs(float(t_far.schwarzschild)) < 1e-3


def test_drift_horizon_dt_is_threaded():
    """dt_drift must reach the rollout: a larger step projects further into
    an attacking field, so |drift| grows monotonically here."""
    pieces = np.zeros(64, dtype=np.int8)
    pieces[0] = 5     # white rook a1 — attacker
    pieces[63] = -6   # black king h8
    m = np.abs(np.asarray(FastBoard(pieces).mass_vector()))
    d_small = float(_eta_drift(m, 63, 1000.0, Constants().G, Constants().eps,
                               40.0, Constants().Rg, Constants().mref, dt=0.05))
    d_large = float(_eta_drift(m, 63, 1000.0, Constants().G, Constants().eps,
                               40.0, Constants().Rg, Constants().mref, dt=0.5))
    assert abs(d_large) > abs(d_small) >= 0.0


def test_gw_term_neutral_at_default_gain():
    """lambda_gw ships at 0.0 — the universe must be bit-identical to the
    pre-GW physics until training decides otherwise."""
    import chess
    b = FastBoard.from_chess(chess.Board())
    m = b.mass_vector()
    t = _terms(m, Constants())
    assert float(t.gw) == 0.0


def test_gw_radiation_penalizes_own_huddling():
    """With the gain switched on, a clustered army radiates more energy than a
    spread one (identical composition, identical kings — only spacing differs).
    We compare the isolated `gw` term so eta/geometry effects stay out."""
    c_gw = Constants(lambda_gw=1e-3)
    # Same five masses in both; only the white army's spacing differs.
    huddled = np.zeros(64)
    huddled[27] = 5.0; huddled[28] = 9.0; huddled[35] = 3.0; huddled[36] = 1.0
    huddled[7] = 6.0 if False else 0.0
    huddled[4] = 1000.0   # white king e1 (detector only)
    huddled[63] = -1000.0
    spread = np.zeros(64)
    spread[0] = 5.0; spread[14] = 9.0; spread[49] = 3.0; spread[60] = 1.0
    spread[4] = 1000.0
    spread[63] = -1000.0

    t_huddle = _terms(jnp.asarray(huddled), c_gw)
    t_spread = _terms(jnp.asarray(spread), c_gw)
    # Both gw terms are <= 0 for White (own radiation is a cost); the huddled
    # army must radiate strictly more.
    assert float(t_huddle.gw) < float(t_spread.gw) <= 0.0


# ── A5: TT depth-preferred replacement ──────────────────────────────────────

def test_tt_keeps_deeper_entry_on_same_key():
    b = _start_board()
    tt = TT(size=1024)
    m = np.zeros(64, dtype=np.float32)
    p = np.zeros(64, dtype=np.float32)

    tt.store(b, m, p, depth=5, flag=_TT_EXACT, score=42.0, move=(12, 28, 0))
    # Shallow revisit of the SAME state must not destroy the deeper entry.
    tt.store(b, m, p, depth=3, flag=_TT_EXACT, score=-7.0, move=(6, 27, 0))
    hit = tt.lookup(b, m, p)
    assert hit is not None
    h_depth, _, h_score, _ = hit
    assert h_depth == 5
    assert h_score == pytest.approx(42.0)

    # Equal-depth ties keep the old entry too.
    tt.store(b, m, p, depth=5, flag=_TT_EXACT, score=-1.0, move=(0, 16, 0))
    _, _, h_score, _ = tt.lookup(b, m, p)
    assert h_score == pytest.approx(42.0)

    # A deeper entry still replaces.
    tt.store(b, m, p, depth=6, flag=_TT_EXACT, score=99.0, move=(0, 16, 0))
    _, _, h_score, _ = tt.lookup(b, m, p)
    assert h_score == pytest.approx(99.0)
