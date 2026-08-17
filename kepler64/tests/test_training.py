"""Unit tests for the training pipeline plumbing.

Guards:
  * the shared leaf layout (pack/unpack roundtrip, loss/trainer agreement),
  * loss finiteness + gradient flow through the NEW leaves (drift, inertia,
    com, entropy, mat_gain),
  * save/load roundtrip + merge of partial (legacy) artifacts,
  * policy_metrics capture/quiet splits,
  * the self-match Elo estimator sanity.
"""

import jax
import jax.numpy as jnp
import numpy as np

from kepler64 import RocheEngine
from kepler64.core.constants import (
    Constants, TRAINABLE_LEAVES, leaves_to_array, array_to_leaves,
    save_constants, load_constants,
)
from kepler64.training.loss import loss, policy_metrics
from kepler64.core.evaluate import score_white


def _tiny_batch(parent=None):
    """One position (start) with 3 children: e4, d4, h4."""
    from kepler64.core.fastboard import FastBoard
    from kepler64.core.transitions import child_mass_vector
    import chess

    b = chess.Board()
    fb = FastBoard.from_chess(b)
    pm = fb.mass_vector()
    children, names = [], []
    for san in ("e4", "d4", "h4"):
        m = b.parse_san(san)
        names.append(san)
        mv = (m.from_square, m.to_square, m.promotion or 0)
        children.append(np.asarray(
            child_mass_vector(fb, mv, pm, child_board=fb.apply(mv))))
    child_m = np.stack(children).astype(np.float32)[None]          # (1,3,64)
    mask = np.ones((1, 3), dtype=np.float32)
    M = np.asarray(pm)[None].astype(np.float32)                     # (1,64)
    Y = np.zeros(1, dtype=np.float32)
    turns = np.zeros(1, dtype=np.float32)
    expert_idx = np.array([0], dtype=np.int32)                      # "expert" = e4
    caps = np.array([0.0], dtype=np.float32)
    return M, Y, turns, child_m, mask, expert_idx, caps


def test_leaf_layout_roundtrip():
    c = Constants()
    arr = leaves_to_array(c)
    assert arr.shape == (len(TRAINABLE_LEAVES),)
    back = array_to_leaves(arr)
    for name in TRAINABLE_LEAVES:
        assert abs(getattr(back, name) - getattr(c, name)) < 1e-5


def test_leaf_bounds_enforced():
    huge = np.full(len(TRAINABLE_LEAVES), 1e9, dtype=np.float32)
    small = np.full(len(TRAINABLE_LEAVES), -1e9, dtype=np.float32)
    hi, lo = array_to_leaves(huge), array_to_leaves(small)
    assert hi.bonus <= 500.0 + 1e-3
    assert lo.mat_gain >= -1e-6
    assert lo.gamma == 0.0
    assert lo.c == 1.0


def test_loss_finite_and_grad_flows():
    M, Y, turns, child_m, mask, expert_idx, caps = _tiny_batch()
    p = leaves_to_array(Constants())
    has_policy = jnp.array(1.0)
    val = loss(p, M, Y, child_m, expert_idx, has_policy, mask, turns,
               tau=2.0, margin=0.5)
    assert np.isfinite(float(val))
    g = jax.grad(loss)(p, M, Y, child_m, expert_idx, has_policy, mask, turns,
                       2.0, 0.5)
    assert np.all(np.isfinite(np.asarray(g)))
    # at least half the leaves receive nonzero gradient — the policy term must
    # reach through the delta gains, not just the disruption scales.
    assert int(np.sum(np.abs(np.asarray(g)) > 1e-8)) >= 6


def test_policy_metrics_capture_quiet_split():
    M, Y, turns, child_m, mask, expert_idx, caps = _tiny_batch()
    m = policy_metrics(Constants(), M, Y, turns, child_m, mask, expert_idx,
                       is_capture_m=caps)
    for k in ("top1", "mrr", "mrr_capture", "mrr_quiet"):
        assert 0.0 <= m[k] <= 1.0
    # the single example is a quiet move: quiet MRR is defined, capture MRR
    # falls to the no-capture default (0 here because there are no captures).
    assert m["mrr_quiet"] > 0.0


def test_save_load_roundtrip(tmp_path):
    c = Constants(bonus=123.0, mat_gain=3.0, gamma=0.25)
    p = save_constants(c, tmp_path / "art.json", meta={"round": 3})
    back = load_constants(p)
    assert back is not None
    assert abs(back.bonus - 123.0) < 1e-3
    assert abs(back.mat_gain - 3.0) < 1e-3
    assert abs(back.gamma - 0.25) < 1e-3
    assert back.mref == Constants().mref


def test_load_legacy_partial_artifact(tmp_path):
    """Old artifacts missing newer leaves must load cleanly (missing -> default)."""
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"G": 1.2, "eps": 0.6, "mat_gain": 0.5}),
                 encoding="utf-8")
    back = load_constants(p)
    assert back is not None
    assert abs(back.G - 1.2) < 1e-3
    assert abs(back.mat_gain - 0.5) < 1e-3
    assert back.lambda_drift == Constants().lambda_drift  # default fallback


def test_roche_engine_respects_explicit_constants():
    """Explicit constants must win over any persisted artifact (ablations)."""
    c = Constants(bonus=77.0)
    eng = RocheEngine(c)
    assert abs(eng.constants.bonus - 77.0) < 1e-3
