"""Learnable physical constants of the chess universe.

chess:  these are the "hyperparameters" of evaluation.
physics: G (gravitational constant), eps (Plummer softening), c (speed of
         light in squares/ply), roche (critical tidal-disruption parameter).
         All are leaves that gradient descent can move; c is prior-bounded.

Scale discipline (why the defaults are what they are):
  the eval must keep MATERIAL decisive. A captured rook must move the score
  more than any positional term can swing on a quiet move, otherwise the
  search cannot see a hanging piece (the measured failure mode: a rook
  "advance" scored +240 while capturing a free rook moved the score by +5,
  so the engine sacrificed edge rooks and pawns for advance-bonus noise).
  Every positional gain below is normalized so a large quiet-move swing is
  ~1-3 units while mat_gain * 5 (a rook) is ~10.
"""

from dataclasses import dataclass, fields, asdict
import json
import pathlib

import jax.numpy as jnp


@dataclass
class Constants:
    G: float = 1.0
    eps: float = 0.5  # Plummer softening length (prevents force singularities)
    c: float = 4.0  # speed of light, squares/ply (learned, prior-bounded)
    roche: float = 1.0  # critical eta / army-force threshold (learned)
    # Disruption force-sigmoid scale/gain (learnable, on-theme physics knobs).
    # The gauge is the ARMY force at the king (kings excluded as field sources:
    # the king is the disruption detector, not part of the attacking field).
    # Army force is ~0.001 in the opening, ~0.1-0.5 in a real attack, 2+ in a
    # mating net, so the sigmoid with roche=1.0 sits on its inflection region
    # only once an attack exists, instead of being saturated to +/-150 by the
    # enemy king's own distant pull (the old saturation bug).
    bonus: float = 300.0  # magnitude of the king-disruption force term
    kgain: float = 4.0  # sharpness of the disruption force-sigmoid
    # Binding-energy (cohesion) edge weight. ZERO by default since 2026-08-17:
    # the absolute cohesion gap (bind_b - bind_w) STRUCTURALLY rewards keeping
    # one's army packed and punishes development — starting from the packed
    # home rank, EVERY developing move unbinds your army (Nf3 costs ~2.4 units
    # while h3 costs ~0.9), so the depth-3 search preferred the flank shuffle
    # that kept the army bound. Cohesion as ADVANTAGE is already covered by the
    # move-sensitive inertia term. The term stays in the evaluation and gamma
    # stays a trainable leaf: training can learn its sign and scale from the
    # supervised signal instead of the hand-picked anti-development default.
    gamma: float = 0.0
    Rg: float = 1.0  # king radius of gyration (lattice units); dynamic under accretion
    # Reference tidal-stress scale.  The disruption index eta = Rg^3 * lambda1
    # / mref^2 (NOT / Mking^2): the King's own 1000-mass self-gravity must not
    # sit in the denominator, else eta ~ 1e-6 and the Roche limit is never
    # reachable.  mref is a FIXED unit scale (~ a minor piece), not a trained
    # weight: it puts eta in [~0.05 quiet .. ~1+ under a committed attack].
    mref: float = 3.5
    # ── Move-sensitivity & structural gains (the "delta" levers) ────────────
    # These reward what a MOVE *does* (child minus parent) rather than the
    # static look of a position, which is the only way to get a signal that
    # varies across sibling moves (see Kepler-64_Training_Thesis.md G1b and the
    # roadmap's "Every Signal Must Be Move-Sensitive" principle).
    #
    # REBALANCED 2026-08-17 after the sacrifice-bug diagnosis: the old
    # mean-rank CoM term with com_gain=40 scored a suicidal edge-rook charge at
    # +240 while a captured rook registered +5, and it DEGENERATED to zero
    # exactly when the enemy army was nearly gone (the 1/(sum+1e-9) fraction).
    # Every positional gain is now bounded per-move (~1-4 units) so material
    # (mat_gain=2: pawn=2, minor=6, rook=10, queen=18) stays the decisive
    # signal and the physics levers act as refinement, not override. All gains
    # remain trainable leaves — the training pipeline discovers their balance.
    lambda_delta: float = 2.0    # weight of the Δη = dη/dt tidal-rate term
    com_gain: float = 1.0        # momentum-flux (mass-weighted CoM advance) delta
    inertia_gain: float = 1.0    # attack/defense concentration delta
    entropy_gain: float = 4.0    # coordination-vs-scatter delta
    # Verlet tidal-drift gain: weight of the "impending collapse" term. A short
    # symplectic (Leapfrog) rollout projects each King's continuous coordinate
    # forward under the OPPONENT'S ARMY field and reads the change in tidal
    # stress at the projected location vs now (dη over the horizon). It is the
    # physics-native "threat a few plies out" signal a static eval cannot see.
    lambda_drift: float = 1.0
    # Gravitational material edge: you command more matter -> stronger field.
    # mat_gain=2: a captured rook moves the score by ~10, a queen by ~18 —
    # decisively above the positional noise floor (~1-4), which is the fix for
    # the measured "give away free rooks/pieces on the a/h files" behaviour.
    # Trainable: gradient descent finds the balance instead of us fixing it.
    mat_gain: float = 2.0

    def c_prior(self, lam_fast: float = 0.1, lam_slow: float = 0.1):
        """Monotonicity prior: keep c in [1.0, 10.0] with a sweet spot ~3-6.

        jnp.maximum (not Python max) so the prior is differentiable through
        jax.grad when c is a traced constant during training. Without this,
        gradient descent pushes c -> inf (instantaneous gravity is always
        easier to optimize) and the retarded-potential story collapses.
        """
        return -lam_fast * jnp.maximum(0.0, 2.0 - self.c) - lam_slow * jnp.maximum(0.0, self.c - 10.0)


# Trainable leaf names in params-array order (single source of truth shared by
# the trainer's _to_arr/_from_arr and by JSON persistence).
TRAINABLE_LEAVES = (
    "G", "eps", "c", "roche", "bonus", "kgain", "gamma", "Rg",
    "mat_gain", "lambda_delta", "com_gain", "inertia_gain",
    "entropy_gain", "lambda_drift",
)

# Physical projection bounds per leaf (same order as TRAINABLE_LEAVES).
LEAF_LO = (0.01, 0.01, 1.0, 0.05, 0.01, 0.01, 0.0, 0.1, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0)
LEAF_HI = (50.0, 20.0, 10.0, 20.0, 500., 50.0, 50., 10.0, 10.0,
           10.0, 10.0, 10.0, 10.0, 10.0)


def leaves_to_array(c: "Constants"):
    """Pack the 14 trainable leaves into a float32 array (TRAINABLE_LEAVES order)."""
    return jnp.array([getattr(c, name) for name in TRAINABLE_LEAVES],
                     dtype=jnp.float32)


def array_to_leaves(a) -> "Constants":
    """Unpack a leaf array, projecting into the physical bounds."""
    lo, hi = (jnp.asarray(LEAF_LO, dtype=jnp.float32),
              jnp.asarray(LEAF_HI, dtype=jnp.float32))
    a = jnp.clip(jnp.asarray(a, dtype=jnp.float32), lo, hi)
    return Constants(**{name: float(a[i]) for i, name in enumerate(TRAINABLE_LEAVES)})


# JIT-traceable bound arrays and projection: the loss clips the traced param
# vector with THESE bounds (same single source as array_to_leaves), so the
# optimizer, the loss and persistence can never drift apart.
LEAF_LO_F = jnp.asarray(LEAF_LO, dtype=jnp.float32)
LEAF_HI_F = jnp.asarray(LEAF_HI, dtype=jnp.float32)


def clip_leaves_traced(a):
    """Project a traced leaf array into the physical bounds (jit/grad-safe)."""
    return jnp.clip(jnp.asarray(a, dtype=jnp.float32), LEAF_LO_F, LEAF_HI_F)

# The trained-constants artifact the pipeline writes and the engine loads.
TRAINED_CONSTANTS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "training" / "trained_constants.json"
)


def save_constants(c: "Constants", path=None, meta: dict | None = None) -> pathlib.Path:
    """Persist the full learnable state (all TRAINABLE_LEAVES + mref + provenance).

    `meta` (optional) is stored under the "_meta" key for provenance — round
    number, validation metrics, match record. Loaders ignore unknown keys.
    """
    path = pathlib.Path(path) if path else TRAINED_CONSTANTS_PATH
    payload = {"mref": float(c.mref)}
    for name in TRAINABLE_LEAVES:
        payload[name] = float(getattr(c, name))
    if meta:
        payload["_meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_constants(path=None, default: "Constants | None" = None) -> "Constants":
    """Load a persisted constants artifact, merging into defaults.

    Old artifacts that predate newer leaves load cleanly: every missing leaf
    falls back to the default. Returns None when the file does not exist, so
    callers fall back to the pristine universe (no stale constants).
    """
    path = pathlib.Path(path) if path else TRAINED_CONSTANTS_PATH
    base = default or Constants()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    kwargs = {}
    for name in fields(Constants):
        if name.name in data:
            try:
                kwargs[name.name] = float(data[name.name])
            except (TypeError, ValueError):
                continue
    return type(base)(**{**asdict(base), **kwargs})


# Standard piece masses (chess material, repurposed as gravitational mass).
# King is intentionally enormous so its self-gravity dominates local pawns.
PIECE_MASSES = {
    "P": 1.0,
    "N": 3.0,
    "B": 3.0,
    "R": 5.0,
    "Q": 9.0,
    "K": 1000.0,
}
