"""Learnable physical constants of the chess universe.

chess:  these are the "hyperparameters" of evaluation.
physics: G (gravitational constant), eps (Plummer softening), c (speed of
         light in squares/ply), roche (critical tidal-disruption parameter).
         All are leaves that gradient descent can move; c is prior-bounded.
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Constants:
    G: float = 1.0
    eps: float = 0.5  # Plummer softening length (prevents force singularities)
    c: float = 4.0  # speed of light, squares/ply (learned, prior-bounded)
    roche: float = 1.0  # critical eta: init at 1.0 so learning is interpretable
    # Disruption force-sigmoid scale/gain (learnable, on-theme physics knobs):
    bonus: float = 50.0  # magnitude of the king-disruption force term
    kgain: float = 4.0  # sharpness of the disruption force-sigmoid
    gamma: float = 0.25  # weight of the global field-energy material edge
    Rg: float = 1.0  # king radius of gyration (lattice units); becomes dynamic under accretion/Lorentz
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
    lambda_delta: float = 1.0   # weight of the Δη = dη/dt tidal-rate term
    com_gain: float = 1.0        # center-of-mass advance/aggression delta
    inertia_gain: float = 0.01   # attack/defense concentration delta
    entropy_gain: float = 0.5    # coordination-vs-scatter delta
    # Gravitational material edge: you command more mass -> stronger field.
    # On-theme (mass == matter) and gives the search a clean capture/advantage
    # gradient so it plays real chess instead of drifting to a flat equilibrium.
    # Trainable scale (unfrozen): lets gradient descent find the balance between
    # material and the tidal/disruption terms instead of us hard-coding it.
    mat_gain: float = 0.3

    def c_prior(self, lam_fast: float = 0.1, lam_slow: float = 0.1):
        """Monotonicity prior: keep c in [1.0, 10.0] with a sweet spot ~3-6.

        jnp.maximum (not Python max) so the prior is differentiable through
        jax.grad when c is a traced constant during training. Without this,
        gradient descent pushes c -> inf (instantaneous gravity is always
        easier to optimize) and the retarded-potential story collapses.
        """
        return -lam_fast * jnp.maximum(0.0, 2.0 - self.c) - lam_slow * jnp.maximum(0.0, self.c - 10.0)


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
