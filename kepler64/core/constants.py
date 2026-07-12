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
