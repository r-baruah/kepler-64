"""Lorentz mass escalation — relativistic mass from move velocity.

chess:  moving the same piece repeatedly inflates its mass and warps the local
        tidal tensor: a real anti-repetition heuristic disguised as relativity.
physics: gamma = 1/sqrt(1 - u^2), u = squares_moved / (squares_moved + c) in
        (0,1) so gamma in (1, inf) and never hits the v > c singularity.
        Lorentz mass = own mass; accretion mass (multiverse/accretion) = stolen
        mass. They pair: fast + greedy pieces become supermassive and fragile.

STATUS — IMPLEMENTED BUT NOT WIRED IN:
    `boost_masses` / `lorentz_mass` are complete and JAX-traceable, but NO
    per-piece velocity accumulator exists in `core.board.py` (Board.mass_vector
    returns only static signed masses). Because there is no clean velocity hook
    to feed, these functions are intentionally INACTIVE: nothing calls them and
    they are deliberately NOT injected into mass_vector / _score_body. Do not
    add a half-baked velocity hook; wire this only once a real velocity feed
    exists in the board state.
"""

import jax.numpy as jnp


def lorentz_factor(squares_moved: float, c: float) -> float:
    u = squares_moved / (squares_moved + max(float(c), 1e-3))
    return 1.0 / jnp.sqrt(1.0 - u * u)


def lorentz_mass(base_mass: float, squares_moved: float, c: float) -> float:
    return base_mass * lorentz_factor(squares_moved, c)


def boost_masses(masses, move_velocity, c):
    """Scale each piece's mass by its Lorentz factor given per-piece velocity."""
    u = move_velocity / (move_velocity + max(float(c), 1e-3))
    gamma = 1.0 / jnp.sqrt(1.0 - u * u)
    return masses * gamma
