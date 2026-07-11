"""Lorentz mass escalation — relativistic mass from move velocity.

chess:  moving the same piece repeatedly inflates its mass and warps the local
        tidal tensor: a real anti-repetition heuristic disguised as relativity.
physics: gamma = 1/sqrt(1 - u^2), u = squares_moved / (squares_moved + c) in
        (0,1) so gamma in (1, inf) and never hits the v > c singularity.
        Lorentz mass = own mass; accretion mass (multiverse/accretion) = stolen
        mass. They pair: fast + greedy pieces become supermassive and fragile.
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
