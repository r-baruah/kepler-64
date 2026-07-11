"""Symplectic (Leapfrog) forward simulation of the King's trajectory.

chess:  instead of a deep Minimax search, project whether the enemy King will
        cross its Roche threshold a few plies from now.
physics: a symplectic integrator conserves energy, so a short fixed-dt Leapfrog
        rollout honestly projects the King's continuous coordinate under the
        local tidal force without drift. The rollout is differentiable (lax.scan).

Efficiency: only the force at the King's *current* continuous position is needed
each step (an O(64) gather-sum), not the full (64,64) field — so the rollout is
64x cheaper than recomputing the whole board field per step.
"""

import jax
import jax.numpy as jnp

from .gravity import _COORDS


def force_at(pos, masses, constants):
    """Gravitational acceleration at an arbitrary continuous position."""
    d = _COORDS - pos  # (64, 2)
    r2 = jnp.sum(d**2, axis=-1) + constants.eps**2
    inv = masses / (r2 * jnp.sqrt(r2))
    return constants.G * jnp.einsum("i,ij->j", inv, d)


@jax.jit
def rollout(masses, king_sq, constants, steps: int = 4, dt: float = 0.1):
    """Project the King coordinate forward under the local force field.

    Returns (final_pos, trajectory). Used to detect impending collapse without
    deep search.
    """
    abs_m = jnp.abs(masses)
    pos = _COORDS[int(king_sq)].astype(jnp.float32)
    vel = jnp.zeros(2, dtype=jnp.float32)

    def step(carry, _):
        pos, vel = carry
        F = force_at(pos, abs_m, constants)
        vel = vel + F * dt
        pos = pos + vel * dt
        return (pos, vel), pos

    (pos, vel), traj = jax.lax.scan(step, (pos, vel), None, length=steps)
    return pos, traj
