"""2C Accretion — captured mass is not deleted, it is absorbed.

chess:  captures make you heavier but more structurally fragile — a real reason
        to capture beyond material, and it conserves mass (fixing the broken
        energy-conservation blunder metric).
physics: m_new = m_captor + eta_acc * m_captured. The capturing piece grows
        (Eddington-style) but its effective radius grows too, so its tidal
        eigenvalue rises — over-extended pieces become disruptable by a pawn.
"""

import jax.numpy as jnp


def apply_capture(masses: "jnp.ndarray", captor_sq: int, captured_sq: int, eta_acc: float = 0.8):
    """Return a new mass vector with the captured mass accreted onto the captor."""
    m_cap = float(masses[captured_sq])
    m_new = float(masses[captor_sq]) + eta_acc * m_cap
    return masses.at[captor_sq].set(m_new).at[captured_sq].set(0.0)
