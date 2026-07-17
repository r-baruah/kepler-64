"""2C Accretion — captured mass is not deleted, it is absorbed.

chess:  captures make you heavier but more structurally fragile — a real reason
        to capture beyond material, and it conserves mass (fixing the broken
        energy-conservation blunder metric).
physics: m_new = m_captor + eta_acc * m_captured. The capturing piece grows
        (Eddington-style) but its effective radius grows too, so its tidal
        eigenvalue rises — over-extended pieces become disruptable by a pawn.
"""

import jax.numpy as jnp


def apply_capture(masses: "jnp.ndarray", captor_sq: int, captured_sq: int,
                  eta_acc: float = 0.8, Rg_old: float = 1.0):
    """Return (masses_new, Rg_new) after accreting captured mass onto the captor.

    The captor's mass grows, so its king radius of gyration grows with the
    cube-root of mass: Rg_new = Rg_old * (m_new / m_old) ** (1/3). This is the
    Rg consumed by `core.evaluate._eta` / `core.tidal`, so it MUST be threaded
    back to the caller — inflation of Rg is what makes an over-extended piece
    disruptable. Uses jnp ops throughout so it stays JAX-traceable (no python
    int on a traced value).
    """
    m_old = masses[captor_sq]
    m_cap = masses[captured_sq]
    m_new = m_old + eta_acc * m_cap
    masses_new = masses.at[captor_sq].set(m_new).at[captured_sq].set(0.0)
    # Guard m_old==0 (degenerate) so the cube-root stays finite under tracing.
    ratio = jnp.where(m_old > 0.0, m_new / m_old, 1.0)
    Rg_new = Rg_old * jnp.power(ratio, 1.0 / 3.0)
    return masses_new, Rg_new
