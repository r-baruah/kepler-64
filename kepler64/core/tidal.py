"""Tidal tensor, closed-form 2x2 eigenvalues, and the Roche/tidal parameter.

chess:  the enemy King "loses" when the position structurally rips apart along
        its principal axis of stress.
physics: the tidal tensor is the Hessian (gradient of the gradient) of the
         gravitational potential at the King's coordinate. Its largest
         eigenvalue lambda1 is the principal stretching rate. Collapse when the
         dimensionless tidal-disruption parameter eta = lambda1 * Rg^3 /
         mref^2 exceeds the learned critical value (roche).
"""

import jax.numpy as jnp
from .gravity import potential_field, _COORDS


def eig2x2(A: "jnp.ndarray"):
    """Closed-form eigenvalues of a symmetric 2x2 matrix. ~10 FLOPs, no LAPACK.

    lambda = tr/2 +/- sqrt((tr/2)^2 - det); argument clamped >= 0 for stability.
    """
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    s = jnp.sqrt(jnp.maximum(tr * tr / 4.0 - det, 1e-12))
    return tr / 2.0 + s, tr / 2.0 - s  # (lambda1 >= lambda2)


def tidal_tensor_at(U: "jnp.ndarray", king_sq):
    """Hessian of the potential field at the King's square (JAX-traceable).

    The potential is padded by 1 on every side (edge mode) so central
    second-difference stencils are valid even when the King sits on rank 1 or
    8 / file a or h, where a naive stencil would collapse to a first derivative
    and break the win condition for ~25% of King positions.

    Coordinate convention (consistent with gravity._COORDS): _COORDS[sq] =
    (file, rank), so the tensor's index 0 is the FILE axis and index 1 is the
    RANK axis. The second-difference along the rank direction lives at [1,1]
    and along the file direction at [0,0]. (An earlier version had these two
    diagonals swapped; the dominant eigenvalue is invariant under the swap, so
    scores were numerically unaffected, but the off-diagonal eigenvectors were
    implicitly mis-labeled. This ordering is now correct for any future
    eigenvector-based feature.)
    """
    r = king_sq // 8
    f = king_sq % 8
    Ug = U.reshape(8, 8)
    Up = jnp.pad(Ug, 1, mode="edge")  # (10, 10)
    R = r + 1
    F = f + 1
    u_file = Up[R, F + 1] - 2 * Up[R, F] + Up[R, F - 1]   # d²U/d(file)²   -> [0,0]
    u_rank = Up[R + 1, F] - 2 * Up[R, F] + Up[R - 1, F]   # d²U/d(rank)²   -> [1,1]
    uxy = (Up[R + 1, F + 1] - Up[R + 1, F - 1] - Up[R - 1, F + 1] + Up[R - 1, F - 1]) / 4.0
    return jnp.array([[u_file, uxy], [uxy, u_rank]], dtype=jnp.float32)


def tidal_disruption(
    masses: "jnp.ndarray",
    king_sq: int,
    constants,
    Rg: float = 1.0,
):
    """Return (eta, lambda1, lam2). eta > constants.roche => King collapses."""
    c_val = getattr(constants, "c", 10.0)
    U = potential_field(masses, constants.eps, constants.G, c_val)
    A = tidal_tensor_at(U, king_sq)
    lam1, lam2 = eig2x2(A)
    mref = getattr(constants, "mref", 3.5)
    eta = lam1 * (Rg**3) / (mref**2 + 1e-9)
    return eta, lam1, lam2
