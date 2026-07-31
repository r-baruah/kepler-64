"""Mass-state transitions shared by search, training, and analysis."""

import numpy as np

from .fastboard import _MASS_LUT

_ACCRETION = 0.8


def child_mass_vector(board, move, parent_masses, accretion: float = _ACCRETION):
    """Apply a legal move to a possibly accretion-adjusted mass vector.

    Pure-numpy: no JAX device round-trips, so it is safe to call per child
    inside the training-data hot loop. `parent_masses` may be a numpy array or
    a JAX array; the result is a numpy (64,) float32 vector.

    FastBoard stores piece identities, while the mass vector can carry mass
    accumulated on earlier captures. Updating the vector directly preserves
    that state across quiet moves and makes every data/search path agree.
    """
    from_sq, to_sq, promotion = (int(move[0]), int(move[1]), int(move[2]))
    masses = np.asarray(parent_masses, dtype=np.float32)
    moving = masses[from_sq]
    sign = 1.0 if moving >= 0.0 else -1.0

    captured_sq = to_sq
    piece = int(board.pieces[from_sq])
    if abs(piece) == 1 and to_sq == board.ep and int(board.pieces[to_sq]) == 0:
        captured_sq = to_sq - (8 if piece > 0 else -8)

    captured = masses[captured_sq]
    moved_mass = moving + sign * accretion * abs(captured)
    if promotion:
        moved_mass = sign * float(_MASS_LUT[promotion]) + sign * accretion * abs(captured)

    child = masses.copy()
    child[from_sq] = 0.0
    child[captured_sq] = 0.0
    child[to_sq] = moved_mass

    # Castling also moves the rook and must preserve any rook accretion.
    if abs(piece) == 6 and abs(to_sq - from_sq) == 2:
        rook_from = 7 if to_sq > from_sq else 0
        rook_to = 5 if to_sq > from_sq else 3
        if piece < 0:
            rook_from += 56
            rook_to += 56
        child[rook_to] = masses[rook_from]
        child[rook_from] = 0.0
    return child
