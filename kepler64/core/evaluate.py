"""Evaluation: physics (tidal disruption) + a soft, differentiable tactical layer.

chess:  the score tells the search which move keeps your King bound and the
        enemy King disrupted.
physics: Eval = +eta_enemy - eta_self + bonus(your force on enemy king)
                - penalty(enemy force on your king) + global field-energy edge.

        Disruption is SOURCE-ATTRIBUTED: a king is disrupted by the OPPONENT's
        masses, not by its own. This is the physically correct reading (a body's
        own gravity binds it; the enemy's gravity tears it) and it removes the
        old perverse incentive where capturing an enemy piece looked BAD because
        it lowered the total force on the enemy king.

        `roche` is the learned disruption threshold: the force-sigmoid flips
        around it, so gradient descent learns the tidal limit instead of us
        hand-picking it.

All heavy math is pure JAX. Two wrappers share one body:
  * _score_core_static  — constants are STATIC (compile-time). Fast inference /
    vmap / search; XLA folds G, eps, c into the kernel.
  * _score_core         — constants are TRACED. Used only by training so
    jax.grad flows into G, eps, c, roche.
"""

import jax
import jax.numpy as jnp

from .gravity import force_field, potential_field
from .tidal import tidal_tensor_at, eig2x2
from .constants import Constants

_MAX_MOVES = 218  # theoretical max legal moves in chess


def _king_idx(masses, sign: float):
    """Traced index of the king of the given color (sign +1 white, -1 black)."""
    mask = (jnp.abs(jnp.abs(masses) - 1000.0) < 0.5) & (jnp.sign(masses) == sign)
    return jnp.argmax(mask.astype(jnp.float32))


def _eta(U, king_sq, Rg: float = 1.0, mref: float = 3.5) -> float:
    """Tidal-disruption index at a king from the supplied potential field U.

    eta = Rg^3 * lambda1 / mref^2: the dominant tidal eigenvalue (tearing)
    scaled by the king's spatial extent (Rg) and a reference tidal-stress
    scale (mref).  G is intentionally ABSENT: lambda1 is already proportional
    to G (the tidal tensor is the Hessian of the potential), so including G in
    the denominator would cancel it and make eta independent of the field
    strength — physically the disruption criterion is G-independent, which is
    correct.  We divide by mref^2 (a minor-piece-scale constant), NOT by
    Mking^2: the King's own 1000-mass self-gravity would otherwise shrink eta
    to ~1e-6 and the Roche limit would never be reachable.

    Larger eta = closer to disruption.
    """
    A = tidal_tensor_at(U, king_sq)
    lam1, _ = eig2x2(A)
    return (Rg**3) * lam1 / (mref**2 + 1e-9)


def _score_body(masses, G: float, eps: float, c: float, roche: float,
                 bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                 Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 1.0):
    """Evaluation from White's perspective (positive = good for White)."""
    abs_m = jnp.abs(masses)
    white_m = jnp.where(masses > 0.0, abs_m, 0.0)   # your masses
    black_m = jnp.where(masses < 0.0, abs_m, 0.0)   # enemy masses

    # Source-attributed force & potential: white's field vs black's field.
    F_w = force_field(white_m, eps, G, c)   # force everywhere due to YOUR masses
    F_b = force_field(black_m, eps, G, c)   # force everywhere due to ENEMY masses
    U_w = potential_field(white_m, eps, G, c)  # your potential (tears their king)
    U_b = potential_field(black_m, eps, G, c)  # their potential (tears your king)

    wk = _king_idx(masses, 1.0)
    bk = _king_idx(masses, -1.0)

    # Disruption is caused by the OPPONENT's masses only.
    eta_w = _eta(U_b, wk, Rg, mref)   # enemy masses stressing your king : bad for White
    eta_b = _eta(U_w, bk, Rg, mref)   # your masses stressing enemy king : good for White

    # Force-based disruption, flipped around the learned Roche threshold.
    bonus_b = bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_w[bk] + 1e-9) - roche))
    pen_w = -bonus * jax.nn.sigmoid(kgain * (jnp.linalg.norm(F_b[wk] + 1e-9) - roche))

    # Global field-energy edge: more total disruptive mass/reach = stronger.
    # Kings are EXCLUDED — a king's own 1000-mass self-field would otherwise
    # dwarf every piece and erase the signal. This measures your *piece* reach.
    is_king = jnp.abs(masses) > 500.0
    white_p = jnp.where(is_king, 0.0, white_m)
    black_p = jnp.where(is_king, 0.0, black_m)
    e_w = jnp.sum(jnp.sum(force_field(white_p, eps, G, c) ** 2, axis=-1))
    e_b = jnp.sum(jnp.sum(force_field(black_p, eps, G, c) ** 2, axis=-1))
    global_edge = gamma * (e_w - e_b)

    # Gravitational material edge: you command more mass -> stronger field.
    # Rewards captures / winning material; gives the search a clean, varied
    # gradient instead of the flat equilibrium of the disruption term alone.
    material = mat_gain * (jnp.sum(white_m) - jnp.sum(black_m))

    return eta_b - eta_w + bonus_b + pen_w + global_edge + material


@jax.jit
def _score_core(masses, G: float, eps: float, c: float, roche: float,
                bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 1.0):
    """Traced constants — for training (gradient flows into all 8 leaves)."""
    return _score_body(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                       mref, mat_gain)


@jax.jit
def _score_core_static(masses, G: float, eps: float, c: float, roche: float,
                       bonus: float = 50.0, kgain: float = 4.0, gamma: float = 0.25,
                       Rg: float = 1.0, mref: float = 3.5, mat_gain: float = 1.0):
    """Static constants (compile-time) — fast inference / vmap / search."""
    return _score_body(masses, G, eps, c, roche, bonus, kgain, gamma, Rg,
                       mref, mat_gain)


def score_white(masses: "jnp.ndarray", constants) -> float:
    return _score_core_static(
        masses, constants.G, constants.eps, constants.c, constants.roche,
        constants.bonus, constants.kgain, constants.gamma, constants.Rg,
        constants.mref, constants.mat_gain,
    )


def evaluate(board, constants) -> float:
    """Score from the perspective of the side to move."""
    masses = board.mass_vector()
    s = float(score_white(masses, constants))
    return s if board.turn == 0 else -s


def batch_score(masses_list, turns, constants, pad: int = _MAX_MOVES):
    """Pad a list of child mass vectors to `pad` (the 218-max), vmap-evaluate.

    Returns (N,) side-to-move scores with dummy slots masked to -inf so they
    never win a node. This is the "sub-ms 218-move sweep".
    """
    n = len(masses_list)
    buf = jnp.zeros((pad, 64), dtype=jnp.float32)
    if n:
        buf = buf.at[:n].set(jnp.stack([jnp.asarray(m, dtype=jnp.float32) for m in masses_list]))
    turns_buf = jnp.array([*(turns if n else [0]), *([0] * (pad - n))], dtype=jnp.int32)
    white = jax.vmap(score_white, in_axes=(0, None))(buf, constants)  # (pad,)
    side = jnp.where(turns_buf == 0, white, -white)
    mask = jnp.concatenate([jnp.ones(n), jnp.zeros(pad - n)])
    return jnp.where(mask > 0.5, side, -jnp.inf)
