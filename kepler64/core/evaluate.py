"""Evaluation: physics (tidal disruption) + a soft, differentiable tactical layer.

chess:  the score tells the search which move keeps your King bound and the
        enemy King disrupted.
physics: Eval = +eta_enemy - eta_self + bonus(enemy king pressure)
               - penalty(own king pressure).
        The tactical terms are sigmoids on local force magnitude, expressed in
        the force language, so the whole pipeline stays differentiable and the
        "fully differentiable evaluation" hook survives (no boolean branch).

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


def _eta(masses, U, king_sq, G: float, Rg: float = 1.0) -> float:
    A = tidal_tensor_at(U, king_sq)
    lam1, _ = eig2x2(A)
    Mking = jnp.abs(masses[king_sq]) + 1e-9
    # eta = lambda1 * Rg^3 / (G * Mking^2)  (Rg^3 kept explicit for generality)
    return lam1 * (Rg**3) / (G * Mking**2 + 1e-9)


def _score_body(masses, G: float, eps: float, c: float, roche: float):
    """Evaluation from White's perspective (positive = good for White)."""
    abs_m = jnp.abs(masses)
    F = force_field(abs_m, eps, G, c)
    U = potential_field(abs_m, eps, G, c)
    wk = _king_idx(masses, 1.0)
    bk = _king_idx(masses, -1.0)
    eta_w = _eta(masses, U, wk, G)  # own King stress: bad for White
    eta_b = _eta(masses, U, bk, G)  # enemy King stress: good for White
    bonus_b = 50.0 * jax.nn.sigmoid(4.0 * (jnp.linalg.norm(F[bk] + 1e-9) - 1.0))
    pen_w = -50.0 * jax.nn.sigmoid(4.0 * (jnp.linalg.norm(F[wk] + 1e-9) - 1.0))
    return eta_b - eta_w + bonus_b + pen_w


@jax.jit
def _score_core(masses, G: float, eps: float, c: float, roche: float):
    """Traced constants — for training (gradient flows into G, eps, c, roche)."""
    return _score_body(masses, G, eps, c, roche)


@jax.jit
def _score_core_static(masses, G: float, eps: float, c: float, roche: float):
    """Static constants (compile-time) — fast inference / vmap / search."""
    return _score_body(masses, G, eps, c, roche)


def score_white(masses: "jnp.ndarray", constants) -> float:
    return _score_core_static(masses, constants.G, constants.eps, constants.c, constants.roche)


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
