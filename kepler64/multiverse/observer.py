"""2B "the Observer" — the true layer-2 (absurdity about absurdity).

chess:  the evaluation is not a fixed function; it co-evolves with the game.
physics: choosing a move updates the engine's belief about which physics it
        inhabits (posterior shift of (G, c)), KL-anchored to the learned base so
        it cannot hallucinate. This is online (meta-)variational inference; the
        "Born rule" is the absurdity wrapper.
"""

import jax.numpy as jnp

from ..core.constants import Constants


def observe_update(base: Constants, masses_after, alpha: float = 1e-2, beta_kl: float = 0.1):
    """Tiny KL-anchored shift of G and c based on the post-move position.

    If the move created a violently disrupted enemy King (large |score|), lean
    into a stronger-gravity, slightly-faster-c universe; otherwise drift back to
    the prior. Bounded by the monotonicity prior on c.
    """
    score = float(_score(masses_after, base))
    push = jnp.tanh(score)  # -1..1
    G_new = base.G * (1.0 + alpha * push)
    c_new = float(jnp.clip(base.c + alpha * 2.0 * push, 1.0, 10.0))
    # KL anchor back toward base
    G_new = G_new + beta_kl * (base.G - G_new)
    c_new = c_new + beta_kl * (base.c - c_new)
    return Constants(G=G_new, eps=base.eps, c=c_new, roche=base.roche)


def _score(masses, constants):
    from ..core.evaluate import score_white

    return score_white(masses, constants)
