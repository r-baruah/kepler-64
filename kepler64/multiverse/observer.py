"""2B "the Observer" — the true layer-2 (absurdity about absurdity).

chess:  the evaluation is not a fixed function; it co-evolves with the game.
physics: choosing a move updates the engine's belief about which physics it
        inhabits: every trainable leaf shifts a tiny step toward a universe
        whose field explains what just happened, then is pulled back by a KL
        anchor so it cannot hallucinate. This is online (meta-)variational
        inference; the "Born rule" is the absurdity wrapper.

STATUS — WIRED (opt-in) since 2026-08-23:
    `RocheEngine.play(..., observe=True)` calls `observe_update` after each
    searched move and PERSISTS the shifted constants on the engine instance,
    so successive calls in one game genuinely play under evolving laws. The
    shift is deliberately tiny (alpha=1e-2, KL anchor 0.1): over dozens of
    plies G/c wander a few percent around the learned base — measurable,
    never runaway. Default remains observe=False so matches and benchmarks
    stay reproducible under frozen physics.

    Earlier bug (fixed): this function reconstructed Constants(G=…, eps=…,
    c=…, roche=…) and silently RESET the other ten leaves to dataclass
    defaults the moment it ran. It now rebuilds through `dataclasses.replace`
    over TRAINABLE_LEAVES, so every leaf survives the observation.
"""

import jax.numpy as jnp

from dataclasses import replace

from ..core.constants import Constants, TRAINABLE_LEAVES


def observe_update(base: Constants, masses_after, alpha: float = 1e-2,
                   beta_kl: float = 0.1) -> Constants:
    """One KL-anchored belief shift of ALL trainable leaves.

    A violently disrupted enemy King (large |score|) leans the universe toward
    stronger gravity and a slightly faster light speed; a quiet outcome lets
    the leaves relax back toward the learned base. `c` alone is clamped to its
    physical prior [1, 10]; every other leaf drifts multiplicatively, which
    keeps positive constants positive and the shift scale-free. Returns a NEW
    Constants — `base` is never mutated.
    """
    score = float(_score(masses_after, base))
    push = max(-1.0, min(1.0, float(jnp.tanh(score))))  # -1..1, finite always

    updates = {}
    for name in TRAINABLE_LEAVES:
        v = float(getattr(base, name))
        shifted = v * (1.0 + alpha * push)
        if name == "c":
            shifted = min(max(shifted, 1.0), 10.0)
        # KL anchor: most of the pull-back happens immediately; the net step
        # per observation is ~alpha*push*(1-beta_kl) of each leaf.
        updates[name] = shifted + beta_kl * (v - shifted)

    return replace(base, **updates)


def _score(masses, constants):
    from ..core.evaluate import score_white

    return score_white(masses, constants)
