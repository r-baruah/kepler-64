"""2A "the Multiverse" — Bayesian model average over possible physics.

chess:  instead of asking "which move is best under one set of rules?", the
        engine asks "which move is best across possible universes?"
physics: sample K realizations of the physics from the posterior; the
        evaluation is their average with EQUAL posterior mass (true Bayesian
        model average). The retarded Green's function G_ret(theta_i) is the
        mathematical core: the engine does not know which universe's c governs
        the delay reaching the enemy King.

NOTE: the live, jax-traced implementation (with parent/delta-term threading)
lives in `kepler64.core.evaluate.multiverse_score_white`. This module now just
re-exports it so older import paths keep working. The old local copy (which
returned a plain Python float and was never wired into search/training) has
been removed to avoid two diverging implementations of the same idea.
"""

from ..core.evaluate import multiverse_score_white  # noqa: F401
from ..core.constants import Constants             # noqa: F401
