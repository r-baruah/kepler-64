"""Multiverse mechanics: Layer-2 Bayesian model average, Observer variational shifts, accretion, and fluid flow."""

from .accretion import apply_capture
from .fluid import uncertainty_field, stokes_flow
from .observer import observe_update
from .posterior import multiverse_score_white

__all__ = [
    "apply_capture",
    "uncertainty_field",
    "stokes_flow",
    "observe_update",
    "multiverse_score_white",
]
