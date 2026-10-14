"""JAX CPU tuning for GPU-less machines (stdlib-only: safe to load before jax).

Forces XLA to expose all logical threads as host devices. Skipped when XLA_FLAGS
is already set, a GPU is present (nvidia-smi), or KEPLER64_NO_XLA_TUNE is set:
the device-count flag would pin JAX to CPU and strangle cloud-GPU runs.
"""
from __future__ import annotations

import os
import shutil


def tune_xla() -> None:
    n_logical = os.cpu_count() or 4
    if ("XLA_FLAGS" not in os.environ
            and not os.environ.get("KEPLER64_NO_XLA_TUNE")
            and shutil.which("nvidia-smi") is None):
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={n_logical}")
    os.environ.setdefault("OMP_NUM_THREADS", str(n_logical))
