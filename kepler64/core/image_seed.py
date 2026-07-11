"""Image-seeded universe genesis.

chess:  the starting masses and the fundamental constants of the universe are
        derived from a single photograph — not from a UNIX timestamp.
physics: SHA-256(byte_array) -> PRNGKey (decorative determinism). The REAL hook:
        FFT of the image byte stream -> |bins|^2 -> normalized -> [G, eps, roche,
        c]_init. An 8x8-resized crop maps pixel intensity -> initial piece
        masses, so the opening position itself is image-derived.

Implemented without PIL: the raw file bytes are reshaped to a square and
2D-rFFT'd, so any image (or any file) seeds the universe.
"""

import hashlib
import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from .constants import Constants, PIECE_MASSES


def seed_from_image(path: str, base: Constants) -> Constants:
    with open(path, "rb") as fh:
        raw = fh.read()
    _ = int.from_bytes(hashlib.sha256(raw).digest(), "big")  # deterministic seed

    spectrum = _fft_magnitudes(raw)
    norm = spectrum / (spectrum.sum() + 1e-9)
    g = float(norm[0]) if len(norm) > 0 else 1.0
    eps = float(norm[1]) if len(norm) > 1 else 0.5
    roche = float(norm[2]) if len(norm) > 2 else 1.0
    c = float(norm[3]) if len(norm) > 3 else 4.0
    return replace(
        base,
        G=max(1e-3, g * 5.0),
        eps=max(1e-2, eps * 2.0),
        roche=max(1e-2, roche * 2.0),
        c=min(10.0, max(1.0, c * 10.0)),
    )


def initial_masses_from_image(path: str):
    """8x8 intensity crop -> (64,) initial piece-mass multipliers in [0,1]."""
    with open(path, "rb") as fh:
        raw = fh.read()
    n = 64
    buf = np.frombuffer(raw[: n * n], dtype=np.uint8)
    if buf.size < n * n:
        buf = np.pad(buf, (0, n * n - buf.size))[: n * n]
    crop = buf.reshape(n, n).astype(np.float32) / 255.0
    return jnp.asarray(crop.reshape(-1))


def _fft_magnitudes(raw: bytes) -> "jnp.ndarray":
    n = int(math.isqrt(max(len(raw), 1)))
    buf = np.frombuffer(raw[: n * n], dtype=np.uint8)
    if buf.size < n * n:
        buf = np.pad(buf, (0, n * n - buf.size))[: n * n]
    img = buf.reshape(n, n).astype(np.float32)
    spec = np.abs(np.fft.rfft2(img))
    mag = spec.reshape(-1)
    return jnp.asarray(mag / (mag.sum() + 1e-9))
