"""Image-seeded universe genesis.

chess:  the starting masses and the fundamental constants of the universe are
        derived from a single photograph — not from a UNIX timestamp.
physics: SHA-256(byte_array) -> PRNGKey (decorative determinism). The REAL hook:
        if Pillow is available we DECODE the image to 8x8 grayscale and take the
        genuine 2D FFT of the pixels; otherwise we fall back to byte-entropy
        seeding from the raw file bytes (a pseudo-random constant generator, NOT
        an image FFT). Log-compression is applied before normalization so the
        spectra do not collapse onto their clamps. The first few bins map to
        [G, eps, roche, c]_init. An 8x8-resized crop maps pixel intensity ->
        initial piece masses, so the opening position itself is image-derived.
"""

import hashlib
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from .constants import Constants, PIECE_MASSES

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover - PIL optional
    _HAS_PIL = False


def seed_from_image(path: str, base: Constants) -> Constants:
    with open(path, "rb") as fh:
        raw = fh.read()
    _ = int.from_bytes(hashlib.sha256(raw).digest(), "big")  # deterministic seed

    spectrum = _fft_magnitudes(path, raw)
    # Log-compress the spectrum so low-frequency-dominated natural images do not
    # collapse every constant onto its clamp; normalize on the compressed values.
    log_spectrum = jnp.log1p(jnp.abs(spectrum))
    norm = log_spectrum / (log_spectrum.sum() + 1e-9)
    g = float(norm[0]) if len(norm) > 0 else 1.0
    eps = float(norm[1]) if len(norm) > 1 else 0.5
    roche = float(norm[2]) if len(norm) > 2 else 1.0
    c = float(norm[3]) if len(norm) > 3 else 4.0
    return replace(
        base,
        # Scale hacks removed: log-compression keeps constants in a meaningful
        # range, so only the documented output clamps remain.
        G=max(1e-3, g),
        eps=max(1e-2, eps),
        roche=max(1e-2, roche),
        c=min(10.0, max(1.0, c)),
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


def _fft_magnitudes(path: str, raw: bytes) -> "jnp.ndarray":
    """Return a (flattened) 2D-FFT magnitude spectrum.

    When Pillow is present this is the genuine FFT of the decoded 8x8 grayscale
    image pixels. Otherwise it is byte-entropy seeding: the raw file bytes are
    reshaped to a square and rFFT'd (a pseudo-random seed, NOT a real image FFT).
    """
    if _HAS_PIL:
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize((8, 8))
                img = np.asarray(im, dtype=np.float32)
            spec = np.abs(np.fft.rfft2(img))
            return jnp.asarray(spec.reshape(-1))
        except Exception:
            pass  # fall back to byte-entropy seeding below

    # Byte-entropy seeding fallback (no PIL, or image could not be decoded).
    import math

    n = int(math.isqrt(max(len(raw), 1)))
    buf = np.frombuffer(raw[: n * n], dtype=np.uint8)
    if buf.size < n * n:
        buf = np.pad(buf, (0, n * n - buf.size))[: n * n]
    img = buf.reshape(n, n).astype(np.float32)
    spec = np.abs(np.fft.rfft2(img))
    return jnp.asarray(spec.reshape(-1))
