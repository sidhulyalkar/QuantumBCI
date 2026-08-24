"""Deterministic synthetic neural signals for mechanism tests and examples."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def synthetic_eeg(
    *,
    duration_s: float = 8.0,
    sample_rate_hz: float = 256.0,
    channels: int = 4,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Generate multi-channel EEG-like oscillations with a transient and colored-ish noise.

    This is a unit-test/demo generator, not a physiological simulator.
    """

    if duration_s <= 0 or sample_rate_hz <= 0 or channels <= 0:
        raise ValueError("duration, sample rate, and channels must be positive")
    n = int(round(duration_s * sample_rate_hz))
    t = np.arange(n) / sample_rate_hz
    rng = np.random.default_rng(seed)
    data = np.empty((n, channels), dtype=float)
    transient = np.exp(-0.5 * ((t - duration_s * 0.55) / 0.12) ** 2)
    for channel in range(channels):
        alpha = np.sin(2 * np.pi * (9.5 + 0.35 * channel) * t + 0.2 * channel)
        beta = 0.35 * np.sin(2 * np.pi * (18.0 + channel) * t - 0.1 * channel)
        white = rng.normal(scale=0.22, size=n)
        smooth_noise = np.convolve(white, np.ones(5) / 5, mode="same")
        data[:, channel] = alpha + beta + (0.5 + 0.1 * channel) * transient + smooth_noise
    return t, data


def window_features(data: Array, *, window: int, step: int) -> Array:
    """Return flattened windows suitable for compact representation experiments."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must have shape (samples, channels)")
    if window <= 0 or step <= 0 or window > x.shape[0]:
        raise ValueError("invalid window/step")
    starts = range(0, x.shape[0] - window + 1, step)
    return np.stack([x[start : start + window].reshape(-1) for start in starts])
