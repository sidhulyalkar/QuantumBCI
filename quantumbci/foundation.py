"""Thin adapters for probing pretrained neural foundation-model representations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .states import density_from_samples

Array = np.ndarray


class FoundationEncoder(Protocol):
    """Minimal contract expected from a LaBraM/EEGPT/other encoder adapter."""

    def encode(self, eeg: Array, *, sample_rate_hz: float) -> Array:
        """Return latent tokens with shape (batch, tokens, features)."""
        ...


def density_states_from_embeddings(embeddings: Array, *, center: bool = True) -> Array:
    """Map foundation-model latent tokens to one density operator per sample.

    This deliberately does not depend on a specific foundation model. The quantum-like
    layer can therefore be compared on identical frozen embeddings from LaBraM, EEGPT,
    BrainWave, a specialist encoder, or a random/control encoder.
    """

    x = np.asarray(embeddings, dtype=float)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError("embeddings must have shape (batch, tokens, features) or (tokens, features)")
    return np.stack([density_from_samples(sample, center=center) for sample in x])
