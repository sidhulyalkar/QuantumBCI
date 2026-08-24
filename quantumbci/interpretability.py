"""Mechanism-level probes and intervention utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .states import expectation, l1_coherence, purity, von_neumann_entropy

Array = np.ndarray


@dataclass(frozen=True)
class StateSignature:
    purity: float
    entropy_bits: float
    l1_coherence: float
    observables: dict[str, float]


def state_signature(
    rho: Array, observables: Mapping[str, Array] | None = None
) -> StateSignature:
    """Expose interpretable scalar observables of a latent density state."""

    measured: dict[str, float] = {}
    for name, operator in (observables or {}).items():
        value = expectation(rho, operator)
        if abs(value.imag) > 1e-8:
            raise ValueError(f"observable {name!r} produced a non-real expectation")
        measured[name] = float(value.real)
    return StateSignature(
        purity=purity(rho),
        entropy_bits=von_neumann_entropy(rho),
        l1_coherence=l1_coherence(rho),
        observables=measured,
    )


def mechanism_delta(
    before: StateSignature, after: StateSignature
) -> dict[str, float]:
    """Return after-before changes for common state observables."""

    result = {
        "purity": after.purity - before.purity,
        "entropy_bits": after.entropy_bits - before.entropy_bits,
        "l1_coherence": after.l1_coherence - before.l1_coherence,
    }
    shared = before.observables.keys() & after.observables.keys()
    result.update(
        {f"observable:{name}": after.observables[name] - before.observables[name] for name in shared}
    )
    return result


def ablation_sensitivity(
    model: Callable[[Array], Array],
    sample: Array,
    groups: Sequence[Sequence[int]],
    *,
    baseline: float = 0.0,
) -> Array:
    """Measure prediction-vector change after explicit feature-group ablations."""

    x = np.asarray(sample, dtype=float)
    reference = np.asarray(model(x.copy()), dtype=float)
    effects = []
    for group in groups:
        perturbed = x.copy()
        perturbed[..., list(group)] = baseline
        candidate = np.asarray(model(perturbed), dtype=float)
        if candidate.shape != reference.shape:
            raise ValueError("model output shape changed under ablation")
        effects.append(float(np.linalg.norm(candidate - reference)))
    return np.asarray(effects)


def bootstrap_stability(
    values: Array, *, n_boot: int = 1000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Return mean and a simple bootstrap 95% interval for an interpretable scalar."""

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two observations are required")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), (float(low), float(high))
