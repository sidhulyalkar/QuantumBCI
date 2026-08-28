"""Dependency-light participant-level uncertainty for confirmatory evidence summaries.

The functions here operate on independent inference-unit effects, normally one value per
participant after any repeated sessions have been summarized according to the declared
study estimand. They intentionally do not resample trials/windows as independent units.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParticipantEffectInference:
    n_units: int
    observed_mean: float
    observed_median: float
    positive_fraction: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_confidence: float
    bootstrap_resamples: int
    sign_flip_pvalue_two_sided: float
    sign_flip_method: str
    sign_flip_draws: int
    seed: int

    def to_mapping(self) -> dict[str, float | int | str]:
        return {
            "n_units": self.n_units,
            "observed_mean": self.observed_mean,
            "observed_median": self.observed_median,
            "positive_fraction": self.positive_fraction,
            "bootstrap_ci_lower": self.bootstrap_ci_lower,
            "bootstrap_ci_upper": self.bootstrap_ci_upper,
            "bootstrap_confidence": self.bootstrap_confidence,
            "bootstrap_resamples": self.bootstrap_resamples,
            "sign_flip_pvalue_two_sided": self.sign_flip_pvalue_two_sided,
            "sign_flip_method": self.sign_flip_method,
            "sign_flip_draws": self.sign_flip_draws,
            "seed": self.seed,
        }


def _effects(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("participant inference requires at least two independent unit effects")
    if not np.all(np.isfinite(array)):
        raise ValueError("participant effects must be finite")
    return array


def _sign_flip_pvalue(
    effects: np.ndarray,
    *,
    rng: np.random.Generator,
    exact_max_units: int,
    monte_carlo_draws: int,
) -> tuple[float, str, int]:
    """Two-sided paired sign-flip randomization test under a symmetric zero-effect null."""

    observed = abs(float(np.mean(effects)))
    n_units = len(effects)
    tolerance = 1e-15
    if n_units <= exact_max_units:
        total = 1 << n_units
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=n_units):
            statistic = abs(float(np.mean(effects * np.asarray(signs, dtype=float))))
            extreme += statistic >= observed - tolerance
        return float(extreme / total), "exact_sign_flip", int(total)

    if monte_carlo_draws < 1000:
        raise ValueError("monte_carlo_draws must be at least 1000")
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(monte_carlo_draws, n_units))
    statistics = np.abs(np.mean(signs * effects[None, :], axis=1))
    extreme = int(np.sum(statistics >= observed - tolerance))
    # Add-one correction prevents a Monte Carlo p-value of exactly zero.
    return (
        float((extreme + 1) / (monte_carlo_draws + 1)),
        "monte_carlo_sign_flip",
        int(monte_carlo_draws),
    )


def participant_effect_inference(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 5000,
    seed: int = 0,
    exact_sign_flip_max_units: int = 16,
    monte_carlo_sign_flip_draws: int = 100000,
) -> ParticipantEffectInference:
    """Summarize one independent-participant effect distribution.

    The percentile bootstrap estimates uncertainty in the participant-level mean. The
    sign-flip test is a paired/randomization-style test of a symmetric zero-effect null.
    Neither statistic substitutes for a preregistered effect-size threshold or power
    analysis, and neither should be applied to trial/window-level pseudo-replicates.
    """

    effects = _effects(values)
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if int(bootstrap_resamples) < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    if int(exact_sign_flip_max_units) < 1:
        raise ValueError("exact_sign_flip_max_units must be positive")

    rng = np.random.default_rng(int(seed))
    n_units = len(effects)
    draws = rng.integers(0, n_units, size=(int(bootstrap_resamples), n_units))
    boot_means = np.mean(effects[draws], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    ci_lower, ci_upper = np.quantile(boot_means, [alpha, 1.0 - alpha])
    pvalue, method, sign_draws = _sign_flip_pvalue(
        effects,
        rng=rng,
        exact_max_units=int(exact_sign_flip_max_units),
        monte_carlo_draws=int(monte_carlo_sign_flip_draws),
    )
    return ParticipantEffectInference(
        n_units=n_units,
        observed_mean=float(np.mean(effects)),
        observed_median=float(np.median(effects)),
        positive_fraction=float(np.mean(effects > 0.0)),
        bootstrap_ci_lower=float(ci_lower),
        bootstrap_ci_upper=float(ci_upper),
        bootstrap_confidence=float(confidence),
        bootstrap_resamples=int(bootstrap_resamples),
        sign_flip_pvalue_two_sided=pvalue,
        sign_flip_method=method,
        sign_flip_draws=sign_draws,
        seed=int(seed),
    )
