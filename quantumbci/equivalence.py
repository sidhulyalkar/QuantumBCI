"""Mathematical equivalence audits for quantum-structured representations.

The purpose of this module is adversarial: before an empirical benchmark can be
interpreted as evidence for a new mechanism, QuantumBCI should detect when the
proposed object is simply a reparameterization of a standard classical statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .states import density_from_samples


@dataclass(frozen=True)
class DensityCovarianceAudit:
    """Audit the current density constructor against normalized second moment.

    For the current QuantumBCI constructor, ``density_from_samples(X)`` is the
    trace-normalized Hermitian second moment ``XᴴX / Tr(XᴴX)`` after optional
    centering. The two objects therefore contain the same information, up to
    floating-point projection noise.
    """

    center: bool
    dimension: int
    max_abs_error: float
    frobenius_error: float
    equivalent_within_tolerance: bool
    tolerance: float
    equivalence_class: str = "trace_normalized_hermitian_second_moment"
    novel_information: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "dimension": self.dimension,
            "max_abs_error": self.max_abs_error,
            "frobenius_error": self.frobenius_error,
            "equivalent_within_tolerance": self.equivalent_within_tolerance,
            "tolerance": self.tolerance,
            "equivalence_class": self.equivalence_class,
            "novel_information": self.novel_information,
            "interpretation": (
                "The density representation contains no information beyond the "
                "corresponding trace-normalized Hermitian second moment. Any empirical "
                "difference must come from downstream parameterization, normalization, "
                "constraints, observables, interventions, or numerical effects."
            ),
        }


def trace_normalized_second_moment(
    samples: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """Return the exact classical statistic underlying ``density_from_samples``."""

    x = np.asarray(samples, dtype=complex)
    if x.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_features)")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("samples must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples contain non-finite values")
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    moment = x.conj().T @ x
    trace = float(np.trace(moment).real)
    if trace <= 0:
        raise ValueError("samples have zero variance/norm")
    return (moment / trace + (moment / trace).conj().T) / 2


def audit_density_covariance_equivalence(
    samples: np.ndarray,
    *,
    center: bool = True,
    atol: float = 1e-10,
) -> DensityCovarianceAudit:
    """Numerically verify density/covariance equivalence on one sample matrix."""

    if atol <= 0:
        raise ValueError("atol must be positive")
    density = density_from_samples(samples, center=center)
    classical = trace_normalized_second_moment(samples, center=center)
    delta = np.asarray(density - classical)
    max_error = float(np.max(np.abs(delta)))
    fro_error = float(np.linalg.norm(delta, ord="fro"))
    return DensityCovarianceAudit(
        center=bool(center),
        dimension=int(density.shape[0]),
        max_abs_error=max_error,
        frobenius_error=fro_error,
        equivalent_within_tolerance=bool(max_error <= atol),
        tolerance=float(atol),
    )


@dataclass(frozen=True)
class BatchEquivalenceAudit:
    n_examples: int
    max_abs_error: float
    max_frobenius_error: float
    equivalent_within_tolerance: bool
    tolerance: float
    equivalence_class: str = "trace_normalized_hermitian_second_moment"
    novel_information: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "max_abs_error": self.max_abs_error,
            "max_frobenius_error": self.max_frobenius_error,
            "equivalent_within_tolerance": self.equivalent_within_tolerance,
            "tolerance": self.tolerance,
            "equivalence_class": self.equivalence_class,
            "novel_information": self.novel_information,
        }


def audit_embedding_batch(
    embeddings: np.ndarray,
    *,
    center_tokens: bool = True,
    atol: float = 1e-10,
) -> BatchEquivalenceAudit:
    """Audit ``examples × tokens × features`` representations before E001 fitting."""

    values = np.asarray(embeddings)
    if values.ndim != 3:
        raise ValueError("embeddings must have shape (examples, tokens, features)")
    if len(values) == 0:
        raise ValueError("embeddings must be non-empty")
    audits = [
        audit_density_covariance_equivalence(example, center=center_tokens, atol=atol)
        for example in values
    ]
    max_abs = max(item.max_abs_error for item in audits)
    max_fro = max(item.frobenius_error for item in audits)
    return BatchEquivalenceAudit(
        n_examples=len(audits),
        max_abs_error=float(max_abs),
        max_frobenius_error=float(max_fro),
        equivalent_within_tolerance=bool(all(item.equivalent_within_tolerance for item in audits)),
        tolerance=float(atol),
    )
