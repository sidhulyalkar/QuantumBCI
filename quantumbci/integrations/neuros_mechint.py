"""QuantumBCI interventions that compose with the optional neuros-mechint package.

The intervention classes are dependency-free and satisfy neuros-mechint's structural
``InputIntervention`` protocol (``name``, ``target``, ``apply``, ``metadata``). The
runner helper imports neuros-mechint lazily so base QuantumBCI remains NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..claims import ClaimClass
from .neuros import NeurOSUnavailableError


def _density_batch(reference: Any) -> np.ndarray:
    value = np.asarray(reference, dtype=complex)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError("density intervention input must end in square matrix dimensions")
    if not np.all(np.isfinite(value)):
        raise ValueError("density intervention input contains non-finite values")
    return value


@dataclass(frozen=True)
class RemoveDensityOffDiagonals:
    """Delete coherence-like off-diagonal structure while preserving the diagonal."""

    name: str = "remove_density_off_diagonals"
    target: str = "density_state.off_diagonal"

    def apply(self, reference: Any) -> np.ndarray:
        value = _density_batch(reference)
        result = np.zeros_like(value)
        index = np.arange(value.shape[-1])
        result[..., index, index] = value[..., index, index]
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "preserves": ["diagonal", "trace"],
            "tests": "whether claimed benefit depends on off-diagonal density structure",
        }


@dataclass(frozen=True)
class PermuteDensityBasis:
    """Apply a deterministic basis permutation, preserving eigenvalues and trace."""

    permutation: tuple[int, ...]
    name: str = "permute_density_basis"
    target: str = "density_state.basis"

    def __post_init__(self) -> None:
        if not self.permutation:
            raise ValueError("permutation must not be empty")
        expected = tuple(range(len(self.permutation)))
        if tuple(sorted(self.permutation)) != expected:
            raise ValueError(f"permutation must contain each index in {expected} exactly once")

    def apply(self, reference: Any) -> np.ndarray:
        value = _density_batch(reference)
        if value.shape[-1] != len(self.permutation):
            raise ValueError(
                "permutation dimension does not match density state: "
                f"{len(self.permutation)} != {value.shape[-1]}"
            )
        index = np.asarray(self.permutation, dtype=int)
        return value[..., index, :][..., :, index]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "permutation": list(self.permutation),
            "preserves": ["trace", "eigenvalues", "purity", "von_neumann_entropy"],
            "tests": "basis dependence of the fitted mechanism/readout",
        }


@dataclass(frozen=True)
class MixWithMaximallyMixedState:
    """Dose-response intervention rho -> (1-alpha)rho + alpha I/d."""

    alpha: float
    name: str = "mix_with_maximally_mixed_state"
    target: str = "density_state.information_content"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

    def apply(self, reference: Any) -> np.ndarray:
        value = _density_batch(reference)
        dimension = value.shape[-1]
        identity = np.eye(dimension, dtype=complex) / dimension
        return (1.0 - float(self.alpha)) * value + float(self.alpha) * identity

    def metadata(self) -> Mapping[str, Any]:
        return {
            "claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "alpha": float(self.alpha),
            "preserves": ["trace", "positive_semidefinite_for_valid_density_input"],
            "tests": "dose response as density information is erased",
        }


def run_neuros_mechint_input_audit(
    reference: Any,
    metric: Callable[[Any], Any],
    interventions: Iterable[Any],
    *,
    controls: Iterable[Any] = (),
    experiment_name: str,
    model_id: str,
    dataset_id: str = "in_memory",
    metric_name: str = "quantumbci_metric",
    seed: int = 0,
    evidence_tier: str = "unit",
    git_sha: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Run QuantumBCI representation interventions through neuros-mechint.

    The return type is neuros-mechint's native ``InputExperimentResult``. QuantumBCI
    deliberately does not wrap or reinterpret its evidence tier/result schema.
    """

    try:
        from neuros_mechint import InputCausalExperiment, InputMetric
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise NeurOSUnavailableError(
            "neuros-mechint is required for the shared causal-evidence runner. "
            "Install the QuantumBCI 'neuros-mechint' extra or the neurOS workspace package."
        ) from exc

    experiment = InputCausalExperiment(
        reference=reference,
        metric=InputMetric(metric, name=metric_name),
        experiment_name=experiment_name,
        model_id=model_id,
        dataset_id=dataset_id,
        seed=int(seed),
        evidence_tier=evidence_tier,
        git_sha=git_sha,
        metadata={
            "quantumbci_claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            **dict(metadata or {}),
        },
    )
    return experiment.run(interventions, controls=controls)
