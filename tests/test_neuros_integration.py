from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from quantumbci.integrations.neuros import (
    DensityGeometryTransform,
    NeurOSFoundationEncoder,
    bind_neuros_evidence,
)


@dataclass(frozen=True)
class FrameLike:
    data: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


def test_density_geometry_transform_preserves_frame_and_metadata() -> None:
    rng = np.random.default_rng(4)
    frame = FrameLike(
        data=rng.normal(size=(4, 64)),
        metadata={"source": "synthetic"},
    )
    transformed = DensityGeometryTransform(output="vector").transform(frame)
    assert isinstance(transformed, FrameLike)
    assert transformed.data.shape == (16,)
    assert np.all(np.isfinite(transformed.data))
    assert transformed.metadata["source"] == "synthetic"
    assert transformed.metadata["representation"] == "quantumbci_density_vector"
    assert transformed.metadata["quantumbci_claim_class"] == "quantum_inspired"
    assert transformed.metadata["quantumbci_density_dimension"] == 4


def test_density_geometry_observables_are_finite() -> None:
    rng = np.random.default_rng(5)
    values = DensityGeometryTransform(output="observables").transform(
        rng.normal(size=(3, 80))
    )
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert 1 / 3 <= values[0] <= 1.0 + 1e-12
    assert values[1] >= 0.0
    assert values[2] >= 0.0


def test_density_geometry_rejects_ambiguous_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="samples and feature"):
        DensityGeometryTransform().transform(np.arange(20.0))


class MockFoundationAdapter:
    def encode(self, eeg: np.ndarray, *, sfreq: float) -> np.ndarray:
        assert sfreq == 250.0
        # Single-example token representation; bridge should add the batch axis.
        return np.stack([eeg.mean(axis=-1), eeg.std(axis=-1)], axis=0)


def test_neuros_foundation_encoder_normalizes_single_example_shape() -> None:
    eeg = np.arange(24.0).reshape(4, 6)
    encoder = NeurOSFoundationEncoder(
        MockFoundationAdapter(),
        sample_rate_kw="sfreq",
    )
    embeddings = encoder.encode(eeg, sample_rate_hz=250.0)
    assert embeddings.shape == (1, 2, 4)


def test_neuros_evidence_binding_is_stable_and_partition_specific() -> None:
    plan = {"plan_id": "plan-123"}
    partition = SimpleNamespace(fingerprint="partition-a")
    split = SimpleNamespace(fingerprint="split-a", partition=partition)
    first = bind_neuros_evidence(
        plan,
        dataset_fingerprint="raw-data-sha256",
        partition=partition,
        calibration_split=split,
        neuros_source_sha="neuros-git-sha",
    )
    second = bind_neuros_evidence(
        plan,
        dataset_fingerprint="raw-data-sha256",
        partition=partition,
        calibration_split=split,
        neuros_source_sha="neuros-git-sha",
    )
    assert first.scientific_run_id == second.scientific_run_id
    assert first.partition_fingerprint == "partition-a"
    assert first.split_fingerprint == "split-a"


def test_neuros_evidence_binding_rejects_cross_partition_split() -> None:
    partition = SimpleNamespace(fingerprint="partition-a")
    other = SimpleNamespace(fingerprint="partition-b")
    split = SimpleNamespace(fingerprint="split-b", partition=other)
    with pytest.raises(ValueError, match="different neurOS partition"):
        bind_neuros_evidence(
            {"plan_id": "plan-123"},
            dataset_fingerprint="raw-data-sha256",
            partition=partition,
            calibration_split=split,
            neuros_source_sha="neuros-git-sha",
        )
