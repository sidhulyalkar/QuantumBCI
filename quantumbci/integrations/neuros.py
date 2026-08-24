"""Optional neurOS bridge for runtime, foundation-model, and evidence interoperability.

QuantumBCI remains independently usable with NumPy only. When neurOS is present this
module lets QuantumBCI participate as an external research extension without making
neurOS depend on QuantumBCI. The integration is intentionally structural and fail-closed:
no random or placeholder foundation-model output is substituted for an unavailable
neurOS adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Any, Mapping

import numpy as np

from ..claims import ClaimClass
from ..states import density_from_samples, l1_coherence, purity, von_neumann_entropy


class NeurOSUnavailableError(ImportError):
    """Raised when a requested neurOS capability is not installed or runnable."""


@dataclass(frozen=True)
class NeurOSIntegrationStatus:
    """Installed neurOS distribution versions visible to QuantumBCI."""

    core: str | None
    foundation: str | None
    mechint: str | None

    @property
    def runtime_available(self) -> bool:
        return self.core is not None

    @property
    def foundation_available(self) -> bool:
        return self.foundation is not None

    @property
    def mechint_available(self) -> bool:
        return self.mechint is not None

    def to_mapping(self) -> dict[str, str | None]:
        return {
            "neuros-core": self.core,
            "neuros-foundation": self.foundation,
            "neuros-mechint": self.mechint,
        }


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def neuros_integration_status() -> NeurOSIntegrationStatus:
    """Return installed neurOS package versions without importing heavy dependencies."""

    return NeurOSIntegrationStatus(
        core=_distribution_version("neuros-core"),
        foundation=_distribution_version("neuros-foundation"),
        mechint=_distribution_version("neuros-mechint"),
    )


def _samples_by_features(data: Any, *, sample_axis: int) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("density geometry requires at least samples and feature dimensions")
    axis = int(sample_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"sample_axis={sample_axis} is invalid for shape {array.shape}")
    moved = np.moveaxis(array, axis, 0)
    samples = moved.reshape(moved.shape[0], -1)
    if samples.shape[0] < 2:
        raise ValueError("density geometry requires at least two samples")
    if samples.shape[1] < 2:
        raise ValueError("density geometry requires at least two features")
    if not np.all(np.isfinite(samples)):
        raise ValueError("density geometry input contains non-finite values")
    return samples


def _vectorize_density(rho: np.ndarray) -> np.ndarray:
    """Encode a Hermitian d x d state into exactly d^2 real features."""

    state = np.asarray(rho, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("rho must be square")
    upper = np.triu_indices(state.shape[0], k=1)
    return np.concatenate(
        [
            np.diag(state).real,
            state[upper].real,
            state[upper].imag,
        ]
    ).astype(float, copy=False)


def _replace_data(item: Any, data: np.ndarray, metadata: Mapping[str, Any]) -> Any:
    """Preserve a neurOS SignalFrame or compatible frozen dataclass when possible."""

    if is_dataclass(item) and hasattr(item, "data") and hasattr(item, "metadata"):
        current = dict(getattr(item, "metadata", {}) or {})
        return replace(item, data=np.asarray(data), metadata={**current, **dict(metadata)})
    return np.asarray(data)


class DensityGeometryTransform:
    """neurOS-compatible transform exposing QuantumBCI density geometry.

    neurOS conventionally carries EEG chunks as ``(channels, samples)`` arrays, so
    ``sample_axis=-1`` is the safe default. For an upstream ``(samples, channels)``
    representation set ``sample_axis=0`` explicitly.

    ``output='vector'`` preserves the full Hermitian density state as d^2 real values.
    ``output='observables'`` emits only purity, von Neumann entropy, and L1 coherence.
    Both outputs remain quantum-inspired representations, not evidence for a physical
    quantum neural substrate.
    """

    def __init__(
        self,
        sample_axis: int = -1,
        output: str = "vector",
        center: bool = True,
    ) -> None:
        normalized = str(output).strip().lower()
        if normalized not in {"vector", "observables"}:
            raise ValueError("output must be 'vector' or 'observables'")
        self.sample_axis = int(sample_axis)
        self.output = normalized
        self.center = bool(center)

    def transform(self, item: Any) -> Any:
        raw = getattr(item, "data", item)
        samples = _samples_by_features(raw, sample_axis=self.sample_axis)
        rho = density_from_samples(samples, center=self.center)
        if self.output == "vector":
            features = _vectorize_density(rho)
            representation = "quantumbci_density_vector"
        else:
            features = np.asarray(
                [purity(rho), von_neumann_entropy(rho), l1_coherence(rho)],
                dtype=float,
            )
            representation = "quantumbci_density_observables"
        metadata = {
            "representation": representation,
            "quantumbci_claim_class": ClaimClass.QUANTUM_INSPIRED.value,
            "quantumbci_density_dimension": int(rho.shape[0]),
            "quantumbci_sample_axis": self.sample_axis,
            "quantumbci_centered": self.center,
        }
        return _replace_data(item, features, metadata)


class NeurOSFoundationEncoder:
    """Adapt a neurOS foundation-model adapter to QuantumBCI's encoder protocol."""

    def __init__(
        self,
        adapter: Any,
        *,
        operation: str = "encode",
        sample_rate_kw: str | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.operation = str(operation)
        self.sample_rate_kw = sample_rate_kw
        self.encode_kwargs = dict(encode_kwargs or {})

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        operation: str = "encode",
        sample_rate_kw: str | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> "NeurOSFoundationEncoder":
        """Resolve one runnable model through neurOS's fail-closed registry."""

        try:
            from neuros.foundation_models import DEFAULT_REGISTRY
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise NeurOSUnavailableError(
                "neuros-foundation is required for registry-backed encoders. "
                "Install the QuantumBCI neurOS extra or the neurOS workspace packages."
            ) from exc
        try:
            adapter = DEFAULT_REGISTRY.adapter(model_id)
        except Exception as exc:  # neurOS owns its richer adapter error taxonomy
            raise NeurOSUnavailableError(str(exc)) from exc
        return cls(
            adapter,
            operation=operation,
            sample_rate_kw=sample_rate_kw,
            encode_kwargs=encode_kwargs,
        )

    def encode(self, eeg: np.ndarray, *, sample_rate_hz: float) -> np.ndarray:
        kwargs = dict(self.encode_kwargs)
        if self.sample_rate_kw is not None:
            kwargs[self.sample_rate_kw] = float(sample_rate_hz)

        direct = getattr(self.adapter, self.operation, None)
        if callable(direct):
            result = direct(eeg, **kwargs)
        else:
            call = getattr(self.adapter, "call", None)
            if not callable(call):
                raise NeurOSUnavailableError(
                    f"neurOS adapter does not expose operation {self.operation!r}"
                )
            result = call(self.operation, eeg, **kwargs)

        # Support common tensor libraries without making them QuantumBCI dependencies.
        detach = getattr(result, "detach", None)
        if callable(detach):
            result = detach()
        cpu = getattr(result, "cpu", None)
        if callable(cpu):
            result = cpu()
        numpy_method = getattr(result, "numpy", None)
        if callable(numpy_method):
            result = numpy_method()

        embeddings = np.asarray(result)
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        if embeddings.ndim != 3:
            raise ValueError(
                "neurOS encoder output must resolve to (batch, tokens, features) "
                f"or (tokens, features); got shape {embeddings.shape}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("neurOS encoder returned non-finite embeddings")
        return embeddings


@dataclass(frozen=True)
class NeurOSEvidenceBinding:
    """Bind QuantumBCI planning identity to immutable neurOS evidence boundaries."""

    plan_id: str
    dataset_fingerprint: str
    partition_fingerprint: str
    split_fingerprint: str
    neuros_source_sha: str
    package_versions: Mapping[str, str | None]

    def __post_init__(self) -> None:
        for name, value in (
            ("plan_id", self.plan_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("partition_fingerprint", self.partition_fingerprint),
            ("split_fingerprint", self.split_fingerprint),
            ("neuros_source_sha", self.neuros_source_sha),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must not be empty")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "neuros_source_sha": self.neuros_source_sha,
            "package_versions": dict(sorted(self.package_versions.items())),
        }

    @property
    def scientific_run_id(self) -> str:
        raw = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()


def bind_neuros_evidence(
    plan: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    partition: Any,
    neuros_source_sha: str,
    calibration_split: Any | None = None,
) -> NeurOSEvidenceBinding:
    """Bind a QuantumBCI plan to neurOS partition/split fingerprints.

    ``dataset_fingerprint`` must be the upstream/raw-data fingerprint supplied by the
    dataset ecosystem. neurOS partition fingerprints intentionally do not replace a raw
    dataset checksum. If a nested calibration split is supplied, its partition must be
    exactly the same partition passed here.
    """

    plan_id = str(plan.get("plan_id", "")).strip()
    if not plan_id:
        raise ValueError("plan must contain a non-empty plan_id")
    partition_fingerprint = str(getattr(partition, "fingerprint", "")).strip()
    if not partition_fingerprint:
        raise TypeError("partition must expose a non-empty neurOS fingerprint")

    if calibration_split is None:
        split_fingerprint = partition_fingerprint
    else:
        split_fingerprint = str(getattr(calibration_split, "fingerprint", "")).strip()
        if not split_fingerprint:
            raise TypeError("calibration_split must expose a non-empty neurOS fingerprint")
        split_partition = getattr(calibration_split, "partition", None)
        split_partition_fingerprint = str(
            getattr(split_partition, "fingerprint", "")
        ).strip()
        if split_partition_fingerprint != partition_fingerprint:
            raise ValueError("calibration split belongs to a different neurOS partition")

    status = neuros_integration_status()
    return NeurOSEvidenceBinding(
        plan_id=plan_id,
        dataset_fingerprint=str(dataset_fingerprint).strip(),
        partition_fingerprint=partition_fingerprint,
        split_fingerprint=split_fingerprint,
        neuros_source_sha=str(neuros_source_sha).strip(),
        package_versions=status.to_mapping(),
    )
