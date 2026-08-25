"""Leakage-resistant authority for continuous latent trajectory experiments.

The neurOS longitudinal authority freezes *which neural samples* belong to source,
calibration, and final evaluation. Dynamics experiments additionally need to freeze
*temporal adjacency*: trajectory identity, exact window geometry, legal transitions,
representation-fit scope, and purge gaps. This module supplies that second authority
without cloning the upstream neural-data stack.

v1 is deliberately conservative: it supports fixed-duration, fixed-stride windows
only. Irregular trajectories need an explicit maximum-gap/missingness contract and
remain unsupported rather than being silently approximated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

Array = np.ndarray
TrajectoryRole = Literal["fit", "calibration", "evaluation"]
TimeStepPolicy = Literal["fixed"]
MissingDataPolicy = Literal["reject"]


def _readonly_array(values: Any, *, dtype: Any | None = None) -> Array:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _indices(values: Any, *, name: str) -> tuple[int, ...]:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    result = tuple(int(value) for value in array.tolist())
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate indices")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} contains negative indices")
    # These are evidence sets, not caller-ordered sequences. Canonical sorting keeps
    # scientific identity invariant to incidental JSON/list ordering.
    return tuple(sorted(result))


def _sha256_array(digest: Any, array: Array) -> None:
    values = np.asarray(array)
    if values.dtype.hasobject:
        raise TypeError("object-dtype arrays cannot be fingerprinted")
    contiguous = np.ascontiguousarray(values)
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    digest.update(b"\0")


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceData:
    """Exact frozen state tensor plus temporal identity for a dynamics study."""

    dataset_id: str
    states: Array
    trajectory_ids: Array
    start_times_s: Array
    stop_times_s: Array
    valid_mask: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        states = _readonly_array(self.states)
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError("states must have shape (windows, features) with >=2 windows")
        if states.dtype.hasobject or not np.issubdtype(states.dtype, np.number):
            raise TypeError("states must be a numeric non-object array")
        if not np.all(np.isfinite(states)):
            raise ValueError("states contain non-finite values")

        trajectory_ids = _readonly_array(np.asarray(self.trajectory_ids).astype(str).reshape(-1))
        starts = _readonly_array(self.start_times_s, dtype=float).reshape(-1)
        stops = _readonly_array(self.stop_times_s, dtype=float).reshape(-1)
        n = states.shape[0]
        if not (len(trajectory_ids) == len(starts) == len(stops) == n):
            raise ValueError("trajectory ids and timestamps must align one-to-one with states")
        if any(not value for value in trajectory_ids.tolist()):
            raise ValueError("trajectory_ids must be non-empty strings")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(stops)):
            raise ValueError("timestamps must be finite")
        if np.any(stops <= starts):
            raise ValueError("every trajectory window must have stop_time > start_time")

        if self.valid_mask is None:
            valid = _readonly_array(np.ones(n, dtype=bool))
        else:
            valid = _readonly_array(self.valid_mask, dtype=bool).reshape(-1)
            if len(valid) != n:
                raise ValueError("valid_mask must align one-to-one with states")

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "trajectory_ids", trajectory_ids)
        object.__setattr__(self, "start_times_s", starts)
        object.__setattr__(self, "stop_times_s", stops)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_windows(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dimension(self) -> int:
        return int(self.states.shape[1])

    @property
    def data_sha256(self) -> str:
        """Hash exact state bytes plus row-level temporal identity and metadata."""

        digest = hashlib.sha256()
        digest.update(b"quantumbci.trajectory-evidence-data.v1\0")
        digest.update(self.dataset_id.encode("utf-8"))
        digest.update(b"\0")
        _sha256_array(digest, self.states)
        identity = {
            "trajectory_ids": self.trajectory_ids.astype(str).tolist(),
            "start_times_s": [float(v) for v in self.start_times_s.tolist()],
            "stop_times_s": [float(v) for v in self.stop_times_s.tolist()],
            "valid_mask": [bool(v) for v in self.valid_mask.tolist()],
            "metadata": dict(self.metadata),
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        )
        return digest.hexdigest()


def _by_trajectory(
    data: TrajectoryEvidenceData,
    indices: tuple[int, ...],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(str(data.trajectory_ids[index]), []).append(index)
    for values in grouped.values():
        values.sort(key=lambda i: float(data.start_times_s[i]))
    return grouped


def _minimum_interval_separation(
    data: TrajectoryEvidenceData,
    left_indices: tuple[int, ...],
    right_indices: tuple[int, ...],
) -> float:
    """Return minimum edge-to-edge separation for same-trajectory cross-role windows.

    Negative values indicate temporal overlap. ``inf`` means the roles occur in
    disjoint trajectory IDs and therefore cannot leak through window overlap.
    """

    minimum = float("inf")
    left_by = _by_trajectory(data, left_indices)
    right_by = _by_trajectory(data, right_indices)
    for trajectory_id in set(left_by) & set(right_by):
        left = left_by[trajectory_id]
        right = right_by[trajectory_id]
        i = j = 0
        while i < len(left) and j < len(right):
            a = left[i]
            b = right[j]
            a_start, a_stop = float(data.start_times_s[a]), float(data.stop_times_s[a])
            b_start, b_stop = float(data.start_times_s[b]), float(data.stop_times_s[b])
            if a_stop <= b_start:
                minimum = min(minimum, b_start - a_stop)
                i += 1
            elif b_stop <= a_start:
                minimum = min(minimum, a_start - b_stop)
                j += 1
            else:
                # Magnitude is overlap duration; only the sign is needed by the gate.
                return -(min(a_stop, b_stop) - max(a_start, b_start))
    return minimum


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceAuthority:
    """Frozen chronology and evidence split for one continuous-dynamics case."""

    dataset_id: str
    case_id: str
    data_sha256: str
    n_windows: int
    state_dimension: int
    fit_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    representation_fit_indices: tuple[int, ...]
    latent_dimension: int
    time_step_policy: TimeStepPolicy
    expected_window_seconds: float
    expected_step_seconds: float
    step_tolerance_seconds: float
    purge_seconds: float
    missing_data_policy: MissingDataPolicy = "reject"
    upstream_authority_fingerprint: str | None = None
    source_revisions: Mapping[str, str] = field(default_factory=dict)
    case_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.case_id:
            raise ValueError("dataset_id and case_id must be non-empty")
        if len(self.data_sha256) != 64:
            raise ValueError("data_sha256 must be a SHA-256 hex digest")
        if self.n_windows < 2 or self.state_dimension < 1:
            raise ValueError("n_windows/state_dimension must be positive")
        if self.latent_dimension != self.state_dimension:
            raise ValueError("latent_dimension must equal the frozen state tensor dimension")
        if self.time_step_policy != "fixed":
            raise ValueError("v1 trajectory authority supports time_step_policy='fixed' only")
        if self.missing_data_policy != "reject":
            raise ValueError("v1 trajectory authority supports missing_data_policy='reject' only")
        if self.expected_window_seconds <= 0 or self.expected_step_seconds <= 0:
            raise ValueError("expected window and step durations must be positive")
        if self.step_tolerance_seconds < 0 or self.purge_seconds < 0:
            raise ValueError("time tolerance and purge_seconds must be non-negative")

        fit = _indices(self.fit_indices, name="fit_indices")
        calibration = _indices(self.calibration_indices, name="calibration_indices")
        evaluation = _indices(self.evaluation_indices, name="evaluation_indices")
        representation = _indices(
            self.representation_fit_indices,
            name="representation_fit_indices",
        )
        if not fit or not evaluation:
            raise ValueError("fit_indices and evaluation_indices must be non-empty")
        if not representation:
            raise ValueError("representation_fit_indices must be non-empty")
        fit_set, calibration_set, evaluation_set = set(fit), set(calibration), set(evaluation)
        if fit_set & calibration_set or fit_set & evaluation_set or calibration_set & evaluation_set:
            raise ValueError("fit/calibration/evaluation indices must be mutually disjoint")
        if not set(representation).issubset(fit_set):
            raise ValueError("representation_fit_indices must be a subset of fit_indices")
        if any(index >= self.n_windows for index in fit + calibration + evaluation + representation):
            raise ValueError("authority contains out-of-range indices")

        object.__setattr__(self, "fit_indices", fit)
        object.__setattr__(self, "calibration_indices", calibration)
        object.__setattr__(self, "evaluation_indices", evaluation)
        object.__setattr__(self, "representation_fit_indices", representation)
        object.__setattr__(self, "source_revisions", MappingProxyType(dict(self.source_revisions)))
        object.__setattr__(self, "case_metadata", MappingProxyType(dict(self.case_metadata)))

    @classmethod
    def from_data(
        cls,
        data: TrajectoryEvidenceData,
        *,
        case_id: str,
        fit_indices: Any,
        calibration_indices: Any,
        evaluation_indices: Any,
        representation_fit_indices: Any,
        latent_dimension: int,
        expected_window_seconds: float,
        expected_step_seconds: float,
        time_step_policy: TimeStepPolicy = "fixed",
        step_tolerance_seconds: float = 1e-6,
        purge_seconds: float = 0.0,
        upstream_authority_fingerprint: str | None = None,
        source_revisions: Mapping[str, str] | None = None,
        case_metadata: Mapping[str, Any] | None = None,
    ) -> "TrajectoryEvidenceAuthority":
        authority = cls(
            dataset_id=data.dataset_id,
            case_id=case_id,
            data_sha256=data.data_sha256,
            n_windows=data.n_windows,
            state_dimension=data.state_dimension,
            fit_indices=_indices(fit_indices, name="fit_indices"),
            calibration_indices=_indices(calibration_indices, name="calibration_indices"),
            evaluation_indices=_indices(evaluation_indices, name="evaluation_indices"),
            representation_fit_indices=_indices(
                representation_fit_indices,
                name="representation_fit_indices",
            ),
            latent_dimension=int(latent_dimension),
            time_step_policy=time_step_policy,
            expected_window_seconds=float(expected_window_seconds),
            expected_step_seconds=float(expected_step_seconds),
            step_tolerance_seconds=float(step_tolerance_seconds),
            purge_seconds=float(purge_seconds),
            upstream_authority_fingerprint=upstream_authority_fingerprint,
            source_revisions={} if source_revisions is None else source_revisions,
            case_metadata={} if case_metadata is None else case_metadata,
        )
        authority.restore(data)
        return authority

    def _role_indices(self, role: TrajectoryRole) -> tuple[int, ...]:
        if role == "fit":
            return self.fit_indices
        if role == "calibration":
            return self.calibration_indices
        if role == "evaluation":
            return self.evaluation_indices
        raise ValueError(f"unsupported role={role!r}")

    def _validate_data_identity(self, data: TrajectoryEvidenceData) -> None:
        if data.dataset_id != self.dataset_id:
            raise ValueError("trajectory dataset_id differs from authority")
        if data.n_windows != self.n_windows or data.state_dimension != self.state_dimension:
            raise ValueError("trajectory tensor shape differs from authority")
        if data.data_sha256 != self.data_sha256:
            raise ValueError("trajectory evidence data SHA-256 differs from authority")

    def transition_pairs(self, data: TrajectoryEvidenceData, role: TrajectoryRole) -> Array:
        """Return legal adjacent transitions entirely inside one evidence role.

        A gap larger than the declared stride breaks a trajectory block. A pair is
        exposed only when its start-time delta matches the declared stride.
        """

        self._validate_data_identity(data)
        pairs: list[tuple[int, int]] = []
        for values in _by_trajectory(data, self._role_indices(role)).values():
            for left, right in zip(values[:-1], values[1:]):
                delta = float(data.start_times_s[right] - data.start_times_s[left])
                if abs(delta - self.expected_step_seconds) <= self.step_tolerance_seconds:
                    pairs.append((left, right))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def restore(self, data: TrajectoryEvidenceData) -> "TrajectoryEvidenceAuthority":
        """Revalidate exact tensor identity, chronology, window geometry, and purge gaps."""

        self._validate_data_identity(data)
        selected = self.fit_indices + self.calibration_indices + self.evaluation_indices
        if any(not bool(data.valid_mask[index]) for index in selected):
            raise ValueError("authority includes invalid/missing trajectory windows")

        # Duplicate starts make adjacency ambiguous even if the duplicate rows were
        # assigned to different evidence roles.
        ids = data.trajectory_ids.astype(str)
        for trajectory_id in sorted(set(ids.tolist())):
            indices = np.flatnonzero(ids == trajectory_id)
            starts = np.sort(data.start_times_s[indices])
            if len(starts) > 1 and np.any(np.diff(starts) <= 0):
                raise ValueError(f"trajectory {trajectory_id!r} has duplicate/non-increasing starts")

        durations = data.stop_times_s[np.asarray(selected)] - data.start_times_s[np.asarray(selected)]
        if np.any(np.abs(durations - self.expected_window_seconds) > self.step_tolerance_seconds):
            raise ValueError("selected windows violate expected_window_seconds")

        roles = {
            "fit": self.fit_indices,
            "calibration": self.calibration_indices,
            "evaluation": self.evaluation_indices,
        }
        role_names = tuple(roles)
        for i, left_name in enumerate(role_names):
            for right_name in role_names[i + 1 :]:
                left, right = roles[left_name], roles[right_name]
                if not left or not right:
                    continue
                separation = _minimum_interval_separation(data, left, right)
                if separation < self.purge_seconds - self.step_tolerance_seconds:
                    raise ValueError(
                        f"temporal leakage between {left_name} and {right_name}: "
                        f"minimum separation={separation:.9g}s < purge={self.purge_seconds:.9g}s"
                    )

        # Gaps larger than the declared stride are legitimate block boundaries. Gaps
        # *smaller* than the stride reveal a denser/overlapping temporal lattice than
        # the contract claims and therefore fail closed.
        for role, indices in roles.items():
            for values in _by_trajectory(data, indices).values():
                for left, right in zip(values[:-1], values[1:]):
                    delta = float(data.start_times_s[right] - data.start_times_s[left])
                    if delta < self.expected_step_seconds - self.step_tolerance_seconds:
                        raise ValueError(
                            f"{role} trajectory contains start-time delta {delta:.9g}s "
                            f"smaller than declared step {self.expected_step_seconds:.9g}s"
                        )

        if len(self.transition_pairs(data, "fit")) == 0:
            raise ValueError("fit authority contains no legal within-trajectory transitions")
        if len(self.transition_pairs(data, "evaluation")) == 0:
            raise ValueError("evaluation authority contains no legal within-trajectory transitions")
        return self

    @property
    def authority_fingerprint(self) -> str:
        payload = self.to_dict(include_fingerprint=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(
        self,
        *,
        include_fingerprint: bool = True,
        data: TrajectoryEvidenceData | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "case_id": self.case_id,
            "data_sha256": self.data_sha256,
            "n_windows": self.n_windows,
            "state_dimension": self.state_dimension,
            "latent_dimension": self.latent_dimension,
            "fit_indices": list(self.fit_indices),
            "calibration_indices": list(self.calibration_indices),
            "evaluation_indices": list(self.evaluation_indices),
            "representation_fit_indices": list(self.representation_fit_indices),
            "time_step_policy": self.time_step_policy,
            "expected_window_seconds": self.expected_window_seconds,
            "expected_step_seconds": self.expected_step_seconds,
            "step_tolerance_seconds": self.step_tolerance_seconds,
            "purge_seconds": self.purge_seconds,
            "missing_data_policy": self.missing_data_policy,
            "upstream_authority_fingerprint": self.upstream_authority_fingerprint,
            "source_revisions": dict(self.source_revisions),
            "case_metadata": dict(self.case_metadata),
        }
        if data is not None:
            self.restore(data)
            payload["transition_counts"] = {
                role: int(len(self.transition_pairs(data, role)))
                for role in ("fit", "calibration", "evaluation")
            }
        if include_fingerprint:
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryEvidenceAuthority":
        authority = cls(
            dataset_id=str(payload["dataset_id"]),
            case_id=str(payload["case_id"]),
            data_sha256=str(payload["data_sha256"]),
            n_windows=int(payload["n_windows"]),
            state_dimension=int(payload["state_dimension"]),
            fit_indices=tuple(int(v) for v in payload["fit_indices"]),
            calibration_indices=tuple(int(v) for v in payload.get("calibration_indices", [])),
            evaluation_indices=tuple(int(v) for v in payload["evaluation_indices"]),
            representation_fit_indices=tuple(int(v) for v in payload["representation_fit_indices"]),
            latent_dimension=int(payload["latent_dimension"]),
            time_step_policy=str(payload.get("time_step_policy", "fixed")),  # type: ignore[arg-type]
            expected_window_seconds=float(payload["expected_window_seconds"]),
            expected_step_seconds=float(payload["expected_step_seconds"]),
            step_tolerance_seconds=float(payload.get("step_tolerance_seconds", 1e-6)),
            purge_seconds=float(payload.get("purge_seconds", 0.0)),
            missing_data_policy=str(payload.get("missing_data_policy", "reject")),  # type: ignore[arg-type]
            upstream_authority_fingerprint=(
                None
                if payload.get("upstream_authority_fingerprint") is None
                else str(payload["upstream_authority_fingerprint"])
            ),
            source_revisions=dict(payload.get("source_revisions", {})),
            case_metadata=dict(payload.get("case_metadata", {})),
            schema_version=int(payload.get("schema_version", 1)),
        )
        expected = payload.get("authority_fingerprint")
        if expected is not None and str(expected) != authority.authority_fingerprint:
            raise ValueError("authority_fingerprint does not match serialized content")
        return authority


def load_trajectory_contract_descriptor(
    descriptor_path: str | Path,
) -> tuple[TrajectoryEvidenceData, TrajectoryEvidenceAuthority]:
    """Load a portable JSON + NumPy trajectory contract and fully validate it.

    File paths are resolved relative to the descriptor but are not included in the
    scientific fingerprint. Identity is content-addressed through the exact state
    tensor and temporal metadata.
    """

    path = Path(descriptor_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported trajectory contract schema_version")
    root = path.parent

    def load(name: str) -> Array:
        relative = payload.get("data", {}).get(name)
        if not relative:
            raise ValueError(f"trajectory contract missing data.{name}")
        return np.load(root / str(relative), allow_pickle=False)

    valid_relative = payload.get("data", {}).get("valid_mask")
    data = TrajectoryEvidenceData(
        dataset_id=str(payload["dataset_id"]),
        states=load("states"),
        trajectory_ids=load("trajectory_ids"),
        start_times_s=load("start_times_s"),
        stop_times_s=load("stop_times_s"),
        valid_mask=(
            None
            if not valid_relative
            else np.load(root / str(valid_relative), allow_pickle=False)
        ),
        metadata=dict(payload.get("data_metadata", {})),
    )

    split = dict(payload.get("split", {}))
    authority = TrajectoryEvidenceAuthority.from_data(
        data,
        case_id=str(payload["case_id"]),
        fit_indices=np.asarray(split.get("fit_indices", []), dtype=np.int64),
        calibration_indices=np.asarray(split.get("calibration_indices", []), dtype=np.int64),
        evaluation_indices=np.asarray(split.get("evaluation_indices", []), dtype=np.int64),
        representation_fit_indices=np.asarray(
            split.get("representation_fit_indices", []),
            dtype=np.int64,
        ),
        latent_dimension=int(payload["latent_dimension"]),
        time_step_policy=str(payload.get("time_step_policy", "fixed")),  # type: ignore[arg-type]
        expected_window_seconds=float(payload["expected_window_seconds"]),
        expected_step_seconds=float(payload["expected_step_seconds"]),
        step_tolerance_seconds=float(payload.get("step_tolerance_seconds", 1e-6)),
        purge_seconds=float(payload.get("purge_seconds", 0.0)),
        upstream_authority_fingerprint=(
            None
            if payload.get("upstream_authority_fingerprint") is None
            else str(payload["upstream_authority_fingerprint"])
        ),
        source_revisions=dict(payload.get("source_revisions", {})),
        case_metadata=dict(payload.get("case_metadata", {})),
    )
    return data, authority
