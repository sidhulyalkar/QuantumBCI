"""Minimal neurOS-style use of QuantumBCI's density transform.

The example uses a tiny frame-compatible dataclass so it runs in base QuantumBCI CI.
When neuros-core is installed, the same transform preserves a real SignalFrame via
``dataclasses.replace`` and is also discoverable as the ``quantumbci-density`` neurOS
transform plugin.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from quantumbci.integrations.neuros import DensityGeometryTransform


@dataclass(frozen=True)
class FrameLike:
    data: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


rng = np.random.default_rng(7)
frame = FrameLike(
    data=rng.normal(size=(8, 250)),
    metadata={"stream_id": "eeg-demo"},
)

observables = DensityGeometryTransform(output="observables").transform(frame)
print(observables.data)
print(observables.metadata)
