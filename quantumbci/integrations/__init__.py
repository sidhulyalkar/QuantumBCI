"""Optional bridges to external neural research ecosystems."""

from .neuros import (
    DensityGeometryTransform,
    NeurOSEvidenceBinding,
    NeurOSFoundationEncoder,
    NeurOSUnavailableError,
    bind_neuros_evidence,
    neuros_integration_status,
)
from .neuros_mechint import (
    MixWithMaximallyMixedState,
    PermuteDensityBasis,
    RemoveDensityOffDiagonals,
    run_neuros_mechint_input_audit,
)

__all__ = [
    "DensityGeometryTransform",
    "MixWithMaximallyMixedState",
    "NeurOSEvidenceBinding",
    "NeurOSFoundationEncoder",
    "NeurOSUnavailableError",
    "PermuteDensityBasis",
    "RemoveDensityOffDiagonals",
    "bind_neuros_evidence",
    "neuros_integration_status",
    "run_neuros_mechint_input_audit",
]
