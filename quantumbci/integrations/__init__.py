"""Optional bridges to external neural research ecosystems."""

from .neuros import (
    DensityGeometryTransform,
    NeurOSEvidenceBinding,
    NeurOSFoundationEncoder,
    NeurOSUnavailableError,
    bind_neuros_evidence,
    neuros_integration_status,
)

__all__ = [
    "DensityGeometryTransform",
    "NeurOSEvidenceBinding",
    "NeurOSFoundationEncoder",
    "NeurOSUnavailableError",
    "bind_neuros_evidence",
    "neuros_integration_status",
]
