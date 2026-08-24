from __future__ import annotations

import numpy as np

from quantumbci.equivalence import (
    audit_density_covariance_equivalence,
    audit_embedding_batch,
    trace_normalized_second_moment,
)
from quantumbci.states import density_from_samples


def test_real_density_is_trace_normalized_covariance() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 8))
    audit = audit_density_covariance_equivalence(x)
    assert audit.equivalent_within_tolerance is True
    assert audit.novel_information is False
    assert audit.max_abs_error < 1e-12
    np.testing.assert_allclose(
        density_from_samples(x),
        trace_normalized_second_moment(x),
        atol=1e-12,
    )


def test_complex_density_is_trace_normalized_hermitian_moment() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 6)) + 1j * rng.normal(size=(48, 6))
    audit = audit_density_covariance_equivalence(x)
    assert audit.equivalent_within_tolerance is True
    assert audit.max_abs_error < 1e-12


def test_batch_audit_has_hard_information_ceiling() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(20, 16, 5))
    audit = audit_embedding_batch(embeddings)
    assert audit.n_examples == 20
    assert audit.equivalent_within_tolerance is True
    assert audit.novel_information is False
    assert audit.equivalence_class == "trace_normalized_hermitian_second_moment"
