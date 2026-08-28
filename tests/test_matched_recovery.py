from __future__ import annotations

from copy import deepcopy

import pytest

from quantumbci.matched_recovery import (
    build_matched_classical_recovery_evidence,
    matched_classical_recovery_from_mapping,
)


def _evidence(*, higher_is_better: bool = True):
    return build_matched_classical_recovery_evidence(
        study_id="causal-study",
        participant_id="p1",
        occasion_id="s1",
        case_id="p1-s1",
        mechanism_id="lindblad_latent_dynamics",
        classical_model_id="matched_nonlinear_control",
        information_set_id="same-evidence-budget-v1",
        metric_name="held_out_score",
        higher_is_better=higher_is_better,
        baseline_metric=1.0 if higher_is_better else 0.2,
        ablated_metric=0.5 if higher_is_better else 0.8,
        recovered_metric=0.6 if higher_is_better else 0.7,
        candidate_evidence_fingerprint="candidate-evidence",
        classical_evidence_fingerprint="classical-evidence",
    )


def test_recovery_fraction_is_derived_from_metrics() -> None:
    evidence = _evidence()
    assert evidence.ablation_loss == pytest.approx(0.5)
    assert evidence.restored_loss == pytest.approx(0.1)
    assert evidence.recovery_fraction == pytest.approx(0.2)
    parsed = matched_classical_recovery_from_mapping(evidence.to_mapping())
    assert parsed.source_fingerprint == evidence.source_fingerprint
    assert parsed.as_causal_recovery().classical_recovery_fraction == pytest.approx(0.2)


def test_lower_is_better_metric_uses_correct_orientation() -> None:
    evidence = _evidence(higher_is_better=False)
    assert evidence.ablation_loss == pytest.approx(0.6)
    assert evidence.restored_loss == pytest.approx(0.1)
    assert evidence.recovery_fraction == pytest.approx(1.0 / 6.0)


def test_control_that_worsens_ablation_gets_zero_recovery_not_negative_credit() -> None:
    evidence = build_matched_classical_recovery_evidence(
        study_id="causal-study",
        participant_id="p1",
        occasion_id="s1",
        case_id="p1-s1",
        mechanism_id="lindblad_latent_dynamics",
        classical_model_id="bad-control",
        information_set_id="same-evidence-budget-v1",
        metric_name="held_out_score",
        higher_is_better=True,
        baseline_metric=1.0,
        ablated_metric=0.5,
        recovered_metric=0.4,
        candidate_evidence_fingerprint="candidate-evidence",
        classical_evidence_fingerprint="classical-evidence",
    )
    assert evidence.restored_loss == 0.0
    assert evidence.recovery_fraction == 0.0


def test_nonpositive_ablation_loss_fails_closed() -> None:
    with pytest.raises(ValueError, match="strictly positive candidate ablation loss"):
        build_matched_classical_recovery_evidence(
            study_id="causal-study",
            participant_id="p1",
            occasion_id="s1",
            case_id="p1-s1",
            mechanism_id="lindblad_latent_dynamics",
            classical_model_id="control",
            information_set_id="same-evidence-budget-v1",
            metric_name="held_out_score",
            higher_is_better=True,
            baseline_metric=0.5,
            ablated_metric=0.5,
            recovered_metric=0.5,
            candidate_evidence_fingerprint="candidate-evidence",
            classical_evidence_fingerprint="classical-evidence",
        )


def test_tampered_recovery_fraction_is_rejected() -> None:
    payload = _evidence().to_mapping()
    payload["classical_recovery_fraction"] = 0.99
    with pytest.raises(ValueError, match="derived field mismatch"):
        matched_classical_recovery_from_mapping(payload)


def test_tampered_metric_is_rejected_by_source_fingerprint() -> None:
    payload = deepcopy(_evidence().to_mapping())
    payload["recovered_metric"] = 0.9
    # Keep the derived values coherent with the tampered metric to isolate the source digest.
    payload["restored_loss"] = 0.4
    payload["classical_recovery_fraction"] = 0.8
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        matched_classical_recovery_from_mapping(payload)


def test_boolean_direction_is_strict() -> None:
    payload = _evidence().to_mapping()
    payload["higher_is_better"] = "false"
    with pytest.raises(TypeError, match="JSON boolean"):
        matched_classical_recovery_from_mapping(payload)
