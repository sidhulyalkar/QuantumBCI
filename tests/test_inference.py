from __future__ import annotations

import pytest

from quantumbci.inference import participant_effect_inference


def test_participant_effect_inference_is_deterministic_and_unit_level() -> None:
    values = [0.10, 0.20, 0.30, 0.40]
    first = participant_effect_inference(values, bootstrap_resamples=500, seed=17)
    second = participant_effect_inference(values, bootstrap_resamples=500, seed=17)
    assert first == second
    assert first.n_units == 4
    assert first.observed_mean == pytest.approx(0.25)
    assert first.positive_fraction == 1.0
    assert first.bootstrap_ci_lower <= first.observed_mean <= first.bootstrap_ci_upper
    assert first.sign_flip_method == "exact_sign_flip"
    assert first.sign_flip_draws == 16
    assert 0.0 <= first.sign_flip_pvalue_two_sided <= 1.0


def test_sign_flip_null_is_not_manufactured_as_significant() -> None:
    result = participant_effect_inference(
        [-0.3, -0.1, 0.1, 0.3], bootstrap_resamples=500, seed=3
    )
    assert result.observed_mean == pytest.approx(0.0)
    assert result.sign_flip_pvalue_two_sided == pytest.approx(1.0)


def test_participant_inference_rejects_pseudoreplication_shaped_single_unit() -> None:
    with pytest.raises(ValueError, match="at least two independent unit"):
        participant_effect_inference([0.2], bootstrap_resamples=500)


def test_participant_inference_rejects_invalid_bootstrap_request() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        participant_effect_inference([0.1, 0.2], bootstrap_resamples=10)
