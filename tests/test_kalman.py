import numpy as np

from quantumbci.kalman import kalman_filter, qlsa_diagnostics


def test_kalman_tracks_constant_state_and_covariance_is_symmetric():
    measurements = np.array([0.9, 1.2, 0.95, 1.05, 1.0])
    estimates, covariances = kalman_filter(
        measurements,
        a=np.array([[1.0]]),
        h=np.array([[1.0]]),
        q=np.array([[0.01]]),
        r=np.array([[0.1]]),
        x0=np.array([0.0]),
        p0=np.array([[1.0]]),
    )
    assert abs(estimates[-1, 0] - 1.0) < 0.1
    assert np.allclose(covariances, covariances.transpose(0, 2, 1))
    assert np.all(np.linalg.eigvalsh(covariances) >= -1e-12)


def test_qlsa_diagnostics_make_readout_caveat_explicit():
    report = qlsa_diagnostics(np.diag([1.0, 2.0, 3.0, 4.0]))
    assert report.hermitian
    assert report.positive_definite
    assert report.power_of_two
    assert any("readout" in caveat.lower() for caveat in report.caveats)
