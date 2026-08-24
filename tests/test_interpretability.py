import numpy as np

from quantumbci.interpretability import ablation_sensitivity, bootstrap_stability, state_signature
from quantumbci.states import project_density_matrix


def test_ablation_sensitivity_identifies_used_feature():
    def model(x):
        return np.array([2.0 * x[0]])

    effects = ablation_sensitivity(model, np.array([3.0, 4.0]), [[0], [1]])
    assert effects[0] > 0
    assert effects[1] == 0


def test_state_signature_and_bootstrap_are_finite():
    rho = project_density_matrix(np.array([[0.7, 0.2], [0.2, 0.3]]))
    signature = state_signature(rho)
    assert np.isfinite(signature.entropy_bits)
    mean, interval = bootstrap_stability(np.array([1.0, 1.1, 0.9, 1.05]), n_boot=200, seed=1)
    assert interval[0] <= mean <= interval[1]
