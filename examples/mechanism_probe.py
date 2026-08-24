"""Minimal end-to-end example of an interpretable quantum-inspired mechanism probe."""

import numpy as np

from quantumbci.contextuality import commutator_norm, order_effect, projector
from quantumbci.interpretability import mechanism_delta, state_signature
from quantumbci.open_system import dephasing_collapse, evolve_lindblad
from quantumbci.states import density_from_samples

rng = np.random.default_rng(7)
latent_samples = rng.normal(size=(256, 2))
latent_samples[:, 1] = 0.75 * latent_samples[:, 0] + 0.25 * latent_samples[:, 1]
rho0 = density_from_samples(latent_samples)

hamiltonian = np.array([[0.0, 0.9], [0.9, 0.15]], dtype=complex)
collapse = [dephasing_collapse(2, 0, 0.5), dephasing_collapse(2, 1, 0.5)]
times = np.linspace(0.0, 1.5, 151)
trajectory = evolve_lindblad(rho0, hamiltonian, times, collapse_operators=collapse)

print("state delta", mechanism_delta(state_signature(trajectory[0]), state_signature(trajectory[-1])))

z = projector([1.0, 0.0])
x = projector([1.0, 1.0])
print("operator non-commutativity", commutator_norm(z, x))
print("order effect", order_effect(rho0, z, x))
