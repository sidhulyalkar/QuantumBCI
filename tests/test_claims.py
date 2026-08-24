from quantumbci.claims import ClaimClass, mechanism_card


def test_mechanism_claim_classes_are_explicit():
    assert mechanism_card("density_geometry").claim_class is ClaimClass.QUANTUM_INSPIRED
    assert mechanism_card("qft_sampling").claim_class is ClaimClass.QUANTUM_ALGORITHM
    assert all(mechanism_card(name).falsifiers for name in ("density_geometry", "lindblad_latent_dynamics", "contextual_measurement", "qft_sampling"))
