import quantumbci

from quantumbci.api_contract import (
    API_CONTRACT_VERSION,
    API_STABILITY,
    COMPATIBILITY_CANDIDATE_ROOT_API,
    missing_compatibility_candidate_exports,
)


def test_pre_1_0_compatibility_candidate_root_api_is_present() -> None:
    assert API_CONTRACT_VERSION == 1
    assert API_STABILITY == "pre-1.0-compatibility-candidate"
    assert len(COMPATIBILITY_CANDIDATE_ROOT_API) == len(set(COMPATIBILITY_CANDIDATE_ROOT_API))
    assert missing_compatibility_candidate_exports(quantumbci) == ()


def test_compatibility_candidate_exports_are_public_root_exports() -> None:
    exported = set(quantumbci.__all__)
    missing = sorted(set(COMPATIBILITY_CANDIDATE_ROOT_API) - exported)
    assert not missing, f"compatibility-candidate names missing from quantumbci.__all__: {missing}"
