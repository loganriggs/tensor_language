import pytest

from capability_population_contract import (
    CapabilityPopulationContractError,
    assert_declared_counts,
    derive_panel_contract,
)


def rows(a1=16, a2=16):
    return [
        {"row_id": f"{panel}-{index}", "transform_id": panel}
        for panel, count in (("A1", a1), ("A2", a2))
        for index in range(count)
    ]


def test_v15_fraction_derives_14_of_16_for_each_panel():
    contract = derive_panel_contract(rows(), panels=("A1", "A2"), minimum_fraction=.875)
    assert contract["panel_counts"] == {"A1": 16, "A2": 16}
    assert contract["required_jointly_capable_counts"] == {"A1": 14, "A2": 14}


def test_copied_32_row_denominator_fails_before_model_execution():
    contract = derive_panel_contract(rows(), panels=("A1", "A2"), minimum_fraction=.875)
    with pytest.raises(CapabilityPopulationContractError, match="declared denominators"):
        assert_declared_counts(
            contract,
            declared_denominators={"A1": 32, "A2": 32},
            declared_required_counts={"A1": 28, "A2": 28},
        )


def test_declared_ratio_must_equal_derived_ratio():
    contract = derive_panel_contract(rows(10, 12), panels=("A1", "A2"), minimum_fraction=.8)
    with pytest.raises(CapabilityPopulationContractError, match="declared required counts"):
        assert_declared_counts(
            contract,
            declared_denominators={"A1": 10, "A2": 12},
            declared_required_counts={"A1": 8, "A2": 8},
        )


def test_duplicate_and_empty_panels_fail_closed():
    duplicate = rows(1, 1)
    duplicate[1]["row_id"] = duplicate[0]["row_id"]
    with pytest.raises(CapabilityPopulationContractError, match="duplicate row IDs"):
        derive_panel_contract(duplicate, panels=("A1", "A2"), minimum_fraction=.8)
    with pytest.raises(CapabilityPopulationContractError, match="empty capability panels"):
        derive_panel_contract(rows(2, 0), panels=("A1", "A2"), minimum_fraction=.8)
