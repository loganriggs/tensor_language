import pytest

import terminal_copy_interaction_cube_v1 as contract


def test_exact_candidate_bank_and_missing_arms():
    assert len(contract.arms()) == 16
    assert len(set(contract.candidate_names())) == 16
    assert len(contract.missing_from_e4()) == 10
    assert sum(len(arm) == 2 for arm in contract.missing_from_e4()) == 6
    assert sum(len(arm) == 3 for arm in contract.missing_from_e4()) == 4


def test_mobius_recovers_known_signed_polynomial_and_order_predictions():
    terms = {
        ("L5H5",): 0.10,
        ("L7H3",): 0.02,
        ("L5H5", "L8H3"): 0.30,
        ("L7H3", "L8H3", "L8H4"): -0.05,
        tuple(contract.HEADS): 0.20,
    }
    values = {}
    for arm in contract.arms():
        values[arm] = sum(value for term, value in terms.items() if set(term) <= set(arm))
    result = contract.analyze_effect_cube(values)
    for arm in contract.arms():
        assert result["mobius"][arm] == pytest.approx(terms.get(arm, 0.0))
    assert result["full_minus_singleton_sum"] == pytest.approx(0.45)
    assert result["full_prediction_through_order"][2] == pytest.approx(0.42)
    assert result["full_prediction_through_order"][4] == pytest.approx(0.57)
    assert sum(result["shapley"].values()) == pytest.approx(0.57)


def test_native_effect_must_be_zero_and_cube_must_be_complete():
    values = {arm: 0.0 for arm in contract.arms()}
    values[()] = 0.01
    with pytest.raises(ValueError, match="native-baseline"):
        contract.analyze_effect_cube(values)
    del values[("L5H5",)]
    with pytest.raises(ValueError, match="factorial cube mismatch"):
        contract.analyze_effect_cube(values)


def test_scaled_curve_reports_only_nonlinearity_not_a_mechanism():
    result = contract.analyze_scaled_full_curve({
        0.25: 0.08, 0.5: 0.20, 0.75: 0.43, 1.0: 0.80,
    })
    assert result["linear_secant_prediction"][0.5] == pytest.approx(0.40)
    assert result["nonlinear_residual"][0.5] == pytest.approx(-0.20)
    assert result["max_abs_nonlinear_residual"] == pytest.approx(0.20)
    with pytest.raises(ValueError, match="exactly the frozen amplitudes"):
        contract.analyze_scaled_full_curve({0.5: 0.2, 1.0: 0.8})
