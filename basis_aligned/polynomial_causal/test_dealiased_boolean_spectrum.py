import numpy as np

import dealiased_boolean_spectrum as spectrum


def test_walsh_transform_reconstructs_and_obeys_parseval():
    rng = np.random.default_rng(4)
    values = rng.normal(size=spectrum.SIZE)
    coefficients = spectrum.coefficients(values)
    assert np.allclose(spectrum.reconstruct(coefficients), values)
    assert np.isclose(
        np.sum(coefficients[1:] ** 2), np.mean((values - values.mean()) ** 2),
    )


def test_known_degree_three_polynomial_has_only_declared_terms():
    coefficients = np.zeros(spectrum.SIZE)
    coefficients[0] = 5
    coefficients[1] = 2
    coefficients[3] = -4
    coefficients[1 | 8 | 16] = 7
    result = spectrum.spectrum_summary(spectrum.reconstruct(coefficients))
    assert np.isclose(result["degree_energy"]["1"], 4)
    assert np.isclose(result["degree_energy"]["2"], 16)
    assert np.isclose(result["degree_energy"]["3"], 49)
    assert result["degree_energy"]["4"] == result["degree_energy"]["5"] == 0
    assert result["best_k_term_curve"]["4"]["rmse"] == 0


def test_four_arms_map_to_exact_five_bit_binary_order():
    cost = {
        "e": np.arange(8), "a": 10 + np.arange(8),
        "m": 20 + np.arange(8), "am": 30 + np.arange(8),
    }
    values = spectrum.build_set_function(cost)
    assert np.array_equal(values[:8], cost["e"])
    assert np.array_equal(values[8:16], cost["a"])
    assert np.array_equal(values[16:24], cost["m"])
    assert np.array_equal(values[24:32], cost["am"])


def test_transfer_curve_separates_support_from_values():
    source_coeff = np.zeros(spectrum.SIZE)
    source_coeff[0], source_coeff[1], source_coeff[3] = 2, 4, 3
    target_coeff = np.zeros(spectrum.SIZE)
    target_coeff[0], target_coeff[1], target_coeff[3] = 7, 8, 6
    result = spectrum.transfer_curve(
        spectrum.reconstruct(source_coeff), spectrum.reconstruct(target_coeff),
    )
    assert result["2"]["target_refit_on_source_support_nre"] == 0
    assert result["2"]["direct_source_values_nre"] > 0


def test_subset_labels_are_physical_and_unambiguous():
    assert spectrum.subset_label(0) == "constant"
    assert spectrum.subset_label(1 | 8 | 16) == "MLP0*A3:8*M3:8"
