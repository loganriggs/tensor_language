import numpy as np
import pytest

from mlp2_implicit_folded_tensor import (
    analyze_bilin18_mlp2_factors,
    analyze_folded_factors,
    balance_product_factors,
    dense_symmetric_core_product_threshold,
    down_svd_price,
    hosvd_relative_error_upper_bound,
    implicit_folded_mode_grams,
    native_mlp_price,
    spectrum_from_psd_gram,
    symmetric_tucker_price,
)


def _explicit_tensor(output: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    raw = np.einsum("ok,ki,kj->oij", output, left, right, optimize=True)
    return 0.5 * (raw + raw.swapaxes(1, 2))


def _explicit_mode_grams(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_unfolding = tensor.reshape(tensor.shape[0], -1)
    input_one = tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
    input_two = tensor.transpose(2, 0, 1).reshape(tensor.shape[2], -1)
    return (
        output_unfolding @ output_unfolding.T,
        input_one @ input_one.T,
        input_two @ input_two.T,
    )


def _assert_spectra_equal(first, second, *, atol: float = 2e-10) -> None:
    np.testing.assert_allclose(first.singular_values, second.singular_values, atol=atol, rtol=2e-10)
    np.testing.assert_allclose(first.eigenvalues, second.eigenvalues, atol=atol, rtol=2e-10)
    assert first.energy_ranks == second.energy_ranks
    assert first.numerical_rank == second.numerical_rank


def test_implicit_folded_grams_match_explicit_symmetric_tensor() -> None:
    rng = np.random.default_rng(1202)
    output = rng.normal(size=(4, 7))
    left = rng.normal(size=(7, 5))
    right = rng.normal(size=(7, 5))
    tensor = _explicit_tensor(output, left, right)
    explicit_output, explicit_input_one, explicit_input_two = _explicit_mode_grams(tensor)

    grams = implicit_folded_mode_grams(output, left, right)
    np.testing.assert_allclose(grams.output, explicit_output, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(grams.input, explicit_input_one, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(grams.input, explicit_input_two, atol=2e-10, rtol=2e-10)
    assert grams.input_modes_shared

    output_spectrum = spectrum_from_psd_gram(grams.output)
    input_spectrum = spectrum_from_psd_gram(grams.input)
    np.testing.assert_allclose(
        output_spectrum.singular_values,
        np.linalg.svd(tensor.reshape(4, -1), compute_uv=False),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        input_spectrum.singular_values,
        np.linalg.svd(tensor.transpose(1, 0, 2).reshape(5, -1), compute_uv=False),
        atol=2e-10,
    )


def test_balancing_preserves_tensor_and_equalizes_three_factor_norms() -> None:
    rng = np.random.default_rng(44)
    output = rng.normal(size=(3, 6))
    left = rng.normal(size=(6, 4))
    right = rng.normal(size=(6, 4))
    original = _explicit_tensor(output, left, right)
    balanced = balance_product_factors(output, left, right)

    np.testing.assert_allclose(
        _explicit_tensor(balanced.output, balanced.left, balanced.right),
        original,
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(np.linalg.norm(balanced.output, axis=0), balanced.common_norms)
    np.testing.assert_allclose(np.linalg.norm(balanced.left, axis=1), balanced.common_norms)
    np.testing.assert_allclose(np.linalg.norm(balanced.right, axis=1), balanced.common_norms)
    assert np.all(balanced.active_mask)


def test_authoritative_spectra_survive_scale_sign_swap_and_permutation_gauges() -> None:
    rng = np.random.default_rng(771)
    output = rng.normal(size=(4, 8))
    left = rng.normal(size=(8, 5))
    right = rng.normal(size=(8, 5))
    bias = rng.normal(size=4)
    levels = (0.8, 0.95, 0.99)
    base = analyze_folded_factors(output, left, right, bias, energy_levels=levels)

    scale_left = np.exp(rng.uniform(-7.0, 7.0, size=8)) * rng.choice([-1.0, 1.0], size=8)
    scale_right = np.exp(rng.uniform(-7.0, 7.0, size=8)) * rng.choice([-1.0, 1.0], size=8)
    permutation = rng.permutation(8)
    changed_left = (left * scale_left[:, None])[permutation]
    changed_right = (right * scale_right[:, None])[permutation]
    changed_output = (output / (scale_left * scale_right)[None, :])[:, permutation]
    changed = analyze_folded_factors(
        changed_output,
        changed_right,
        changed_left,
        bias,
        energy_levels=levels,
    )

    _assert_spectra_equal(base.balanced_down, changed.balanced_down)
    _assert_spectra_equal(base.folded_output, changed.folded_output)
    _assert_spectra_equal(base.folded_input, changed.folded_input)
    assert base.native_price == changed.native_price
    assert [point.price for point in base.hosvd_price_points] == [
        point.price for point in changed.hosvd_price_points
    ]


def test_rank_one_down_can_hide_rank_two_folded_input_mode() -> None:
    output = np.ones((1, 2))
    left = np.eye(2)
    right = np.eye(2)
    result = analyze_folded_factors(output, left, right, np.array([3.0]))

    assert result.balanced_down.numerical_rank == 1
    assert result.folded_output.numerical_rank == 1
    assert result.folded_input.numerical_rank == 2
    assert result.folded_input.energy_ranks["r95"] == 2
    assert result.bias_preserved


def test_hosvd_tail_bound_contains_explicit_projection_error() -> None:
    rng = np.random.default_rng(98)
    output = rng.normal(size=(3, 5))
    left = rng.normal(size=(5, 4))
    right = rng.normal(size=(5, 4))
    tensor = _explicit_tensor(output, left, right)
    grams = implicit_folded_mode_grams(output, left, right)
    output_spectrum = spectrum_from_psd_gram(grams.output)
    input_spectrum = spectrum_from_psd_gram(grams.input)
    output_values, output_vectors = np.linalg.eigh(grams.output)
    input_values, input_vectors = np.linalg.eigh(grams.input)
    output_basis = output_vectors[:, np.argsort(output_values)[::-1][:2]]
    input_basis = input_vectors[:, np.argsort(input_values)[::-1][:2]]
    core = np.einsum(
        "oa,ip,jq,oij->apq",
        output_basis,
        input_basis,
        input_basis,
        tensor,
        optimize=True,
    )
    projected = np.einsum(
        "oa,ip,jq,apq->oij",
        output_basis,
        input_basis,
        input_basis,
        core,
        optimize=True,
    )
    # The verbose contraction above is intentionally avoided in production; this is a
    # tiny known-answer reconstruction of the orthogonal Tucker projection.
    relative_error = np.linalg.norm(tensor - projected) / np.linalg.norm(tensor)
    bound = hosvd_relative_error_upper_bound(
        output_spectrum, input_spectrum, output_rank=2, input_rank=2
    )
    assert relative_error <= bound + 2e-10
    assert 0.0 <= bound <= 1.0


def test_zero_gate_psd_failures_and_prices_preserve_bias() -> None:
    output = np.array([[1.0, 7.0], [2.0, -3.0]])
    left = np.array([[1.0, 2.0], [0.0, 0.0]])
    right = np.array([[3.0, -1.0], [5.0, 6.0]])
    result = analyze_folded_factors(output, left, right, np.array([0.3, -0.7]))
    assert result.active_products == 1
    assert result.zero_products == 1
    assert result.down_price_points[0][1].bilinear_products_per_token == 1
    assert result.native_price.bias_values == 2

    with pytest.raises(ValueError, match="not symmetric"):
        spectrum_from_psd_gram(np.array([[1.0, 0.3], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        spectrum_from_psd_gram(np.diag([1.0, -0.1]))
    with pytest.raises(ValueError, match="finite"):
        analyze_folded_factors(output, left, right, np.array([np.nan, 0.0]))

    native = native_mlp_price(products=4608, input_dim=1152, output_dim=1152)
    assert native.stored_values == 15_926_400
    assert native.bias_values == 1152
    assert native.bilinear_products_per_token == 4608
    down = down_svd_price(products=4608, input_dim=1152, output_dim=1152, rank=512)
    assert down.stored_values == 2 * 4608 * 1152 + 512 * (4608 + 1152) + 1152
    assert down.bilinear_products_per_token == 4608
    tucker = symmetric_tucker_price(
        input_dim=1152, output_dim=1152, input_rank=95, output_rank=64
    )
    assert tucker.bilinear_products_per_token == 4560
    assert tucker.stored_values == 1152 * 95 + 1152 * 64 + 64 * 4560 + 1152
    assert dense_symmetric_core_product_threshold(4608) == 95


def test_strict_mlp2_entry_point_rejects_non_bilin18_shapes() -> None:
    with pytest.raises(ValueError, match="MLP2 Down"):
        analyze_bilin18_mlp2_factors(
            np.ones((2, 3)), np.ones((3, 2)), np.ones((3, 2)), np.ones(2)
        )
