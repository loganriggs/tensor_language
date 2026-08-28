import pytest
import torch

import mlp1_implicit_folded_tensor_v1 as ft


def _small(seed: int = 1, *, output: int = 4, hidden: int = 6, width: int = 3):
    generator = torch.Generator().manual_seed(seed)
    down = torch.randn(output, hidden, generator=generator, dtype=torch.float64)
    left = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    bias = torch.randn(output, generator=generator, dtype=torch.float64)
    return down, left, right, bias


def _materialize(down, left, right):
    unsym = torch.einsum("on,ni,nj->oij", down, left, right)
    return 0.5 * (unsym + unsym.transpose(1, 2))


def _explicit_grams(tensor):
    return (
        torch.einsum("oij,pij->op", tensor, tensor),
        torch.einsum("oij,okj->ik", tensor, tensor),
    )


def test_balance_preserves_function_tensor_and_separate_bias():
    down, left, right, bias = _small()
    original_bias = bias.clone()
    factors = ft.balance_factors(down, left, right, bias)
    x = torch.randn(11, left.shape[1], generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    before = (x @ left.T) * (x @ right.T)
    before = before @ down.T + bias
    torch.testing.assert_close(ft.bilinear_output(factors, x), before, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(
        _materialize(factors.down, factors.left, factors.right),
        _materialize(down, left, right), rtol=2e-13, atol=2e-13,
    )
    torch.testing.assert_close(factors.bias, original_bias, rtol=0, atol=0)
    assert factors.max_log_defect_after < 1e-14


def test_exact_implicit_grams_match_materialized_tensor_for_every_blocking():
    factors = ft.balance_factors(*_small(seed=3))
    expected_out, expected_in = _explicit_grams(
        _materialize(factors.down, factors.left, factors.right)
    )
    for block in (1, 2, 5, 99):
        actual_out, actual_in = ft.exact_folded_mode_grams(factors, hidden_block=block)
        torch.testing.assert_close(actual_out, expected_out, rtol=4e-13, atol=4e-13)
        torch.testing.assert_close(actual_in, expected_in, rtol=4e-13, atol=4e-13)


def test_known_diagonal_tensor_has_unit_spectra_and_full_hosvd_reconstructs():
    eye = torch.eye(3, dtype=torch.float64)
    factors = ft.balance_factors(eye, eye, eye, torch.tensor([1.0, 2.0, 3.0]))
    gout, gin = ft.exact_folded_mode_grams(factors, hidden_block=2)
    torch.testing.assert_close(gout, eye)
    torch.testing.assert_close(gin, eye)
    report = ft.folded_hosvd_spectra(factors, hidden_block=2)
    assert report["output_mode"]["squared_singular_values"] == [1.0, 1.0, 1.0]
    uo, ui = ft.hosvd_bases(gout, gin, output_rank=3, input_rank=3)
    core = ft.project_symmetric_hosvd_core(factors, uo, ui)
    reconstructed = torch.einsum("oa,ib,jc,abc->oij", uo, ui, ui, core)
    torch.testing.assert_close(reconstructed, _materialize(eye, eye, eye), rtol=1e-14, atol=1e-14)


def test_hidden_scale_sign_permutation_and_branch_swap_leave_grams_invariant():
    down, left, right, bias = _small(seed=4)
    base = ft.balance_factors(down, left, right, bias)
    expected = ft.exact_folded_mode_grams(base, hidden_block=2)
    hidden = down.shape[1]
    generator = torch.Generator().manual_seed(5)
    alpha = torch.exp(3.0 * torch.randn(hidden, generator=generator, dtype=torch.float64))
    beta = torch.exp(3.0 * torch.randn(hidden, generator=generator, dtype=torch.float64))
    alpha[::2].neg_()
    beta[1::2].neg_()
    permutation = torch.randperm(hidden, generator=generator)
    gauged = ft.balance_factors(
        (down / (alpha * beta).unsqueeze(0))[:, permutation],
        (left * alpha.unsqueeze(1))[permutation],
        (right * beta.unsqueeze(1))[permutation],
        bias,
    )
    actual = ft.exact_folded_mode_grams(gauged, hidden_block=3)
    swapped = ft.exact_folded_mode_grams(
        ft.balance_factors(down, right, left, bias), hidden_block=4
    )
    for observed in (actual, swapped):
        torch.testing.assert_close(observed[0], expected[0], rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(observed[1], expected[1], rtol=2e-12, atol=2e-12)


def test_balancing_is_stable_under_extreme_but_finite_reciprocal_gauge():
    down, left, right, bias = _small(seed=40)
    alpha = torch.tensor(
        [1e-100, -1e100, 1e-50, -1e50, 1e-20, -1e20], dtype=torch.float64
    )
    beta = torch.tensor(
        [-1e80, 1e-80, -1e40, 1e40, -1e10, 1e10], dtype=torch.float64
    )
    base = ft.balance_factors(down, left, right, bias)
    gauged = ft.balance_factors(
        down / (alpha * beta).unsqueeze(0),
        left * alpha.unsqueeze(1),
        right * beta.unsqueeze(1),
        bias,
    )
    assert torch.isfinite(gauged.term_norms).all()
    expected = ft.exact_folded_mode_grams(base, hidden_block=3)
    actual = ft.exact_folded_mode_grams(gauged, hidden_block=3)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-12, atol=2e-12)


def test_dead_terms_are_exactly_zeroed_without_bias_change():
    down, left, right, bias = _small(seed=6)
    left[2].zero_()
    factors = ft.balance_factors(down, left, right, bias)
    assert factors.dead_units == (2,)
    assert torch.count_nonzero(factors.down[:, 2]) == 0
    assert torch.count_nonzero(factors.left[2]) == 0
    assert torch.count_nonzero(factors.right[2]) == 0
    torch.testing.assert_close(factors.bias, bias, rtol=0, atol=0)


def test_balanced_down_svd_known_answer():
    down = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
    # Unit-norm L/R make the balanced Down columns all geometric means, so build
    # already-balanced factors to isolate the registered matrix calculation.
    left = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
    right = left.clone()
    factors = ft.balance_factors(down, left, right, torch.zeros(3))
    report = ft.balanced_down_svd(factors)
    assert report["squared_singular_values"] == [16.0, 4.0, 1.0]
    assert report["energy_ranks"]["0.900"] == 2
    assert report["energy_ranks"]["0.999"] == 3


def test_sparse_core_curve_uses_symmetric_energy_and_deterministic_ties():
    core = torch.zeros(2, 2, 2, dtype=torch.float64)
    core[0, 0, 1] = core[0, 1, 0] = 3.0  # folded energy 18
    core[1, 1, 1] = 4.0                 # folded energy 16
    core[0, 0, 0] = 1.0
    curve = ft.sparse_core_curve(core, (1, 2, 3), ambient_output=5, ambient_width=7)
    assert curve[0]["indices"] == [[0, 0, 1]]
    assert curve[1]["indices"] == [[0, 0, 1], [1, 1, 1]]
    assert curve[1]["active_input_pairs"] == 2
    assert curve[2]["retained_core_frobenius_fraction"] == 1.0

    tied = torch.zeros(2, 1, 1, dtype=torch.float64)
    tied[:, 0, 0] = 1.0
    assert ft.sparse_core_curve(tied, (1,), ambient_output=2, ambient_width=1)[0]["indices"] == [[0, 0, 0]]


def test_exact_price_formulas_separate_products_and_index_storage():
    assert ft.native_price(5, 7, 3) == {
        "float_storage": 82, "integer_storage": 0,
        "multiply_adds_per_token": 77, "bilinear_products_per_token": 7,
        "bias_additions_per_token": 5, "scalar_multiplications_per_token": 84,
    }
    assert ft.down_rank_price(5, 7, 3, 2)["float_storage"] == 71
    dense = ft.dense_tucker_price(5, 3, 2, 2)
    assert dense["float_storage"] == 27
    assert dense["bilinear_products_per_token"] == 3
    sparse = ft.sparse_tucker_price(5, 3, 2, 2, coefficients=2, active_pairs=1)
    assert sparse["float_storage"] == 23
    assert sparse["integer_storage"] == 6
    assert ft.cp_price(5, 3, 4)["float_storage"] == 49
    assert ft.cp_price(5, 3, 4)["bilinear_products_per_token"] == 4
    with pytest.raises(ValueError, match="active pair"):
        ft.sparse_tucker_price(5, 3, 2, 2, coefficients=1, active_pairs=0)


@pytest.mark.parametrize("mutation", ["shape", "nan", "dtype", "bias"])
def test_malformed_factors_fail_closed(mutation):
    down, left, right, bias = _small()
    if mutation == "shape":
        right = right[:-1]
    elif mutation == "nan":
        left[0, 0] = float("nan")
    elif mutation == "dtype":
        down = down.to(torch.int64)
    else:
        bias = bias[:-1]
    with pytest.raises((TypeError, ValueError)):
        ft.balance_factors(down, left, right, bias)


def test_non_cpu_factor_is_rejected_when_accelerator_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    down, left, right, bias = _small()
    with pytest.raises(ValueError, match="CPU"):
        ft.balance_factors(down.cuda(), left, right, bias)
