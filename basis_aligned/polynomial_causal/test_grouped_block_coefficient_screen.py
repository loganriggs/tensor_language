import math

import torch

import grouped_block_coefficient_screen as screen


def _factors(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    c_proj = torch.randn(5, 5, generator=generator, dtype=torch.float64)
    left = torch.randn(9, 5, generator=generator, dtype=torch.float64)
    right = torch.randn(9, 5, generator=generator, dtype=torch.float64)
    down = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    return c_proj, left, right, down


def test_balance_equalizes_norms_and_preserves_products():
    _, left, right, _ = _factors()
    balanced_left, balanced_right, _ = screen.balance_product_gauge(left, right)
    assert torch.allclose(
        balanced_left.norm(dim=1), balanced_right.norm(dim=1), atol=1e-12,
    )
    x = torch.randn(7, 5, generator=torch.Generator().manual_seed(1), dtype=left.dtype)
    original = torch.nn.functional.linear(x, left) * torch.nn.functional.linear(x, right)
    balanced = (
        torch.nn.functional.linear(x, balanced_left)
        * torch.nn.functional.linear(x, balanced_right)
    )
    assert torch.allclose(original, balanced, atol=1e-12)


def test_balance_is_minimum_along_scale_gauge():
    _, left, right, _ = _factors()
    balanced_left, balanced_right, _ = screen.balance_product_gauge(left, right)
    optimum = balanced_left.square().sum() + balanced_right.square().sum()
    for multiplier in (0.1, 0.5, 2.0, 10.0):
        candidate = (
            (multiplier * balanced_left).square().sum()
            + (balanced_right / multiplier).square().sum()
        )
        assert candidate > optimum


def test_weighted_interface_is_invariant_to_positive_native_scale_gauge():
    c_proj, left, right, down = _factors()
    generator = torch.Generator().manual_seed(2)
    scale = torch.exp(torch.randn(9, generator=generator, dtype=left.dtype))
    first, _ = screen.weighted_interface(c_proj, left, right, down)
    second, _ = screen.weighted_interface(
        c_proj, scale[:, None] * left, right / scale[:, None], down,
    )
    assert torch.allclose(first, second, atol=1e-11)


def test_spectrum_summary_matches_direct_svd():
    generator = torch.Generator().manual_seed(3)
    matrix = torch.randn(13, 5, generator=generator, dtype=torch.float64)
    summary = screen.spectrum_summary(matrix)
    singular = torch.linalg.svdvals(matrix)
    expected = singular.square()
    observed = torch.tensor(summary["singular_value_squared"], dtype=torch.float64)
    assert torch.allclose(observed, expected, atol=1e-10)
    assert math.isclose(
        summary["stable_rank"], float(expected.sum() / expected[0]), rel_tol=1e-12,
    )

