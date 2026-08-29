from __future__ import annotations

import pytest
import torch

import mlp2_cmr_v1_suffix_math as suffix


def fixture():
    generator = torch.Generator().manual_seed(19)
    mean = torch.randn(9, generator=generator, dtype=torch.float64)
    variance = torch.rand(9, generator=generator, dtype=torch.float64) + 0.2
    down = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    return generator, mean, variance, down


def test_centered_dual_write_has_exact_native_baseline_and_centered_gradient() -> None:
    generator, mean, variance, down = fixture()
    product = torch.randn(2, 3, 9, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    std, orientation, permutation = suffix.canonical_derangement(mean, variance, down, 7)
    alpha = torch.ones(2, 9, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(2, 9, dtype=torch.float64, requires_grad=True)
    write = suffix.centered_dual_write(
        product, mean, down, bias, alpha, beta, std, orientation, permutation,
    )
    native = torch.nn.functional.linear(product, down) + bias
    assert torch.equal(write, native)
    gradient = torch.autograd.grad(write.sum(), alpha, retain_graph=True)[0]
    expected = torch.einsum("bth,oh->bh", product - mean, down)
    assert torch.allclose(gradient, expected)
    assert torch.autograd.grad(write.sum(), beta)[0].shape == beta.shape


def test_derangement_is_reciprocal_gauge_invariant() -> None:
    generator, mean, variance, down = fixture()
    std, orientation, permutation = suffix.canonical_derangement(mean, variance, down, 11)
    scales = torch.tensor(
        [2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125, 16.0, -0.0625],
        dtype=torch.float64,
    )
    std2, orientation2, permutation2 = suffix.canonical_derangement(
        mean * scales, variance * scales.square(), down / scales, 11,
    )
    assert permutation2 == permutation
    canonical_down = down * std[None, :] * orientation[None, :]
    canonical_down2 = down / scales * std2[None, :] * orientation2[None, :]
    assert torch.allclose(canonical_down2, canonical_down, rtol=1e-12, atol=1e-12)


def test_derangement_is_permutation_equivariant() -> None:
    _, mean, variance, down = fixture()
    _, _, permutation = suffix.canonical_derangement(mean, variance, down, 13)
    order = (4, 1, 8, 0, 3, 7, 2, 6, 5)
    index = torch.tensor(order)
    _, _, replay = suffix.canonical_derangement(
        mean[index], variance[index], down[:, index], 13,
    )
    assert replay == suffix.mapped_permutation(permutation, order)


def test_hash_random_is_deterministic_gauge_and_permutation_equivariant() -> None:
    _, mean, variance, down = fixture()
    support = suffix.canonical_hash_random_support(mean, variance, down, 5, 20260829)
    assert support == suffix.canonical_hash_random_support(
        mean, variance, down, 5, 20260829,
    )
    assert len(set(support)) == 5
    scales = torch.tensor(
        [2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125, 16.0, -0.0625],
        dtype=torch.float64,
    )
    assert support == suffix.canonical_hash_random_support(
        mean * scales, variance * scales.square(), down / scales, 5, 20260829,
    )
    order = (4, 1, 8, 0, 3, 7, 2, 6, 5)
    index = torch.tensor(order)
    replay = suffix.canonical_hash_random_support(
        mean[index], variance[index], down[:, index], 5, 20260829,
    )
    assert {order[i] for i in replay} == set(support)


def test_support_and_rank_diagnostics() -> None:
    assert suffix.support_jaccard((0, 1, 2), (1, 2, 3)) == 0.5
    assert suffix.support_jaccard(
        torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]),
    ) == 0.5
    first = torch.tensor([1.0, 4.0, 2.0, 3.0])
    second = torch.tensor([2.0, 8.0, 4.0, 6.0])
    reverse = torch.tensor([4.0, 1.0, 3.0, 2.0])
    assert suffix.spearman(first, second) == pytest.approx(1.0)
    assert suffix.spearman(first, reverse) == pytest.approx(-1.0)
