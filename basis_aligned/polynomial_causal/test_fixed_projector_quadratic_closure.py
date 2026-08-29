import torch

import fixed_projector_quadratic_closure as closure


def _block_diagonal_factors():
    left = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
    right = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    decoder = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    basis = torch.eye(4)[:, :2].contiguous()
    return left, right, decoder, basis


def test_exact_direct_sum_has_zero_leakage():
    values = _block_diagonal_factors()
    result = closure.estimate_mixed_block_leakage(*values, samples=256, seed=4)
    assert result.leakage < 1e-12


def test_cross_block_product_has_large_leakage():
    _, _, _, basis = _block_diagonal_factors()
    left = torch.tensor([[1., 0., 0., 0.]])
    right = torch.tensor([[0., 0., 1., 0.]])
    decoder = torch.tensor([[1.], [0.], [0.], [0.]])
    result = closure.estimate_mixed_block_leakage(
        left, right, decoder, basis, samples=1024, seed=5,
    )
    assert result.leakage > 0.99


def test_orthonormal_union_and_haar_are_deterministic():
    first = closure.haar_basis(12, 3, 9)
    assert torch.equal(first, closure.haar_basis(12, 3, 9))
    second = closure.haar_basis(12, 2, 10)
    union = closure.orthonormal_union(first, second)
    assert union.shape == (12, 5)
    assert float((union.T @ union - torch.eye(5)).abs().max()) < 1e-5


def test_reciprocal_product_gauge_leaves_estimate_fixed():
    left, right, decoder, basis = _block_diagonal_factors()
    one = closure.estimate_mixed_block_leakage(
        left, right, decoder, basis, samples=64, seed=6,
    )
    two = closure.estimate_mixed_block_leakage(
        7 * left, right / 7, decoder, basis, samples=64, seed=6,
    )
    assert abs(one.leakage - two.leakage) < 1e-12

