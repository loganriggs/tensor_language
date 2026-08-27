import torch

from compile_mlp0_native_down_fit_v1 import (
    centered_null,
    physical_program,
    residual_cross_covariance,
)
from mlp0_native_down_program import compact_centered_codebook


def test_residual_cross_covariance_subtracts_state_baseline_term():
    covariance = torch.tensor([[2., 0.], [0., 3.]])
    down = torch.tensor([[1., 4.], [2., 5.]])
    state_sums = torch.tensor([[1., 2.], [3., 4.]])
    centroids = torch.tensor([[2., 0.], [0., 1.]])
    observed = residual_cross_covariance(covariance, down, state_sums, centroids, 10)
    expected = covariance @ down.T - state_sums.T @ centroids / 10
    assert torch.allclose(observed, expected)


def test_physical_program_absorbs_centering_into_one_intercept():
    generator = torch.Generator().manual_seed(2)
    coefficient = torch.randn(5, 3, generator=generator)
    basis, _ = torch.linalg.qr(torch.randn(3, 2, generator=generator))
    mean_h = torch.randn(5, generator=generator)
    mean_down = torch.randn(3, generator=generator)
    centroids = torch.randn(2, 3, generator=generator)
    masses = torch.tensor([3., 7.])
    program = physical_program(
        coefficient, basis, 2, mean_h, mean_down, centroids,
        torch.tensor([0, 1, 2]), masses,
    )
    left = program["left"].to(torch.bfloat16).float()
    right = program["right"].to(torch.bfloat16).float()
    code = program["centroids"].to(torch.bfloat16).float()
    mean_b = (code * masses[:, None]).sum(0) / masses.sum()
    reconstructed_mean = mean_b + program["intercept"] + left @ (right @ mean_h)
    assert torch.allclose(reconstructed_mean, mean_down, atol=2e-5, rtol=2e-5)


def test_structured_null_is_recentred_in_the_same_additive_gauge():
    table = torch.tensor([[1., 0.], [3., 0.], [5., 0.], [7., 0.]])
    count = torch.tensor([1., 2., 3., 4.])
    labels = torch.tensor([0, 0, 1, 2])
    codebook = compact_centered_codebook(table, count, labels, nominal_states=3)
    null, report = centered_null(codebook)
    assert report["fixed_points"] == 0
    assert torch.allclose(
        (null * codebook["masses"][:, None]).sum(0), torch.zeros(2), atol=1e-6
    )
