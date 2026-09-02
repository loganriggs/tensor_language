import torch

import equality_paired_response_kernel_rung471 as subject


def test_region_sums_partition_causal_prefix():
    contribution = torch.arange(10, dtype=torch.float64) - 4
    signed, absolute, future = subject.region_sums(contribution, query=8, predecessor=3)
    assert torch.equal(signed, torch.tensor([4., -1., 6., -9.], dtype=torch.float64))
    assert torch.equal(absolute, torch.tensor([4., 1., 6., 9.], dtype=torch.float64))
    assert future == 5
    assert signed.sum() == contribution[:9].sum()


def test_selection_is_first_two_per_document():
    rows = torch.tensor([[1, 2, 1, 1, 9], [3, 3, 4, 3, 3]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool)
    selected = subject.select_coordinates(rows, mask, 0, 2)
    assert selected == [(0, 2, 0), (0, 3, 2), (1, 1, 0), (1, 3, 1)]


def test_metrics_compare_against_frozen_context_control():
    target = torch.tensor([0., 1., 2., 3.])
    prediction = target.clone()
    control = torch.ones(4)
    report = subject._metrics(target, prediction, control)
    assert report["pearson"] > .999
    assert abs(report["rmse_improvement"] - 1) < 1e-12


def test_scale_is_frozen_least_squares():
    local = torch.tensor([1., -2., 3., -4.])
    exact = 1.75 * local
    assert abs(subject._fit_scale(local, exact) - 1.75) < 1e-12
