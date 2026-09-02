import torch

import equality_context_causal_state_rung470 as subject


def test_standardization_uses_fit_statistics():
    x = torch.tensor([
        [1., 1., 2., 3., 4., 5., 6., 7.],
        [1., 3., 4., 5., 6., 7., 8., 9.],
    ], dtype=torch.float64)
    mean, std = subject.standardize_fit(x)
    z = subject.standardize_apply(x, mean, std)
    assert torch.equal(z[:, 0], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(z[:, 1:].mean(0), torch.zeros(7, dtype=torch.float64))


def test_ridge_recovers_planted_rule():
    generator = torch.Generator().manual_seed(470)
    x = torch.randn(100, 8, generator=generator, dtype=torch.float64)
    x[:, 0] = 1
    beta = torch.arange(8, dtype=torch.float64) / 10
    y = x @ beta
    fitted = subject.fit_ridge(x, y, penalty=1e-10)
    assert torch.allclose(fitted, beta, atol=1e-8, rtol=1e-8)


def test_quadrant_baseline_is_fit_only():
    q = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    y = torch.tensor([1., 3., 2., 4., 10., 14., -2., 2.])
    means = subject.fit_quadrant(q, y)
    assert torch.equal(means, torch.tensor([2., 3., 12., 0.], dtype=torch.float64))


def test_prediction_metrics_scores_improvement():
    y = torch.tensor([0., 1., 2., 3.])
    prediction = torch.tensor([0., 1., 2., 3.])
    baseline = torch.ones(4)
    metrics = subject.prediction_metrics(y, prediction, baseline)
    assert metrics["pearson"] > .999
    assert abs(metrics["rmse_improvement"] - 1) < 1e-12


def test_aggregate_cells_handles_overlapping_context_masks():
    values = torch.tensor([1., 2., 3., 4.])
    memberships = torch.tensor([
        [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],
    ], dtype=torch.bool)
    assert torch.equal(subject.aggregate_cells(values, memberships),
                       torch.tensor([1.5, 3.5, 2., 3.], dtype=torch.float64))
