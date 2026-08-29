import pytest
import torch

import mlp2_error_rayleigh_metrics as metrics


def test_symmetric_jvp_and_fisher_shift_invariance():
    derivative = torch.tensor([[[1.0, -1.0, 0.0], [2.0, 0.0, -2.0]]])
    alpha = 0.125
    center = torch.randn(1, 2, 3)
    recovered = metrics.symmetric_jvp(center + alpha * derivative,
                                      center - alpha * derivative, alpha)
    assert torch.allclose(recovered, derivative.double())
    logits = torch.zeros_like(derivative)
    original = metrics.categorical_fisher_quadratic(logits, derivative)
    shifted = metrics.categorical_fisher_quadratic(logits, derivative + 7.0)
    assert torch.allclose(original, shifted)
    assert torch.allclose(original, torch.tensor([5.0 / 3.0], dtype=torch.float64))


def test_teacher_kl_and_normalized_energy():
    native = torch.tensor([[[2.0, 0.0], [1.0, -1.0]]])
    assert torch.equal(metrics.teacher_kl(native, native), torch.zeros(1, dtype=torch.float64))
    delta = torch.ones(2, 3, 4)
    reference = 2 * torch.ones_like(delta)
    assert torch.allclose(metrics.normalized_response_energy(delta, reference),
                          torch.full((2,), 0.25, dtype=torch.float64))


def test_spearman_with_ties_and_tangent_gate():
    x = torch.tensor([1.0, 2.0, 2.0, 4.0])
    assert metrics.spearman(x, x) == pytest.approx(1.0)
    passed = metrics.tangent_scale_gate(x, 1.1 * x)
    assert passed["passes"]
    failed = metrics.tangent_scale_gate(x, torch.tensor([4.0, 3.0, 2.0, 1.0]))
    assert not failed["passes"]


def test_fisher_kl_gate_exact_quadratic_and_failure():
    q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    alpha = 0.125
    observed = 0.5 * alpha ** 2 * q
    assert metrics.fisher_kl_gate(observed, q, alpha)["passes"]
    assert not metrics.fisher_kl_gate(2.0 * observed, q, alpha)["passes"]


def test_predictor_and_finite_interaction_gates():
    target = torch.arange(1.0, 9.0)
    local = target + torch.tensor([2.0, -2.0] * 4)
    final = target + torch.tensor([1.0, -1.0] * 4)
    full = target + torch.tensor([0.1, -0.1] * 4)
    result = metrics.predictor_gate(target, local, final, full)
    assert result["passes"]
    finite = metrics.finite_interaction_gate(
        torch.tensor([0.008, 0.007, 0.006]), torch.tensor([0.009, 0.0075, 0.007]),
    )
    assert finite["passes"]
    assert not metrics.finite_interaction_gate(
        torch.tensor([-0.008, 0.007, 0.006]), torch.tensor([0.009, 0.0075, 0.007]),
    )["passes"]


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        metrics.symmetric_jvp(torch.ones(2), torch.ones(3), 0.1)
    with pytest.raises(ValueError):
        metrics.spearman(torch.ones(4), torch.ones(4))
    with pytest.raises(ValueError):
        metrics.finite_interaction_gate(torch.ones(2), torch.ones(2))
