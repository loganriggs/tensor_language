import math

import pytest
import torch

import equality_distributed_finite_transition_quotient_rung528_math as rung


def test_fit_signed_scale_and_metrics_are_exact():
    source = torch.tensor([1.0, -2.0, 3.0])
    target = -1.75 * source
    beta = rung.fit_signed_scale(target, source)
    metrics = rung.relation_metrics(target, source, beta)
    assert beta == pytest.approx(-1.75)
    assert metrics["cosine"] == pytest.approx(-1.0)
    assert metrics["relative_residual"] == pytest.approx(0.0)


def test_zero_source_scale_is_nonfinite():
    beta = rung.fit_signed_scale(torch.ones(3), torch.zeros(3))
    assert math.isnan(beta)


def test_factorial_interaction_uses_registered_order():
    effects = torch.tensor([[[1.0], [3.0], [7.0], [13.0]]])
    assert torch.equal(rung.factorial_interaction(effects), torch.tensor([[4.0]]))


def test_factorial_interaction_rejects_wrong_arm_count():
    with pytest.raises(ValueError, match="four arms"):
        rung.factorial_interaction(torch.zeros(2, 3, 5))


def test_pair_scale_is_fit_only_on_half_zero():
    generator = torch.Generator().manual_seed(5)
    source_circuit = torch.randn(2, 4, 32, generator=generator) * .01
    source_task = torch.randn(2, 4, 4, generator=generator) * .01
    target_circuit = 2.0 * source_circuit
    target_task = 2.0 * source_task
    target_circuit[1] = 3.0 * source_circuit[1]
    target_task[1] = 3.0 * source_task[1]
    result = rung.score_pair(target_circuit, source_circuit, target_task, source_task)
    assert result["beta"] == pytest.approx(2.0)
    assert result["circuit"][0]["relative_residual"] == pytest.approx(0.0)
    assert result["circuit"][1]["relative_residual"] == pytest.approx(1.0 / 3.0)
    assert result["passes_without_controls"]


def test_shape_and_finiteness_are_fail_closed():
    good_circuit = torch.ones(2, 4, 32) * .01
    good_task = torch.ones(2, 4, 4) * .01
    with pytest.raises(ValueError, match="shape"):
        rung.score_pair(good_circuit[0], good_circuit, good_task, good_task)
    bad = good_circuit.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        rung.score_pair(bad, good_circuit, good_task, good_task)


def test_planted_suite_recovers_signed_relation_and_rejects_unrelated():
    result = rung.planted_suite()
    assert result["passes"]
    assert result["positive"]["beta"] == pytest.approx(1.5)
    assert result["wrong_sign"]["beta"] == pytest.approx(-1.5)
    assert not result["wrong_sign"]["passes_without_controls"]
    assert not result["unrelated"]["passes_without_controls"]
    assert result["factorial_interaction"] == [4.0, 5.0]
