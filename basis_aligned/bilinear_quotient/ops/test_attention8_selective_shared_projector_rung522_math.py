"""CPU-only tests for the rung-522 mathematical core and planted toy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    path = OPS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("attention8_selective_shared_projector_rung522_math")
TOY = _load("attention8_selective_shared_projector_rung522_toy_preflight")


def test_exact_objective_uses_maximum_target_not_average():
    one = CORE.TargetResponse(
        full_member=torch.ones(4, dtype=torch.float64),
        projected_member=torch.zeros(4, dtype=torch.float64),
        projected_control=torch.zeros(4, dtype=torch.float64),
    )
    two = CORE.TargetResponse(
        full_member=torch.ones(4, dtype=torch.float64),
        projected_member=torch.full((4,), 0.5, dtype=torch.float64),
        projected_control=torch.zeros(4, dtype=torch.float64),
    )
    result = CORE.exact_max_target_objective(
        {"lower": two, "higher": one}, control_coefficient=8.0, epsilon=1e-12
    )
    assert result.maximizing_target == "higher"
    assert float(result.maximum) == pytest.approx(1.0, abs=2e-12)
    assert float(result.maximum) != pytest.approx(
        sum(float(loss.total) for loss in result.per_target.values()) / 2
    )


def test_loss_normalization_and_control_coefficient_are_literal():
    response = CORE.TargetResponse(
        full_member=torch.tensor([1.0, -1.0], dtype=torch.float64),
        projected_member=torch.tensor([0.5, -0.5], dtype=torch.float64),
        projected_control=torch.tensor([0.25, -0.25], dtype=torch.float64),
    )
    loss = CORE.target_loss(response, control_coefficient=8.0, epsilon=1e-30)
    assert float(loss.member) == pytest.approx(0.25)
    assert float(loss.control) == pytest.approx(0.0625)
    assert float(loss.total) == pytest.approx(0.75)


def test_response_metrics_have_expected_scale_and_concentration():
    full = torch.tensor([2.0, -4.0, 6.0], dtype=torch.float64)
    projected = full / 2
    metrics = CORE.signed_response_metrics(projected, full)
    assert metrics["signed_cosine"] == pytest.approx(1.0)
    assert metrics["optimal_scale_projected_to_full"] == pytest.approx(2.0)
    assert metrics["relative_residual"] == pytest.approx(0.0)
    assert metrics["aligned_recovery"] == pytest.approx(0.5)
    concentration = CORE.response_concentration(projected, projected / 4)
    assert concentration["control_to_member_ratio"] == pytest.approx(0.25)
    assert concentration["member_to_control_concentration"] == pytest.approx(4.0)


def test_bilinear_response_and_projector_are_gauge_invariant():
    generator = torch.Generator().manual_seed(522)
    x = torch.randn(20, 12, generator=generator, dtype=torch.float64)
    reader = torch.randn(20, 12, generator=generator, dtype=torch.float64)
    frame = CORE.deterministic_haar_frame(12, 4, 52200)
    rotation = CORE.deterministic_haar_frame(4, 4, 52201)
    first = CORE.projected_bilinear_response(x, reader, frame)
    second = CORE.projected_bilinear_response(x, reader, frame @ rotation)
    assert torch.allclose(first, second, atol=2e-12, rtol=0)


def test_seeded_initialization_has_a_float32_science_path_and_recovery_only_is_legal():
    frame = CORE.deterministic_haar_frame(12, 4, 52200, dtype=torch.float32, device="cpu")
    assert frame.dtype == torch.float32
    assert frame.device.type == "cpu"
    assert float(CORE.daslib.orthonormality_error(frame)) <= 1e-5
    CORE.OptimizerConfig(control_coefficient=0.0).validate(1152)


def test_planted_problem_equalizes_member_fit_but_control_rejects_broad():
    problem = TOY.make_problem(samples=64)
    selective = TOY._objective(problem, problem.selective, "validation")
    broad = TOY._objective(problem, problem.broad, "validation")
    assert max(float(x.member) for x in selective.per_target.values()) == pytest.approx(
        max(float(x.member) for x in broad.per_target.values()), abs=1e-12
    )
    assert float(selective.maximum) < float(broad.maximum)


def test_frozen_optimizer_recovers_planted_selective_projector_one_seed():
    problem = TOY.make_problem(samples=96)
    fit = CORE.fit_projector(
        32,
        52200,
        lambda frame, _step: TOY._responses(problem.train, frame),
        lambda frame, _step: TOY._responses(problem.validation, frame),
        dtype=torch.float64,
    )
    metrics = TOY._heldout_metrics(problem, fit.frame)
    assert fit.healthy, fit.health_failures
    assert fit.final_orthonormality_error <= 1e-5
    assert metrics["selective_projector_overlap"] >= 0.90
    assert metrics["broad_projector_overlap"] <= 0.10
    assert metrics["signed_cosine"] >= 0.99
    assert metrics["member_to_control_concentration"] >= 4.0


def test_model_gradient_audit_fails_closed():
    okay = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    CORE.assert_parameters_have_no_gradients([okay])
    offender = torch.nn.Parameter(torch.ones(2))
    offender.grad = torch.ones_like(offender)
    with pytest.raises(RuntimeError, match="acquired gradients"):
        CORE.assert_parameters_have_no_gradients([offender])
