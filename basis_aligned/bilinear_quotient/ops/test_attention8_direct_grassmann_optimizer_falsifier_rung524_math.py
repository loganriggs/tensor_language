"""CPU tests for rung 524's pure geometry and fail-closed scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATH = _load("attention8_direct_grassmann_optimizer_falsifier_rung524_math")


def _frame(seed: int, d: int = MATH.D, rank: int = MATH.RANK):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return MATH.canonical_qr(torch.randn(d, rank, generator=generator, dtype=torch.float64))


def _passing_fit():
    return MATH.PretestFit(
        accepted_updates=MATH.UPDATES,
        finite_losses=True,
        finite_gradients=True,
        maximum_evaluated_loss=50.0,
        orthonormality_error=1e-12,
        initial_fit_loss=20.0,
        final_fit_loss=0.5,
        initial_validation_loss=20.0,
        final_validation_loss=0.5,
        projector_error=0.05,
        minimum_principal_cosine=0.999,
    )


def test_tangent_is_horizontal_and_retraction_is_orthonormal():
    q = _frame(1)
    generator = torch.Generator(device="cpu").manual_seed(2)
    gradient = torch.randn(q.shape, generator=generator, dtype=q.dtype)
    tangent = MATH.grassmann_tangent(q, gradient)
    assert torch.allclose(q.mT @ tangent, torch.zeros(MATH.RANK, MATH.RANK, dtype=q.dtype), atol=1e-12)
    moved = MATH.retract(q, -0.1 * tangent)
    assert torch.allclose(moved.mT @ moved, torch.eye(MATH.RANK, dtype=q.dtype), atol=1e-12)


def test_projector_scores_are_basis_invariant():
    q = _frame(3)
    rotation = _frame(4, d=MATH.RANK, rank=MATH.RANK)
    rotated = q @ rotation
    assert MATH.projector_error(q, rotated) < 1e-12
    assert MATH.minimum_principal_cosine(q, rotated) == pytest.approx(1.0, abs=1e-12)


def test_pass_opens_ood_and_licenses_only_after_ood_pass():
    pretest = MATH.score_pretest([_passing_fit() for _ in range(MATH.FIT_COUNT)])
    assert pretest["pretest_passes"]
    ood = MATH.score_ood([0.01] * MATH.FIT_COUNT)
    assert MATH.final_decision(pretest, ood)["licenses_one_model_calibration"]


def test_failed_pretest_rejects_ood_and_pivots():
    fits = [_passing_fit() for _ in range(MATH.FIT_COUNT)]
    fits[0] = MATH.PretestFit(**{**fits[0].__dict__, "projector_error": 0.2})
    pretest = MATH.score_pretest(fits)
    assert not pretest["pretest_passes"]
    decision = MATH.final_decision(pretest, None)
    assert decision["next_action"] == "pivot_to_mlp0_exact_branch_decomposition"
    with pytest.raises(ValueError, match="OOD result exists"):
        MATH.final_decision(pretest, MATH.score_ood([0.01] * MATH.FIT_COUNT))
