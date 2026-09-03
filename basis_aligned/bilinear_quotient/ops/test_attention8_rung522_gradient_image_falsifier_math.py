"""CPU tests for the conditional rung-522 gradient-image mathematics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).with_name("attention8_rung522_gradient_image_falsifier_math.py")
SPEC = importlib.util.spec_from_file_location("rung522_gradient_math", PATH)
MATH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATH
SPEC.loader.exec_module(MATH)


def test_exact_ridge_gradients_and_excitation_are_recovered():
    generator = torch.Generator().manual_seed(7)
    frame, _ = torch.linalg.qr(torch.randn(12, 4, generator=generator, dtype=torch.float64))
    gradient_coordinates = torch.randn(80, 4, generator=generator, dtype=torch.float64)
    displacement_coordinates = torch.randn(80, 4, generator=generator, dtype=torch.float64)
    gradients = gradient_coordinates @ frame.mT
    displacements = displacement_coordinates @ frame.mT
    result = MATH.summarize_gradient_image(frame, gradients, displacements)
    assert result.gradient_inside_fraction == pytest.approx(1.0, abs=1e-12)
    assert result.projected_to_full_signed_cosine == pytest.approx(1.0, abs=1e-12)
    assert result.projected_to_full_relative_residual == pytest.approx(0.0, abs=1e-12)
    assert result.projected_to_full_aligned_recovery == pytest.approx(1.0, abs=1e-12)
    assert result.excitation_numerical_rank == 4


def test_tangent_transport_can_be_exact_when_full_gradient_is_not_contained():
    frame = torch.eye(8)[:, :2]
    gradients = torch.tensor([
        [1.0, 2.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
    ])
    displacements = torch.tensor([
        [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    result = MATH.summarize_gradient_image(frame, gradients, displacements)
    assert result.gradient_inside_fraction < 0.1
    assert result.projected_to_full_relative_residual == pytest.approx(0.0)
    assert result.projected_to_full_aligned_recovery == pytest.approx(1.0)


def test_summary_is_invariant_to_the_projector_gauge():
    generator = torch.Generator().manual_seed(19)
    frame, _ = torch.linalg.qr(torch.randn(10, 3, generator=generator, dtype=torch.float64))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    gradients = torch.randn(30, 10, generator=generator, dtype=torch.float64)
    displacements = torch.randn(30, 10, generator=generator, dtype=torch.float64)
    first = MATH.summarize_gradient_image(frame, gradients, displacements)
    second = MATH.summarize_gradient_image(frame @ rotation, gradients, displacements)
    for field in (
        "gradient_inside_fraction",
        "full_tangent_rms",
        "projected_tangent_rms",
        "orthogonal_tangent_rms",
        "projected_to_full_signed_cosine",
        "projected_to_full_relative_residual",
        "projected_to_full_aligned_recovery",
    ):
        assert getattr(first, field) == pytest.approx(getattr(second, field), abs=1e-12)
    assert first.excitation_singular_values == pytest.approx(
        second.excitation_singular_values, abs=1e-12
    )


def test_missing_natural_excitation_is_reported_even_for_a_rank_four_frame():
    frame = torch.eye(9)[:, :4]
    gradients = torch.randn(20, 9, generator=torch.Generator().manual_seed(3))
    displacements = torch.zeros(20, 9)
    displacements[:, 0] = torch.linspace(-1, 1, 20)
    result = MATH.summarize_gradient_image(frame, gradients, displacements)
    assert result.rank == 4
    assert result.excitation_numerical_rank == 1
    assert result.excitation_relative_singular_values[1:] == pytest.approx((0.0, 0.0, 0.0))
