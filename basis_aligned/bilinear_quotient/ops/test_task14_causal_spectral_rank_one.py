"""CPU-only tests for the Task 14 causal-spectral initializer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
PATH = OPS / "task14_causal_spectral_rank_one.py"
SPEC = importlib.util.spec_from_file_location("task14_causal_spectral_rank_one", PATH)
SPECTRAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPECTRAL
SPEC.loader.exec_module(SPECTRAL)


def _random_inputs(count: int = 31, dimension: int = 7):
    generator = torch.Generator().manual_seed(141131)
    deltas = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    gradients = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    effects = torch.rand(count, generator=generator, dtype=torch.float64) + 0.25
    return deltas, gradients, effects


def test_planted_axis_recovery_and_full_space_local_closure() -> None:
    axis = torch.tensor([2.0, -1.0, 3.0, 0.5], dtype=torch.float64)
    axis = SPECTRAL.canonicalize_direction_sign(axis)
    amplitudes = torch.linspace(0.5, 2.5, 23, dtype=torch.float64)
    slopes = torch.linspace(1.0, 3.0, 23, dtype=torch.float64)
    deltas = amplitudes[:, None] * axis
    gradients = slopes[:, None] * axis
    effects = amplitudes * slopes

    result = SPECTRAL.causal_spectral_rank_one(deltas, gradients, effects)

    assert torch.allclose(result.direction, axis, atol=2e-15, rtol=0)
    assert torch.allclose(result.projector, torch.outer(axis, axis), atol=2e-15, rtol=0)
    assert result.diagnostics.top_eigenvalue == pytest.approx(1.0, abs=2e-15)
    assert result.diagnostics.eigengap == pytest.approx(1.0, abs=2e-15)
    assert result.diagnostics.operator_trace == pytest.approx(1.0, abs=2e-15)
    assert result.diagnostics.full_space_local_closure_mean_ratio == pytest.approx(
        1.0, abs=2e-15
    )
    assert result.diagnostics.full_space_local_closure_mean_absolute_error <= 2e-15
    assert result.diagnostics.full_space_local_closure_max_absolute_error <= 2e-15
    assert result.diagnostics.operator_trace_identity_error <= 2e-15
    assert result.diagnostics.symmetry_error == 0.0
    assert result.diagnostics.unit_norm_error <= 2e-15


def test_permuting_examples_does_not_change_operator_or_projector() -> None:
    deltas, gradients, effects = _random_inputs()
    original = SPECTRAL.causal_spectral_rank_one(deltas, gradients, effects)
    order = torch.randperm(
        len(effects), generator=torch.Generator().manual_seed(141132)
    )
    permuted = SPECTRAL.causal_spectral_rank_one(
        deltas[order], gradients[order], effects[order]
    )
    assert torch.allclose(original.operator, permuted.operator, atol=2e-16, rtol=0)
    assert torch.allclose(original.projector, permuted.projector, atol=2e-15, rtol=0)
    assert torch.allclose(original.direction, permuted.direction, atol=2e-15, rtol=0)
    assert torch.allclose(
        original.diagnostics.spectrum,
        permuted.diagnostics.spectrum,
        atol=2e-15,
        rtol=0,
    )


def test_direction_sign_and_simultaneous_input_sign_leave_projector_response_unchanged() -> None:
    deltas, gradients, effects = _random_inputs()
    result = SPECTRAL.causal_spectral_rank_one(deltas, gradients, effects)
    positive = SPECTRAL.normalized_rank_one_local_responses(
        deltas, gradients, effects, result.direction
    )
    negative = SPECTRAL.normalized_rank_one_local_responses(
        deltas, gradients, effects, -result.direction
    )
    sign_flipped_inputs = SPECTRAL.causal_spectral_rank_one(
        -deltas, -gradients, effects
    )
    assert torch.equal(positive, negative)
    assert torch.equal(
        torch.outer(result.direction, result.direction),
        torch.outer(-result.direction, -result.direction),
    )
    assert torch.equal(result.operator, sign_flipped_inputs.operator)
    assert torch.equal(result.projector, sign_flipped_inputs.projector)
    pivot = int(torch.argmax(torch.abs(result.direction)))
    assert result.direction[pivot] > 0


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d, g, e: (d[:, :-1], g, e), "identical shapes"),
        (lambda d, g, e: (d[0], g, e), "rank-2"),
        (lambda d, g, e: (d.to(torch.int64), g, e), "floating dtype"),
        (
            lambda d, g, e: (
                d.index_put(
                    (torch.tensor([0]), torch.tensor([0])),
                    d.new_tensor(float("nan")),
                ),
                g,
                e,
            ),
            "non-finite",
        ),
        (
            lambda d, g, e: (
                d,
                g.index_put(
                    (torch.tensor([0]), torch.tensor([0])),
                    g.new_tensor(float("inf")),
                ),
                e,
            ),
            "non-finite",
        ),
        (lambda d, g, e: (d, g, e[:-1]), "one value per example"),
        (
            lambda d, g, e: (
                d,
                g,
                e.index_put(
                    (torch.tensor([0]),), e.new_tensor(float("nan"))
                ),
            ),
            "non-finite",
        ),
        (
            lambda d, g, e: (
                d,
                g,
                e.index_put((torch.tensor([0]),), e.new_tensor(0.0)),
            ),
            "strictly above",
        ),
        (
            lambda d, g, e: (
                d,
                g,
                e.index_put((torch.tensor([0]),), e.new_tensor(-1.0)),
            ),
            "strictly above",
        ),
        (
            lambda d, g, e: (
                d,
                g,
                e.index_put((torch.tensor([0]),), e.new_tensor(5e-13)),
            ),
            "strictly above",
        ),
    ],
)
def test_malformed_nonfinite_and_unsafe_denominators_fail_closed(mutator, match) -> None:
    deltas, gradients, effects = _random_inputs()
    changed = mutator(deltas.clone(), gradients.clone(), effects.clone())
    with pytest.raises(SPECTRAL.CausalSpectralInputError, match=match):
        SPECTRAL.causal_spectral_rank_one(*changed)


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_denominator_floor_fails_closed(floor: float) -> None:
    deltas, gradients, effects = _random_inputs()
    with pytest.raises(SPECTRAL.CausalSpectralInputError, match="denominator_floor"):
        SPECTRAL.causal_spectral_rank_one(
            deltas, gradients, effects, denominator_floor=floor
        )


def test_deterministic_matched_rank_haar_controls() -> None:
    seeds = (141201, 141202, 141203)
    first = SPECTRAL.deterministic_rank_one_haar_frames(11, seeds)
    second = SPECTRAL.deterministic_rank_one_haar_frames(11, list(seeds))
    assert first.shape == (3, 11, 1)
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.allclose(
        torch.linalg.vector_norm(first[:, :, 0], dim=1),
        torch.ones(3, dtype=torch.float64),
        atol=2e-15,
        rtol=0,
    )
    assert not torch.equal(first[0], first[1])
    for frame in first:
        pivot = int(torch.argmax(torch.abs(frame[:, 0])))
        assert frame[pivot, 0] > 0


@pytest.mark.parametrize(
    "dimension,seeds",
    [(1, (1,)), (4, ()), (4, (1, 1)), (4, (True,)), (4, (-1,))],
)
def test_invalid_haar_control_specification_fails_closed(dimension, seeds) -> None:
    with pytest.raises(SPECTRAL.CausalSpectralInputError):
        SPECTRAL.deterministic_rank_one_haar_frames(dimension, seeds)


def test_random_initializer_is_bitwise_deterministic() -> None:
    inputs = _random_inputs(count=41, dimension=13)
    first = SPECTRAL.causal_spectral_rank_one(*inputs)
    second = SPECTRAL.causal_spectral_rank_one(*[value.clone() for value in inputs])
    assert torch.equal(first.operator, second.operator)
    assert torch.equal(first.direction, second.direction)
    assert torch.equal(first.projector, second.projector)
    assert torch.equal(first.diagnostics.spectrum, second.diagnostics.spectrum)
    assert first.diagnostics.sample_count == second.diagnostics.sample_count
    assert first.diagnostics.ambient_dimension == second.diagnostics.ambient_dimension
    assert first.diagnostics.top_eigenvalue == second.diagnostics.top_eigenvalue
    assert first.diagnostics.eigengap == second.diagnostics.eigengap
    assert (
        first.diagnostics.full_space_local_closure_mean_ratio
        == second.diagnostics.full_space_local_closure_mean_ratio
    )
    assert (
        first.diagnostics.full_space_local_closure_mean_absolute_error
        == second.diagnostics.full_space_local_closure_mean_absolute_error
    )
    assert (
        first.diagnostics.full_space_local_closure_max_absolute_error
        == second.diagnostics.full_space_local_closure_max_absolute_error
    )
