from __future__ import annotations

import math
import torch

from causal_response_residual_unfolding_certificate import tensor_certificate


def _cp_tensor(shape: tuple[int, ...], rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    factors = [torch.randn((dimension, rank), generator=generator, dtype=torch.float64) for dimension in shape]
    return torch.einsum("ir,jr,kr,lr->ijkl", *factors)


def test_rank_two_toy_has_zero_tail_after_rank_two() -> None:
    receipt = tensor_certificate(_cp_tensor((3, 6, 7, 9), 2, 1))
    assert receipt["cp_rank_lower_bound_95"] <= 2
    assert receipt["cp_approximation_error_lower_bound_tail_fraction"]["2"] < 1e-28


def test_higher_rank_toy_falsifies_rank_two_correction() -> None:
    receipt = tensor_certificate(_cp_tensor((6, 7, 8, 9), 5, 2))
    assert receipt["cp_rank_lower_bound_95"] > 2
    assert receipt["cp_approximation_error_lower_bound_tail_fraction"]["2"] > 1e-3


def test_certificate_is_invariant_to_axis_permutation_as_a_lower_bound() -> None:
    value = _cp_tensor((4, 5, 6, 7), 4, 3)
    first = tensor_certificate(value)
    second = tensor_certificate(value.permute(2, 0, 3, 1).contiguous())
    for rank, tail in first["cp_approximation_error_lower_bound_tail_fraction"].items():
        assert math.isclose(
            tail,
            second["cp_approximation_error_lower_bound_tail_fraction"][rank],
            rel_tol=1e-13,
            abs_tol=1e-28,
        )
