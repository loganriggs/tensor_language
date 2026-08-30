"""Pure two-source value tensor for an ordered-successor attention head.

The Bilin18 value bus at a later attention layer mixes a projection of the live
current state with the corresponding already-projected head slice of block 0's
saved ``v1`` value.
Given an already-computed attention score tensor, the complete value/output map is

    sum_k score[q,k] O ((1-lambda) V_current z_current[k]
                                + lambda V_v1 z_v1[k]).

This module implements exactly that contraction.  It performs no model calls, QK
computation, token lookup, lexicon routing, fitting, or file I/O.  A caller that
supplies native scores is therefore using a conditional value-path diagnostic, not
an autonomous zero-native-call circuit extraction.
"""

from __future__ import annotations

from enum import Enum
import math

import torch


class SuccessorArm(str, Enum):
    """Additive intervention arms around one isolated head contribution."""

    NATIVE = "native"
    REMOVE = "remove"
    EXTRACT = "extract"
    DERANGED = "deranged"


def _finite_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not torch.is_tensor(value) or value.ndim != ndim or min(value.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty rank-{ndim} tensor")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _mix_scalar(
    mix_v1: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if torch.is_tensor(mix_v1):
        if mix_v1.ndim != 0 or mix_v1.device != reference.device or (
            mix_v1.dtype != reference.dtype
        ) or not bool(torch.isfinite(mix_v1.detach())):
            raise ValueError("mix_v1 must be one finite scalar matching tensor currency")
        return mix_v1
    if type(mix_v1) is not float or not math.isfinite(mix_v1):
        raise ValueError("mix_v1 must be a finite Python float or matching scalar tensor")
    return reference.new_tensor(mix_v1)


def two_source_successor_write(
    scores: torch.Tensor,
    current_states: torch.Tensor,
    v1_states: torch.Tensor,
    current_value_factor: torch.Tensor,
    v1_value_factor: torch.Tensor,
    output_factor: torch.Tensor,
    mix_v1: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact shared-output, two-source value contraction.

    Shapes are ``scores [..., Q, K]``, states ``[..., K, D_source]``, value
    factors ``[R, D_source]``, and output factor ``[D_out, R]``.  Leading axes
    must be identical; this function never performs implicit batch/head routing.
    """

    if not torch.is_tensor(scores) or scores.ndim < 2:
        raise ValueError("scores must have shape [..., query, key]")
    _finite_tensor("scores", scores, scores.ndim)
    _finite_tensor("current_states", current_states, scores.ndim)
    _finite_tensor("v1_states", v1_states, scores.ndim)
    _finite_tensor("current_value_factor", current_value_factor, 2)
    _finite_tensor("v1_value_factor", v1_value_factor, 2)
    _finite_tensor("output_factor", output_factor, 2)
    tensors = (
        current_states, v1_states, current_value_factor, v1_value_factor, output_factor,
    )
    if any(value.device != scores.device for value in tensors):
        raise ValueError("all tensors must be on one device")
    if any(value.dtype != scores.dtype for value in tensors):
        raise ValueError("all tensors must use one dtype")
    if current_states.shape[:-2] != scores.shape[:-2] or (
        v1_states.shape[:-2] != scores.shape[:-2]
    ):
        raise ValueError("scores and states must have identical leading axes")
    key_length = scores.shape[-1]
    if current_states.shape[-2] != key_length or v1_states.shape[-2] != key_length:
        raise ValueError("both state key axes must match the score key axis")
    rank = current_value_factor.shape[0]
    if v1_value_factor.shape[0] != rank or output_factor.shape[1] != rank:
        raise ValueError("value and output factors must share one head-coordinate rank")
    if current_value_factor.shape[1] != current_states.shape[-1] or (
        v1_value_factor.shape[1] != v1_states.shape[-1]
    ):
        raise ValueError("each value factor input must match its state dimension")

    weight = _mix_scalar(mix_v1, scores)
    return two_source_preweighted_write(
        scores,
        current_states,
        v1_states,
        (1 - weight) * current_value_factor,
        weight * v1_value_factor,
        output_factor,
    )


def two_source_preweighted_write(
    scores: torch.Tensor,
    current_states: torch.Tensor,
    v1_states: torch.Tensor,
    current_right_factor: torch.Tensor,
    v1_right_factor: torch.Tensor,
    output_factor: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a two-source map whose fixed mixture is absorbed into its factors.

    This is the deployment form of a truncated SVD of the physical folded map.  It
    avoids charging a redundant scalar and makes the current-only and v1-only
    subfamilies literal by allowing one right factor to be exactly zero.
    """

    if not torch.is_tensor(scores) or scores.ndim < 2:
        raise ValueError("scores must have shape [..., query, key]")
    _finite_tensor("scores", scores, scores.ndim)
    _finite_tensor("current_states", current_states, scores.ndim)
    _finite_tensor("v1_states", v1_states, scores.ndim)
    _finite_tensor("current_right_factor", current_right_factor, 2)
    _finite_tensor("v1_right_factor", v1_right_factor, 2)
    _finite_tensor("output_factor", output_factor, 2)
    tensors = (
        current_states, v1_states, current_right_factor, v1_right_factor, output_factor,
    )
    if any(value.device != scores.device for value in tensors):
        raise ValueError("all tensors must be on one device")
    if any(value.dtype != scores.dtype for value in tensors):
        raise ValueError("all tensors must use one dtype")
    if current_states.shape[:-2] != scores.shape[:-2] or (
        v1_states.shape[:-2] != scores.shape[:-2]
    ):
        raise ValueError("scores and states must have identical leading axes")
    key_length = scores.shape[-1]
    if current_states.shape[-2] != key_length or v1_states.shape[-2] != key_length:
        raise ValueError("both state key axes must match the score key axis")
    rank = current_right_factor.shape[0]
    if v1_right_factor.shape[0] != rank or output_factor.shape[1] != rank:
        raise ValueError("right and output factors must share one candidate rank")
    if current_right_factor.shape[1] != current_states.shape[-1] or (
        v1_right_factor.shape[1] != v1_states.shape[-1]
    ):
        raise ValueError("each right factor input must match its state dimension")

    mixed_values = torch.matmul(current_states, current_right_factor.T)
    mixed_values = mixed_values + torch.matmul(v1_states, v1_right_factor.T)
    return torch.matmul(torch.matmul(scores, mixed_values), output_factor.T)


def folded_two_source_map(
    current_value_factor: torch.Tensor,
    v1_value_factor: torch.Tensor,
    output_factor: torch.Tensor,
    mix_v1: float | torch.Tensor,
) -> torch.Tensor:
    """Return the physical map ``[z_current; z_v1] -> delivered value write``."""

    _finite_tensor("current_value_factor", current_value_factor, 2)
    _finite_tensor("v1_value_factor", v1_value_factor, 2)
    _finite_tensor("output_factor", output_factor, 2)
    if current_value_factor.device != v1_value_factor.device or (
        current_value_factor.device != output_factor.device
    ):
        raise ValueError("all factors must be on one device")
    if current_value_factor.dtype != v1_value_factor.dtype or (
        current_value_factor.dtype != output_factor.dtype
    ):
        raise ValueError("all factors must use one dtype")
    rank = current_value_factor.shape[0]
    if v1_value_factor.shape[0] != rank or output_factor.shape[1] != rank:
        raise ValueError("value and output factors must share one head-coordinate rank")
    weight = _mix_scalar(mix_v1, current_value_factor)
    right = torch.cat(
        [(1 - weight) * current_value_factor, weight * v1_value_factor], dim=1,
    )
    return torch.matmul(output_factor, right)


def tolerance_rank(
    matrix: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> int:
    """Return a rank with an explicit, reproducible strict singular-value cutoff."""

    _finite_tensor("matrix", matrix, 2)
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be one finite nonnegative Python float")
    singular_values = torch.linalg.svdvals(matrix.detach().double().cpu())
    threshold = max(absolute_tolerance, relative_tolerance * float(singular_values[0]))
    return int((singular_values > threshold).sum())


def spectral_deranged_control(
    physical_map: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Materialize a same-spectrum null by mismatching left/right singular directions.

    The returned map has the same shape, Frobenius norm, singular values, and rank as
    ``physical_map``.  The permutation must be a fixed-point-free permutation of all
    ``min(output,input)`` singular coordinates.  Since repeated singular values leave
    SVD bases non-unique, experiments must materialize and hash the returned null before
    opening outcomes rather than claiming an abstract canonical null.
    """

    _finite_tensor("physical_map", physical_map, 2)
    size = min(physical_map.shape)
    if not torch.is_tensor(permutation) or permutation.dtype != torch.long or (
        permutation.ndim != 1 or permutation.shape[0] != size
    ):
        raise ValueError("permutation must be a rank-1 int64 tensor over singular coordinates")
    perm = permutation.detach().cpu()
    if not torch.equal(torch.sort(perm).values, torch.arange(size)):
        raise ValueError("permutation must be a bijection")
    if bool(torch.equal(perm, torch.arange(size))) or bool((perm == torch.arange(size)).any()):
        raise ValueError("permutation must have no fixed singular coordinates")
    original_dtype, original_device = physical_map.dtype, physical_map.device
    left, singular_values, right = torch.linalg.svd(
        physical_map.detach().double().cpu(), full_matrices=False,
    )
    null = (left * singular_values.unsqueeze(0)) @ right[perm]
    return null.to(device=original_device, dtype=original_dtype).contiguous()


def factor_complete_parameter_count(
    current_state_dim: int,
    v1_state_dim: int,
    head_rank: int,
    output_dim: int,
) -> int:
    """Serialized scalar count after absorbing the fixed mix into the two right factors."""

    dimensions = (current_state_dim, v1_state_dim, head_rank, output_dim)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("all factor dimensions must be positive Python integers")
    return head_rank * (current_state_dim + v1_state_dim + output_dim)


def autonomous_successor_parameter_count(
    current_state_dim: int,
    saved_value_dim: int,
    qk_rank: int,
    value_rank: int,
    output_dim: int,
    *,
    include_current: bool,
    include_saved: bool,
) -> int:
    """Factor-complete QK+value/output storage for one autonomous bilinear head.

    Four Q/K factors have shape ``[qk_rank, current_state_dim]``.  The live-state
    right factor is ``[value_rank, current_state_dim]``; the already-projected v1
    right factor is ``[value_rank, saved_value_dim]``; the shared output factor is
    ``[output_dim, value_rank]``. Fixed mixture scalars are absorbed.
    """

    dimensions = (current_state_dim, saved_value_dim, qk_rank, value_rank, output_dim)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("all factor dimensions must be positive Python integers")
    if type(include_current) is not bool or type(include_saved) is not bool or not (
        include_current or include_saved
    ):
        raise ValueError("at least one exact boolean value source must be included")
    right_values = (
        current_state_dim * int(include_current)
        + saved_value_dim * int(include_saved)
    )
    return 4 * qk_rank * current_state_dim + value_rank * (right_values + output_dim)


def shared_bus_producer_parameter_count(state_dim: int, head_dim: int) -> int:
    """Stored values in block 0's one-head projection that mints the shared bus."""

    if type(state_dim) is not int or state_dim <= 0 or type(head_dim) is not int or (
        head_dim <= 0
    ):
        raise ValueError("shared-bus dimensions must be positive Python integers")
    return state_dim * head_dim


def compose_successor_arm(
    residual_without_head: torch.Tensor,
    native_write: torch.Tensor,
    extracted_write: torch.Tensor,
    deranged_write: torch.Tensor,
    arm: SuccessorArm,
) -> torch.Tensor:
    """Compose one intervention arm from an explicitly head-free residual state."""

    values = (native_write, extracted_write, deranged_write)
    if not torch.is_tensor(residual_without_head) or any(
        not torch.is_tensor(value) or value.shape != residual_without_head.shape
        or value.device != residual_without_head.device
        or value.dtype != residual_without_head.dtype
        for value in values
    ):
        raise ValueError("residual and all writes must have identical tensor currency")
    if type(arm) is not SuccessorArm:
        raise ValueError("arm must be a SuccessorArm")
    if arm is SuccessorArm.NATIVE:
        return residual_without_head + native_write
    if arm is SuccessorArm.REMOVE:
        return residual_without_head.clone()
    if arm is SuccessorArm.EXTRACT:
        return residual_without_head + extracted_write
    if arm is SuccessorArm.DERANGED:
        return residual_without_head + deranged_write
    raise AssertionError("unreachable SuccessorArm")


__all__ = [
    "SuccessorArm",
    "compose_successor_arm",
    "factor_complete_parameter_count",
    "folded_two_source_map",
    "autonomous_successor_parameter_count",
    "spectral_deranged_control",
    "shared_bus_producer_parameter_count",
    "tolerance_rank",
    "two_source_successor_write",
    "two_source_preweighted_write",
]
