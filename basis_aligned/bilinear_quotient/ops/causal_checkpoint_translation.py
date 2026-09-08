"""Exact checkpoint contractions for causally admitted head and MLP pieces."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch

import normalized_weight_reader_contract as normalized_reader


class CausalCheckpointTranslationError(ValueError):
    pass


def _finite_tensor(name, tensor, *, minimum_rank=1):
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim < minimum_rank
            or not tensor.is_floating_point()):
        raise CausalCheckpointTranslationError(
            f"{name} must be a floating tensor with rank at least {minimum_rank}")
    if not bool(torch.isfinite(tensor).all()):
        raise CausalCheckpointTranslationError(f"{name} must be finite")


def _linear_write(delta, weight, *, name):
    """Apply a PyTorch-Linear checkpoint weight `[out, in]` to a delta."""
    _finite_tensor(name, delta)
    _finite_tensor("weight", weight, minimum_rank=2)
    if weight.ndim != 2 or delta.shape[-1] != weight.shape[1]:
        raise CausalCheckpointTranslationError(
            f"{name} trailing width must equal weight input width")
    return torch.matmul(delta.float(), weight.float().transpose(0, 1))


def attention_head_write(head_delta, c_proj_weight, *, head, num_heads):
    """Translate one pre-`c_proj` head delta into its physical residual write.

    `c_proj_weight` follows `torch.nn.Linear` orientation `[residual, heads*width]`.
    Leading batch/token dimensions on `head_delta` are preserved.
    """
    _finite_tensor("head_delta", head_delta)
    _finite_tensor("c_proj_weight", c_proj_weight, minimum_rank=2)
    if c_proj_weight.ndim != 2 or not isinstance(num_heads, int) or num_heads < 1:
        raise CausalCheckpointTranslationError(
            "c_proj_weight must be rank two and num_heads must be positive")
    if not isinstance(head, int) or head < 0 or head >= num_heads:
        raise CausalCheckpointTranslationError("head index is out of range")
    input_width = int(c_proj_weight.shape[1])
    if input_width % num_heads:
        raise CausalCheckpointTranslationError(
            "c_proj input width is not divisible by num_heads")
    head_width = input_width // num_heads
    if head_delta.shape[-1] != head_width:
        raise CausalCheckpointTranslationError(
            "head delta width does not match the selected c_proj slice")
    start = head * head_width
    weight_slice = c_proj_weight[:, start:start + head_width]
    return _linear_write(head_delta, weight_slice, name="head_delta")


def bilinear_product_factors(live_left, live_right, writer_left, writer_right):
    """Return the exact arm-local Left, Right, and interaction product deltas."""
    tensors = (live_left, live_right, writer_left, writer_right)
    for name, tensor in zip(
        ("live_left", "live_right", "writer_left", "writer_right"), tensors
    ):
        _finite_tensor(name, tensor)
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise CausalCheckpointTranslationError(
            "live and writer Left/Right tensors must share one shape")
    left0, right0, left1, right1 = (tensor.float() for tensor in tensors)
    delta_left = left1 - left0
    delta_right = right1 - right0
    return {
        "left": delta_left * right0,
        "right": left0 * delta_right,
        "interaction": delta_left * delta_right,
    }


def mlp_factor_write(factor, down_weight):
    """Translate one product-factor delta through `Down.weight [residual, hidden]`."""
    return _linear_write(factor, down_weight, name="factor")


def mlp_factor_writes(live_left, live_right, writer_left, writer_right, down_weight):
    """Return all three exact residual-write tensors and their exact sum."""
    factors = bilinear_product_factors(
        live_left, live_right, writer_left, writer_right
    )
    writes = {name: mlp_factor_write(value, down_weight)
              for name, value in factors.items()}
    writes["all_three"] = sum(writes.values())
    return writes


def normalized_reader_output(residual_state, residual_write, reader_weight, *, eps=None):
    """Return the exact finite RMSNorm-aware change seen by a checkpoint reader.

    RMSNorm is evaluated in the state tensor's deployed dtype. The resulting finite
    input secant and checkpoint reader weight are contracted in float32. This is a
    weight-based reader nomination; causal reader-side interchange remains a separate
    identification test.
    """
    _finite_tensor("residual_state", residual_state, minimum_rank=2)
    _finite_tensor("residual_write", residual_write, minimum_rank=2)
    _finite_tensor("reader_weight", reader_weight, minimum_rank=2)
    if residual_state.shape != residual_write.shape:
        raise CausalCheckpointTranslationError(
            "residual state and write must share one shape")
    if reader_weight.ndim != 2 or reader_weight.shape[1] != residual_state.shape[-1]:
        raise CausalCheckpointTranslationError(
            "reader weight must have shape [output, residual_width]")
    if eps is None:
        eps = torch.finfo(residual_state.dtype).eps
    try:
        epsilon = float(eps)
    except (TypeError, ValueError):
        raise CausalCheckpointTranslationError("RMSNorm epsilon must be finite")
    if not torch.isfinite(torch.tensor(epsilon)) or epsilon < 0:
        raise CausalCheckpointTranslationError(
            "RMSNorm epsilon must be finite and nonnegative")
    normalized0 = torch.nn.functional.rms_norm(
        residual_state, (residual_state.shape[-1],), eps=epsilon
    )
    normalized1 = torch.nn.functional.rms_norm(
        residual_state + residual_write,
        (residual_state.shape[-1],), eps=epsilon,
    )
    return _linear_write(
        normalized1.float() - normalized0.float(), reader_weight,
        name="normalized_input_delta",
    )


def normalized_reader_report(residual_state, residual_write, reader_weight, *, eps=None):
    """Reuse the canonical raw/tangent/exact reader diagnostic on any leading shape."""
    _finite_tensor("residual_state", residual_state, minimum_rank=2)
    _finite_tensor("residual_write", residual_write, minimum_rank=2)
    _finite_tensor("reader_weight", reader_weight, minimum_rank=2)
    if residual_state.shape != residual_write.shape:
        raise CausalCheckpointTranslationError(
            "residual state and write must share one shape")
    if reader_weight.ndim != 2 or reader_weight.shape[1] != residual_state.shape[-1]:
        raise CausalCheckpointTranslationError(
            "reader weight must have shape [output, residual_width]")
    epsilon = torch.finfo(residual_state.dtype).eps if eps is None else eps
    flat_state = residual_state.reshape(-1, residual_state.shape[-1]).float()
    flat_write = residual_write.reshape(-1, residual_write.shape[-1]).float()
    try:
        return normalized_reader.reader_response(
            torch, reader_weight.float(), flat_state, flat_write, eps=epsilon
        )
    except (TypeError, ValueError, RuntimeError):
        raise CausalCheckpointTranslationError(
            "normalized reader diagnostic failed")


def reader_response_match(candidate, reference):
    """Measure whether two task writes are operationally equivalent to one reader."""
    _finite_tensor("candidate", candidate)
    _finite_tensor("reference", reference)
    if candidate.shape != reference.shape:
        raise CausalCheckpointTranslationError(
            "candidate and reference reader responses must share one shape")
    candidate, reference = candidate.float(), reference.float()
    reference_norm = reference.norm().clamp_min(1e-30)
    candidate_norm = candidate.norm()
    cosine = float(
        (candidate.reshape(-1) @ reference.reshape(-1))
        / (candidate_norm * reference_norm).clamp_min(1e-30)
    )
    return {
        "cosine": cosine,
        "relative_l2": float((candidate - reference).norm() / reference_norm),
        "norm_ratio": float(candidate_norm / reference_norm),
        "sign_agreement": float(
            ((candidate == 0) & (reference == 0)
             | (candidate.sign() == reference.sign())).float().mean()
        ),
    }
