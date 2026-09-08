"""Exact checkpoint contractions for causally admitted head and MLP pieces."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch


class CausalCheckpointTranslationError(ValueError):
    pass


def _finite_tensor(name, tensor, *, minimum_rank=1):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < minimum_rank:
        raise CausalCheckpointTranslationError(
            f"{name} must be a tensor with rank at least {minimum_rank}")
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
