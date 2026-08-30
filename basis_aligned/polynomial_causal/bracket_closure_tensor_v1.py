"""Owned fixed-tensor arms for the L13H8 bracket-closure canary.

Every executable arm stores and evaluates all nine heads of layer-13 squared
attention through :class:`TensorPreservingSquaredAttention`.  The fixed projector is
constant on the head leg.  No delimiter label, parser state, target mask, or selected
key is available to execution.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Mapping

import torch

from tensor_preserving_attention import (
    PROJECTION_NAMES,
    StoredLinear,
    TensorPreservingSquaredAttention,
)


TARGET_SITE = 13
TARGET_HEAD = 8
PRODUCTION_WIDTH = 1152
PRODUCTION_HEADS = 9
PRODUCTION_HEAD_DIM = 128
PRODUCTION_ROTARY_VALUES = 64


class BracketTensorArm(str, Enum):
    STORED_ALL_HEADS = "stored_l13_all_heads"
    DELETE_H8 = "stored_l13_delete_h8"
    DERANGED_H8 = "stored_l13_deranged_h8"


def exact_stored_attention_price(
    *, width: int, heads: int, rotary_values: int,
) -> int:
    """Six dense projections plus lambda, rotary constants, and head projector."""

    if any(type(value) is not int or value <= 0 for value in (width, heads, rotary_values)):
        raise ValueError("attention price dimensions must be positive Python integers")
    if width % heads:
        raise ValueError("attention width must be divisible by head count")
    return 6 * width * width + 1 + rotary_values + heads


PRODUCTION_STORED_VALUES = exact_stored_attention_price(
    width=PRODUCTION_WIDTH,
    heads=PRODUCTION_HEADS,
    rotary_values=PRODUCTION_ROTARY_VALUES,
)


def cyclic_derangement(size: int) -> torch.Tensor:
    if type(size) is not int or size <= 1:
        raise ValueError("derangement size must exceed one")
    return torch.cat([torch.arange(1, size), torch.zeros(1, dtype=torch.long)])


def spectral_derange_output_head(
    output_weight: torch.Tensor,
    *,
    head: int,
    head_dim: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Derange one output-column slice while preserving that slice's spectrum.

    SVD is performed in CPU float64.  The returned dense matrix must be materialized
    and hashed before outcomes; repeated singular values make the abstract SVD basis
    noncanonical even though the resulting stored matrix is exact once frozen.
    """

    if not torch.is_tensor(output_weight) or output_weight.ndim != 2 or (
        not output_weight.dtype.is_floating_point
    ) or not bool(torch.isfinite(output_weight.detach()).all()):
        raise ValueError("output projection must be one finite floating matrix")
    if output_weight.shape[0] != output_weight.shape[1]:
        raise ValueError("output projection must be square")
    width = output_weight.shape[1]
    if type(head_dim) is not int or head_dim <= 1 or width % head_dim:
        raise ValueError("head dimension does not divide output width")
    heads = width // head_dim
    if type(head) is not int or not 0 <= head < heads:
        raise ValueError("head index is outside output projection")
    if not torch.is_tensor(permutation) or permutation.dtype != torch.long or (
        permutation.ndim != 1 or permutation.shape[0] != head_dim
    ):
        raise ValueError("derangement must be one int64 head-coordinate permutation")
    perm = permutation.detach().cpu()
    identity = torch.arange(head_dim)
    if not torch.equal(torch.sort(perm).values, identity) or bool((perm == identity).any()):
        raise ValueError("derangement must be a fixed-point-free bijection")

    start = head * head_dim
    source = output_weight[:, start:start + head_dim].detach().double().cpu()
    left, singular_values, right = torch.linalg.svd(source, full_matrices=False)
    replacement = (left * singular_values.unsqueeze(0)) @ right[perm]
    result = output_weight.detach().clone()
    result[:, start:start + head_dim] = replacement.to(
        device=result.device, dtype=result.dtype,
    )
    return result.contiguous()


def _native_projection_weights(attention: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    names = {
        "q": "c_q", "k": "c_k", "q2": "c_q2", "k2": "c_k2",
        "v": "c_v", "proj": "c_proj",
    }
    try:
        weights = {name: getattr(attention, source).weight.detach() for name, source in names.items()}
    except AttributeError as error:
        raise ValueError("native attention projection schema changed") from error
    if set(weights) != set(PROJECTION_NAMES):
        raise AssertionError("projection name closure changed")
    return weights


def build_bracket_tensor_program(
    attention: torch.nn.Module,
    arm: BracketTensorArm,
    *,
    permutation: torch.Tensor | None = None,
) -> TensorPreservingSquaredAttention:
    """Copy one native attention into an owned, dense, zero-native-call arm."""

    if type(arm) is not BracketTensorArm:
        raise ValueError("arm must be a BracketTensorArm")
    try:
        width = int(attention.n_embd)
        heads = int(attention.n_head)
        inv_freq = attention.rotary.inv_freq.detach()
        lamb = attention.lamb.detach()
    except AttributeError as error:
        raise ValueError("native attention topology changed") from error
    if width <= 0 or heads <= 0 or width % heads:
        raise ValueError("native attention topology is malformed")
    head_dim = width // heads
    if TARGET_HEAD >= heads:
        raise ValueError("target head is outside native attention")
    weights = dict(_native_projection_weights(attention))
    head_weights = torch.ones(heads, device=weights["proj"].device, dtype=weights["proj"].dtype)
    if arm is BracketTensorArm.DELETE_H8:
        if permutation is not None:
            raise ValueError("deletion arm cannot carry a derangement")
        head_weights[TARGET_HEAD] = 0
    elif arm is BracketTensorArm.DERANGED_H8:
        if permutation is None:
            raise ValueError("deranged arm requires its frozen permutation")
        weights["proj"] = spectral_derange_output_head(
            weights["proj"], head=TARGET_HEAD, head_dim=head_dim,
            permutation=permutation,
        )
    elif permutation is not None:
        raise ValueError("stored replay arm cannot carry a derangement")

    program = TensorPreservingSquaredAttention(
        {name: StoredLinear(weight=weights[name]) for name in PROJECTION_NAMES},
        lamb=lamb,
        inv_freq=inv_freq,
        n_head=heads,
        head_weights=head_weights,
    )
    receipt = program.cost_receipt()
    expected = exact_stored_attention_price(
        width=width, heads=heads, rotary_values=inv_freq.numel(),
    )
    if receipt.total_stored_values != expected or receipt.native_calls_per_forward != 0 or (
        receipt.token_table_values != 0 or not receipt.total_input_support
    ):
        raise RuntimeError("bracket tensor program price/support receipt changed")
    return program


def program_state_sha256(program: TensorPreservingSquaredAttention) -> str:
    """Hash exact ordered tensor bytes, dtypes, shapes, and state names."""

    if not isinstance(program, TensorPreservingSquaredAttention):
        raise ValueError("program state must be a TensorPreservingSquaredAttention")
    digest = hashlib.sha256()
    for name, value in sorted(program.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


__all__ = (
    "BracketTensorArm",
    "PRODUCTION_STORED_VALUES",
    "TARGET_HEAD",
    "TARGET_SITE",
    "build_bracket_tensor_program",
    "cyclic_derangement",
    "exact_stored_attention_price",
    "program_state_sha256",
    "spectral_derange_output_head",
)
