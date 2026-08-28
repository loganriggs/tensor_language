"""Zero-native-call squared-attention tensor program.

The discovery compilers replace an attention *output* after the native module has
already run.  This module instead stores every projection it executes and evaluates
the native tensor contraction directly.  It therefore provides an executable target
for typed projection compression without token tables or native fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_NAMES = ("q", "k", "q2", "k2", "v", "proj")


class StoredLinear(nn.Module):
    """A bias-free dense or factored linear map with an explicit storage price."""

    def __init__(
        self, *, weight: torch.Tensor | None = None,
        input_factor: torch.Tensor | None = None,
        output_factor: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        dense = weight is not None
        factored = input_factor is not None or output_factor is not None
        if dense == factored:
            raise ValueError("provide exactly one of dense weight or two factors")
        if dense:
            if weight.ndim != 2 or not weight.is_floating_point():
                raise ValueError("dense projection is malformed")
            self.register_buffer("weight", weight.detach().clone())
            self.register_buffer("input_factor", None)
            self.register_buffer("output_factor", None)
        else:
            if (
                input_factor is None or output_factor is None
                or input_factor.ndim != 2 or output_factor.ndim != 2
                or input_factor.shape[0] != output_factor.shape[1]
                or not input_factor.is_floating_point()
                or not output_factor.is_floating_point()
            ):
                raise ValueError("factored projection is malformed")
            self.register_buffer("weight", None)
            self.register_buffer("input_factor", input_factor.detach().clone())
            self.register_buffer("output_factor", output_factor.detach().clone())

    @classmethod
    def from_weight(cls, weight: torch.Tensor, rank: int | None = None) -> "StoredLinear":
        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        limit = min(weight.shape)
        if rank is None or rank >= limit:
            return cls(weight=weight)
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        # W = (U S) Vh. F.linear uses x W^T, so Vh is the first map and
        # U S is the second map.
        u, singular, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        return cls(
            input_factor=vh[:rank].to(weight.dtype),
            output_factor=(u[:, :rank] * singular[:rank]).to(weight.dtype),
        )

    @property
    def is_dense(self) -> bool:
        return self.weight is not None

    @property
    def stored_values(self) -> int:
        if self.weight is not None:
            return self.weight.numel()
        assert self.input_factor is not None and self.output_factor is not None
        return self.input_factor.numel() + self.output_factor.numel()

    @property
    def rank(self) -> int:
        if self.weight is not None:
            return min(self.weight.shape)
        assert self.input_factor is not None
        return self.input_factor.shape[0]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            return F.linear(value, self.weight.to(value.dtype))
        assert self.input_factor is not None and self.output_factor is not None
        hidden = F.linear(value, self.input_factor.to(value.dtype))
        return F.linear(hidden, self.output_factor.to(value.dtype))


@dataclass(frozen=True)
class AttentionCostReceipt:
    projection_values: Mapping[str, int]
    scalar_values: int
    rotary_values: int
    total_stored_values: int
    token_table_values: int
    native_calls_per_forward: int
    total_input_support: bool


class TensorPreservingSquaredAttention(nn.Module):
    """Squared bilinear attention with compressed typed projections.

    The operator retains QK head RMSNorm, RoPE, the product of two QK score
    contractions, the causal mask, the cross-layer first-value bus, c_proj, and the
    caller-owned residual addition.  No native attention object is retained.
    """

    def __init__(
        self, projections: Mapping[str, StoredLinear], *, lamb: torch.Tensor | float,
        inv_freq: torch.Tensor, n_head: int,
    ) -> None:
        super().__init__()
        if set(projections) != set(PROJECTION_NAMES):
            raise ValueError("attention projection set is incomplete")
        if type(n_head) is not int or n_head <= 0:
            raise ValueError("n_head must be positive")
        self.projections = nn.ModuleDict(dict(projections))
        q = self.projections["q"]
        width = q.weight.shape[0] if q.weight is not None else q.output_factor.shape[0]
        if width % n_head:
            raise ValueError("projection width is not divisible by head count")
        if inv_freq.ndim != 1 or inv_freq.numel() * 2 != width // n_head:
            raise ValueError("rotary frequencies do not match head width")
        self.width = int(width)
        self.n_head = n_head
        self.head_dim = self.width // n_head
        self.register_buffer("lamb", torch.as_tensor(lamb).detach().clone().reshape(()))
        self.register_buffer("inv_freq", inv_freq.detach().clone())

    @classmethod
    def from_native(
        cls, attention: nn.Module, *, ranks: Mapping[str, int | None],
    ) -> "TensorPreservingSquaredAttention":
        expected = {"q": "c_q", "k": "c_k", "q2": "c_q2", "k2": "c_k2",
                    "v": "c_v", "proj": "c_proj"}
        if set(ranks) != set(expected):
            raise ValueError("rank specification is incomplete")
        projections = {
            name: StoredLinear.from_weight(
                getattr(attention, source).weight.detach(), ranks[name],
            )
            for name, source in expected.items()
        }
        return cls(
            projections, lamb=attention.lamb.detach(),
            inv_freq=attention.rotary.inv_freq.detach(), n_head=int(attention.n_head),
        )

    def _rotary(self, value: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(value.shape[1], device=value.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(value.device))
        # Match the checkpoint implementation's cached bfloat16 trigonometric tables.
        cosine = angles.cos().bfloat16()[None, :, None, :]
        sine = angles.sin().bfloat16()[None, :, None, :]
        half = value.shape[-1] // 2
        first, second = value[..., :half], value[..., half:]
        rotated = torch.cat([
            first * cosine + second * sine,
            first * (-sine) + second * cosine,
        ], dim=-1)
        return rotated.to(value.dtype)

    def forward(
        self, state: torch.Tensor, first_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[-1] != self.width:
            raise ValueError("attention state shape changed")
        batch, sequence, _ = state.shape

        def head(name: str) -> torch.Tensor:
            return self.projections[name](state).view(
                batch, sequence, self.n_head, self.head_dim,
            )

        query, key = head("q"), head("k")
        query2, key2 = head("q2"), head("k2")
        query = self._rotary(F.rms_norm(query, (self.head_dim,)))
        key = self._rotary(F.rms_norm(key, (self.head_dim,)))
        query2 = self._rotary(F.rms_norm(query2, (self.head_dim,)))
        key2 = self._rotary(F.rms_norm(key2, (self.head_dim,)))
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / self.head_dim
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / self.head_dim
        )
        causal = torch.tril(torch.ones(
            sequence, sequence, dtype=torch.bool, device=state.device,
        ))
        pattern = pattern.masked_fill(~causal, 0.0)
        value = head("v")
        bus = value if first_value is None else first_value.view_as(value)
        mixed = (1 - self.lamb) * value + self.lamb * bus
        output = torch.einsum(
            "bhqk,bkhd->bqhd", pattern.to(mixed.dtype), mixed,
        ).reshape(batch, sequence, self.width)
        return self.projections["proj"](output), bus

    def cost_receipt(self) -> AttentionCostReceipt:
        prices = {name: layer.stored_values for name, layer in self.projections.items()}
        total = sum(prices.values()) + self.lamb.numel() + self.inv_freq.numel()
        return AttentionCostReceipt(
            projection_values=prices,
            scalar_values=self.lamb.numel(),
            rotary_values=self.inv_freq.numel(),
            total_stored_values=total,
            token_table_values=0,
            native_calls_per_forward=0,
            total_input_support=True,
        )

    def multiply_adds(self, *, batch: int, sequence: int) -> int:
        if min(batch, sequence) <= 0:
            raise ValueError("batch and sequence must be positive")
        projection = 0
        for layer in self.projections.values():
            if layer.weight is not None:
                projection += math.prod(layer.weight.shape)
            else:
                assert layer.input_factor is not None and layer.output_factor is not None
                projection += layer.input_factor.numel() + layer.output_factor.numel()
        # Two QK contractions and one pattern-value contraction.
        contractions = 3 * self.n_head * sequence * sequence * self.head_dim
        return batch * (sequence * projection + contractions)
