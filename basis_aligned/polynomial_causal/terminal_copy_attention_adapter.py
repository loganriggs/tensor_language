"""Owned per-head decomposition of bilin18's native tensor attention.

The adapter copies the six native projection matrices and scalar/rotary state.  It
does not retain the native attention module and makes no native projection calls
after construction.  A transaction exposes only cloned sums of selected head
writes; the underlying per-head tensor is revoked on close.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_NAMES = ("q", "k", "q2", "k2", "v", "proj")


@dataclass(frozen=True)
class HeadWriteClosure:
    selected_head_sets: tuple[tuple[int, ...], ...]
    all_head_recomposition_max_abs_error: float
    all_head_recomposition_relative_error: float
    closed: bool


class OwnedPerHeadTensorAttention(nn.Module):
    """Exact, source-owned per-head attention-write decomposition."""

    def __init__(
        self, *, weights: dict[str, torch.Tensor], lamb: torch.Tensor,
        inv_freq: torch.Tensor, n_head: int,
    ) -> None:
        super().__init__()
        if set(weights) != set(PROJECTION_NAMES) or type(n_head) is not int or n_head <= 0:
            raise ValueError("per-head attention source is incomplete")
        shapes = {tuple(value.shape) for value in weights.values()}
        if len(shapes) != 1:
            raise ValueError("attention projections do not share one square shape")
        (shape,) = shapes
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] % n_head:
            raise ValueError("attention projection topology is malformed")
        if any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in weights.values()
        ):
            raise ValueError("attention projections are nonfinite")
        scalar = torch.as_tensor(lamb)
        width, head_dim = shape[0], shape[0] // n_head
        if scalar.numel() != 1 or not scalar.is_floating_point() or not bool(
            torch.isfinite(scalar).all()
        ) or inv_freq.shape != (head_dim // 2,) or not inv_freq.is_floating_point() or (
            not bool(torch.isfinite(inv_freq).all())
        ):
            raise ValueError("attention scalar or rotary state is malformed")
        for name, value in weights.items():
            self.register_buffer(name, value.detach().clone())
        self.register_buffer("lamb", scalar.detach().clone().reshape(()))
        self.register_buffer("inv_freq", inv_freq.detach().clone())
        self.width = width
        self.n_head = n_head
        self.head_dim = head_dim

    @classmethod
    def from_native(cls, attention: nn.Module) -> "OwnedPerHeadTensorAttention":
        names = {
            "q": "c_q", "k": "c_k", "q2": "c_q2", "k2": "c_k2",
            "v": "c_v", "proj": "c_proj",
        }
        try:
            weights = {
                name: getattr(attention, source).weight.detach()
                for name, source in names.items()
            }
            biases = [getattr(attention, source).bias for source in names.values()]
            n_head = int(attention.n_head)
            lamb = attention.lamb.detach()
            inv_freq = attention.rotary.inv_freq.detach()
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("native attention schema changed") from error
        if any(value is not None for value in biases):
            raise ValueError("bilin18 attention projections must be bias-free")
        program = cls(weights=weights, lamb=lamb, inv_freq=inv_freq, n_head=n_head)
        try:
            parameter = next(attention.parameters())
        except StopIteration as error:
            raise ValueError("native attention has no device-bearing parameter") from error
        # The checkpoint's Rotary.inv_freq is a plain float32 attribute rather than
        # a registered buffer, so model.to(bfloat16) does not cast it.  Preserve each
        # copied tensor's source dtype and move only the device; casting the owned
        # inv_freq to the projection dtype measurably changes production RoPE.
        return program.to(device=parameter.device)

    def _project(self, name: str, state: torch.Tensor) -> torch.Tensor:
        return F.linear(state, getattr(self, name).to(state.dtype)).view(
            state.shape[0], state.shape[1], self.n_head, self.head_dim,
        )

    def _rotate(self, value: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            value.shape[1], device=value.device, dtype=self.inv_freq.dtype,
        )
        angles = torch.outer(positions, self.inv_freq.to(value.device))
        cosine = angles.cos().bfloat16()[None, :, None, :]
        sine = angles.sin().bfloat16()[None, :, None, :]
        value = F.rms_norm(value, (self.head_dim,))
        first, second = value[..., : self.head_dim // 2], value[..., self.head_dim // 2 :]
        return torch.cat((
            first * cosine + second * sine,
            first * (-sine) + second * cosine,
        ), dim=-1).to(value.dtype)

    def _decompose(
        self, state: torch.Tensor, first_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[-1] != self.width or not state.is_floating_point() or (
            not bool(torch.isfinite(state).all())
        ):
            raise ValueError("attention state is malformed")
        query, key = self._rotate(self._project("q", state)), self._rotate(
            self._project("k", state)
        )
        query2, key2 = self._rotate(self._project("q2", state)), self._rotate(
            self._project("k2", state)
        )
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / self.head_dim
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / self.head_dim
        )
        causal = torch.tril(torch.ones(
            state.shape[1], state.shape[1], device=state.device, dtype=torch.bool,
        ))
        pattern = pattern.masked_fill(~causal, 0.0)
        value = self._project("v", state)
        if first_value is None:
            bus = value
        else:
            if first_value.shape != value.shape or first_value.device != value.device or (
                first_value.dtype != value.dtype or not bool(torch.isfinite(first_value).all())
            ):
                raise ValueError("first-value bus is malformed")
            bus = first_value
        mixed = (1 - self.lamb) * value + self.lamb * bus
        # Preserve the checkpoint's physical contraction and layout order: its
        # native module first forms [batch,head,query,d_head], then transposes and
        # materializes [batch,query,head,d_head] before c_proj.  Asking einsum for
        # the latter layout directly is algebraically equal but measurably changes
        # bfloat16 accumulation on the production CUDA kernel.
        head_outputs = torch.einsum(
            "bhqk,bkhd->bhqd", pattern.to(mixed.dtype), mixed,
        ).transpose(1, 2).contiguous()
        projection = self.proj.to(head_outputs.dtype).view(
            self.width, self.n_head, self.head_dim,
        )
        # [b,t,h,d_head] contracted with the matching input columns of c_proj.
        head_writes = torch.einsum("bthd,ohd->btho", head_outputs, projection)
        full_write = F.linear(
            head_outputs.reshape(state.shape[0], state.shape[1], self.width),
            self.proj.to(head_outputs.dtype),
        )
        return head_writes, full_write, bus

    def begin(
        self, state: torch.Tensor, first_value: torch.Tensor | None = None,
    ) -> "HeadWriteTransaction":
        return HeadWriteTransaction(self, state, first_value)

    def price(self) -> dict[str, int | bool]:
        projection_values = sum(getattr(self, name).numel() for name in PROJECTION_NAMES)
        return {
            "projection_values": projection_values,
            "scalar_values": self.lamb.numel(),
            "rotary_values": self.inv_freq.numel(),
            "total_stored_values": projection_values + self.lamb.numel() + self.inv_freq.numel(),
            "native_calls_per_forward": 0,
            "token_table_values": 0,
            "total_input_support": True,
        }


class HeadWriteTransaction(AbstractContextManager):
    """Revocable access to cloned sums of selected per-head writes."""

    def __init__(
        self, program: OwnedPerHeadTensorAttention, state: torch.Tensor,
        first_value: torch.Tensor | None,
    ) -> None:
        head_writes, full_write, bus = program._decompose(state, first_value)
        reconstructed = head_writes.sum(2)
        difference = (reconstructed.float() - full_write.float()).abs()
        denominator = max(float(full_write.float().norm()), torch.finfo(torch.float32).tiny)
        self._head_writes: torch.Tensor | None = head_writes
        self._full_write: torch.Tensor | None = full_write
        self._bus: torch.Tensor | None = bus
        self._max_error = float(difference.max())
        self._relative_error = float(difference.norm()) / denominator
        self._selected: list[tuple[int, ...]] = []
        self._n_head = program.n_head
        self._closed = False
        self._closure: HeadWriteClosure | None = None

    def __enter__(self) -> "HeadWriteTransaction":
        if self._closed:
            raise RuntimeError("head-write transaction is closed")
        return self

    def _require_open(self) -> torch.Tensor:
        if self._closed or self._head_writes is None:
            raise RuntimeError("head-write transaction is closed")
        return self._head_writes

    def select(self, heads: Iterable[int]) -> torch.Tensor:
        value = tuple(heads)
        if not value or len(set(value)) != len(value) or any(
            type(head) is not int or not 0 <= head < self._n_head for head in value
        ):
            raise ValueError("selected attention heads are malformed")
        writes = self._require_open()
        self._selected.append(value)
        return writes[:, :, value, :].sum(2).clone()

    def all_heads(self) -> torch.Tensor:
        return self.select(range(self._n_head))

    def native_full_write(self) -> torch.Tensor:
        self._require_open()
        assert self._full_write is not None
        return self._full_write.clone()

    def first_value_bus(self) -> torch.Tensor:
        self._require_open()
        assert self._bus is not None
        return self._bus.clone()

    @property
    def closure(self) -> HeadWriteClosure:
        if not self._closed or self._closure is None:
            raise RuntimeError("head-write transaction is not closed")
        return self._closure

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._closed:
            self._closure = HeadWriteClosure(
                selected_head_sets=tuple(self._selected),
                all_head_recomposition_max_abs_error=self._max_error,
                all_head_recomposition_relative_error=self._relative_error,
                closed=True,
            )
            self._head_writes = None
            self._full_write = None
            self._bus = None
            self._closed = True
        return False
