"""Zero-native-call squared-attention tensor program.

The discovery compilers replace an attention *output* after the native module has
already run.  This module instead stores every projection it executes and evaluates
the native tensor contraction directly.  It therefore provides an executable target
for typed projection compression without token tables or native fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_NAMES = ("q", "k", "q2", "k2", "v", "proj")
QK_NAMES = ("q", "k", "q2", "k2")


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

    @property
    def input_dim(self) -> int:
        if self.weight is not None:
            return self.weight.shape[1]
        assert self.input_factor is not None
        return self.input_factor.shape[1]

    @property
    def output_dim(self) -> int:
        if self.weight is not None:
            return self.weight.shape[0]
        assert self.output_factor is not None
        return self.output_factor.shape[0]

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


@dataclass(frozen=True)
class AttentionBankClosure:
    sites: tuple[tuple[int, int], ...]
    ordered: bool
    block_identity: bool
    first_value_identity: bool
    closed: bool


class SharedInputLinearBank(nn.Module):
    """One encoded input dictionary with multiple typed linear decoders."""

    def __init__(
        self, input_factor: torch.Tensor,
        output_factors: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        if set(output_factors) != set(QK_NAMES) or input_factor.ndim != 2 or (
            not input_factor.is_floating_point()
        ):
            raise ValueError("shared QK bank is malformed")
        rank, width = input_factor.shape
        if rank <= 0 or any(
            factor.ndim != 2 or factor.shape != (width, rank)
            or not factor.is_floating_point()
            or not bool(torch.isfinite(factor).all())
            for factor in output_factors.values()
        ) or not bool(torch.isfinite(input_factor).all()):
            raise ValueError("shared QK factors are malformed")
        self.register_buffer("input_factor", input_factor.detach().clone())
        for name in QK_NAMES:
            self.register_buffer(f"output_{name}", output_factors[name].detach().clone())
        self.width = width
        self.rank = rank

    @classmethod
    def from_basis(
        cls, weights: Mapping[str, torch.Tensor], basis: torch.Tensor,
    ) -> "SharedInputLinearBank":
        if set(weights) != set(QK_NAMES) or basis.ndim != 2:
            raise ValueError("shared QK source is malformed")
        width, rank = basis.shape
        if rank <= 0 or any(weight.shape != (width, width) for weight in weights.values()):
            raise ValueError("shared QK source topology changed")
        return cls(
            basis.T,
            {name: weight.float() @ basis.float() for name, weight in weights.items()},
        )

    @property
    def stored_values(self) -> int:
        return self.input_factor.numel() + sum(
            getattr(self, f"output_{name}").numel() for name in QK_NAMES
        )

    def forward(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in QK_NAMES:
            raise KeyError("unknown shared QK decoder")
        hidden = F.linear(value, self.input_factor.to(value.dtype))
        return F.linear(hidden, getattr(self, f"output_{name}").to(value.dtype))


class TensorPreservingSquaredAttention(nn.Module):
    """Squared bilinear attention with compressed typed projections.

    The operator retains QK head RMSNorm, RoPE, the product of two QK score
    contractions, the causal mask, the cross-layer first-value bus, c_proj, and the
    caller-owned residual addition.  No native attention object is retained.
    """

    def __init__(
        self, projections: Mapping[str, StoredLinear], *, lamb: torch.Tensor | float,
        inv_freq: torch.Tensor, n_head: int,
        shared_qk: SharedInputLinearBank | None = None,
        head_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        expected = {"v", "proj"} if shared_qk is not None else set(PROJECTION_NAMES)
        if set(projections) != expected:
            raise ValueError("attention projection set is incomplete")
        if type(n_head) is not int or n_head <= 0:
            raise ValueError("n_head must be positive")
        self.projections = nn.ModuleDict(dict(projections))
        self.shared_qk = shared_qk
        width = (
            shared_qk.width if shared_qk is not None
            else self.projections["q"].output_dim
        )
        if width % n_head:
            raise ValueError("projection width is not divisible by head count")
        if any(
            layer.input_dim != width or layer.output_dim != width
            for layer in self.projections.values()
        ):
            raise ValueError("attention projections are not square with a common width")
        if any(
            not bool(torch.isfinite(buffer.detach()).all())
            for layer in self.projections.values()
            for buffer in layer.buffers()
            if buffer is not None
        ):
            raise ValueError("attention projection contains nonfinite values")
        if inv_freq.ndim != 1 or inv_freq.numel() * 2 != width // n_head:
            raise ValueError("rotary frequencies do not match head width")
        if not inv_freq.is_floating_point() or not bool(torch.isfinite(inv_freq).all()):
            raise ValueError("rotary frequencies are malformed")
        scalar_lamb = torch.as_tensor(lamb)
        if scalar_lamb.numel() != 1 or not scalar_lamb.is_floating_point() or not bool(
            torch.isfinite(scalar_lamb).all()
        ):
            raise ValueError("attention lambda is malformed")
        self.width = int(width)
        self.n_head = n_head
        self.head_dim = self.width // n_head
        if head_weights is not None and (
            head_weights.ndim != 1 or head_weights.shape[0] != n_head
            or not head_weights.is_floating_point()
            or not bool(torch.isfinite(head_weights).all())
        ):
            raise ValueError("head weights must be one finite scalar per head")
        self.register_buffer("lamb", scalar_lamb.detach().clone().reshape(()))
        self.register_buffer("inv_freq", inv_freq.detach().clone())
        self.register_buffer(
            "head_weights",
            None if head_weights is None else head_weights.detach().clone(),
        )

    @classmethod
    def from_native(
        cls, attention: nn.Module, *, ranks: Mapping[str, int | None],
        head_weights: torch.Tensor | None = None,
    ) -> "TensorPreservingSquaredAttention":
        expected = {"q": "c_q", "k": "c_k", "q2": "c_q2", "k2": "c_k2",
                    "v": "c_v", "proj": "c_proj"}
        if set(ranks) != set(expected):
            raise ValueError("rank specification is incomplete")
        if int(getattr(attention, "n_embd", -1)) <= 0 or int(
            getattr(attention, "n_head", -1)
        ) <= 0 or int(attention.n_embd) % int(attention.n_head):
            raise ValueError("native attention topology is malformed")
        projections = {
            name: StoredLinear.from_weight(
                getattr(attention, source).weight.detach(), ranks[name],
            )
            for name, source in expected.items()
        }
        return cls(
            projections, lamb=attention.lamb.detach(),
            inv_freq=attention.rotary.inv_freq.detach(), n_head=int(attention.n_head),
            head_weights=head_weights,
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
            if name in QK_NAMES and self.shared_qk is not None:
                projected = self.shared_qk(name, state)
            else:
                projected = self.projections[name](state)
            return projected.view(
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
        if first_value is None:
            bus = value
            bus_for_mixing = value
        else:
            if first_value.shape != value.shape or first_value.dtype != value.dtype or (
                first_value.device != value.device
            ) or not bool(torch.isfinite(first_value.detach()).all()):
                raise ValueError("first-value bus is malformed")
            bus = first_value
            bus_for_mixing = first_value.view_as(value)
        mixed = (1 - self.lamb) * value + self.lamb * bus_for_mixing
        output = torch.einsum(
            "bhqk,bkhd->bqhd", pattern.to(mixed.dtype), mixed,
        )
        if self.head_weights is not None:
            # A constant diagonal tensor on the head leg.  This is a global fixed
            # circuit edit, not a token-, context-, or target-dependent router.
            output = output * self.head_weights.to(
                device=output.device, dtype=output.dtype,
            )[None, None, :, None]
        output = output.reshape(batch, sequence, self.width)
        return self.projections["proj"](output), bus

    def cost_receipt(self) -> AttentionCostReceipt:
        prices = {name: layer.stored_values for name, layer in self.projections.items()}
        if self.shared_qk is not None:
            prices["qk_shared"] = self.shared_qk.stored_values
        if self.head_weights is not None:
            prices["head_weights"] = self.head_weights.numel()
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
        if self.shared_qk is not None:
            projection += self.shared_qk.stored_values
        for layer in self.projections.values():
            if layer.weight is not None:
                projection += math.prod(layer.weight.shape)
            else:
                assert layer.input_factor is not None and layer.output_factor is not None
                projection += layer.input_factor.numel() + layer.output_factor.numel()
        # Two QK contractions and one pattern-value contraction.
        contractions = 3 * self.n_head * sequence * sequence * self.head_dim
        return batch * (sequence * projection + contractions)


class TensorAttentionBank(nn.Module):
    """An owned 18-site program bank with no reference to native attention modules."""

    def __init__(self, programs: Sequence[TensorPreservingSquaredAttention]) -> None:
        super().__init__()
        if not programs or not all(
            isinstance(program, TensorPreservingSquaredAttention) for program in programs
        ):
            raise ValueError("tensor attention bank is malformed")
        widths = {program.width for program in programs}
        heads = {program.n_head for program in programs}
        if len(widths) != 1 or len(heads) != 1:
            raise ValueError("tensor attention bank topology is inconsistent")
        self.programs = nn.ModuleList(programs)

    @classmethod
    def from_model(
        cls, model: nn.Module, *, ranks: Mapping[str, int | None],
    ) -> "TensorAttentionBank":
        blocks = tuple(model.transformer.h)
        if len(blocks) != 18:
            raise ValueError("production tensor attention bank requires 18 blocks")
        bank = cls([
            TensorPreservingSquaredAttention.from_native(block.attn, ranks=ranks)
            for block in blocks
        ])
        try:
            parameter = next(model.parameters())
        except StopIteration as error:
            raise ValueError("model has no device-bearing parameter") from error
        return bank.to(device=parameter.device, dtype=parameter.dtype)

    def begin(self, blocks: Sequence[nn.Module]) -> "TensorAttentionTransaction":
        return TensorAttentionTransaction(self, blocks)

    def cost_receipt(self) -> dict[str, object]:
        layers = [asdict(program.cost_receipt()) for program in self.programs]
        return {
            "layers": layers,
            "total_stored_values": sum(int(row["total_stored_values"]) for row in layers),
            "token_table_values": sum(int(row["token_table_values"]) for row in layers),
            "native_calls_per_forward": 0,
            "total_input_support": all(bool(row["total_input_support"]) for row in layers),
        }


class TensorAttentionTransaction(AbstractContextManager):
    """One ordered synchronous dispatch; tensor aliases are revoked on close."""

    def __init__(self, bank: TensorAttentionBank, blocks: Sequence[nn.Module]) -> None:
        self._bank: TensorAttentionBank | None = bank
        self._blocks: tuple[nn.Module, ...] = tuple(blocks)
        if len(self._blocks) != len(bank.programs):
            raise ValueError("attention transaction block count changed")
        self._next_site = 0
        self._root_bus: torch.Tensor | None = None
        self._block_identity = True
        self._bus_identity = True
        self._closed = False
        self._closure: AttentionBankClosure | None = None

    def __enter__(self) -> "TensorAttentionTransaction":
        if self._closed:
            raise RuntimeError("attention transaction is closed")
        return self

    @property
    def closure(self) -> AttentionBankClosure:
        if not self._closed or self._closure is None:
            raise RuntimeError("attention transaction has no completed closure")
        return self._closure

    def __call__(self, event) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed or self._bank is None:
            raise RuntimeError("attention transaction is closed")
        site = self._next_site
        if int(getattr(event, "site", -1)) != site:
            raise RuntimeError("attention program dispatch is missing, repeated, or reordered")
        if getattr(event, "block", None) is not self._blocks[site]:
            self._block_identity = False
            raise RuntimeError("attention program received the wrong block identity")
        incoming = getattr(event, "first_value", None)
        if site == 0:
            if incoming is not None:
                self._bus_identity = False
                raise RuntimeError("attention0 must mint the first-value bus")
        elif incoming is not self._root_bus:
            self._bus_identity = False
            raise RuntimeError("attention first-value bus identity changed")
        write, bus = self._bank.programs[site](event.state, incoming)
        if site == 0:
            self._root_bus = bus
        elif bus is not self._root_bus:
            self._bus_identity = False
            raise RuntimeError("attention program replaced the first-value bus")
        self._next_site += 1
        return write, bus

    def close(self) -> AttentionBankClosure:
        if self._closed:
            raise RuntimeError("attention transaction is already closed")
        complete = self._bank is not None and self._next_site == len(self._blocks)
        sites = tuple((site, 1 if site < self._next_site else 0)
                      for site in range(len(self._blocks)))
        closure = AttentionBankClosure(
            sites=sites, ordered=complete, block_identity=self._block_identity,
            first_value_identity=self._bus_identity, closed=True,
        )
        self._root_bus = None
        self._blocks = ()
        self._bank = None
        self._closed = True
        self._closure = closure
        if not complete:
            raise RuntimeError("attention transaction did not dispatch every site exactly once")
        return closure

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._closed:
            return None
        if exc_type is None:
            self.close()
        else:
            self._root_bus = None
            self._blocks = ()
            self._bank = None
            self._closed = True
        return None
