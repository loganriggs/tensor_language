"""Owned zero-native-call bilinear MLP tensor program for bilin18."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn

from tensor_preserving_attention import StoredLinear


@dataclass(frozen=True)
class MLPCostReceipt:
    projection_values: dict[str, int]
    bias_values: int
    total_stored_values: int
    token_table_values: int
    native_calls_per_forward: int
    total_input_support: bool


@dataclass(frozen=True)
class MLPBankClosure:
    sites: tuple[tuple[int, int], ...]
    ordered: bool
    block_identity: bool
    closed: bool


class TensorPreservingBilinearMLP(nn.Module):
    """Exact or factored ``Down(Left(x) * Right(x)) + bias`` program."""

    def __init__(
        self, left: StoredLinear, right: StoredLinear, down: StoredLinear,
        down_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        if not all(isinstance(layer, StoredLinear) for layer in (left, right, down)):
            raise ValueError("MLP projections must be owned StoredLinear programs")
        width = left.input_dim
        hidden = left.output_dim
        if width <= 0 or hidden <= 0 or (
            right.input_dim != width or right.output_dim != hidden
            or down.input_dim != hidden or down.output_dim != width
        ):
            raise ValueError("bilinear MLP topology changed")
        if down_bias.ndim != 1 or down_bias.shape[0] != width or not (
            down_bias.is_floating_point() and bool(torch.isfinite(down_bias).all())
        ):
            raise ValueError("bilinear MLP Down bias is malformed")
        if any(
            not bool(torch.isfinite(buffer.detach()).all())
            for layer in (left, right, down)
            for buffer in layer.buffers()
            if buffer is not None
        ):
            raise ValueError("bilinear MLP projection contains nonfinite values")
        self.left = left
        self.right = right
        self.down = down
        self.register_buffer("down_bias", down_bias.detach().clone())
        self.width = width
        self.hidden = hidden

    @classmethod
    def from_native(
        cls, native: nn.Module, *, ranks: dict[str, int | None] | None = None,
    ) -> "TensorPreservingBilinearMLP":
        ranks = {"left": None, "right": None, "down": None} if ranks is None else ranks
        if set(ranks) != {"left", "right", "down"}:
            raise ValueError("MLP rank specification is incomplete")
        expected = {"left": "Left", "right": "Right", "down": "Down"}
        if not hasattr(native, "Down_bias"):
            raise ValueError("native bilinear MLP has no Down bias")
        layers = {
            name: StoredLinear.from_weight(
                getattr(native, source).weight.detach(), ranks[name],
            )
            for name, source in expected.items()
        }
        return cls(layers["left"], layers["right"], layers["down"], native.Down_bias.detach())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != self.width:
            raise ValueError("bilinear MLP state shape changed")
        product = self.left(state) * self.right(state)
        return self.down(product) + self.down_bias.to(state.dtype)

    def cost_receipt(self) -> MLPCostReceipt:
        prices = {
            "left": self.left.stored_values,
            "right": self.right.stored_values,
            "down": self.down.stored_values,
        }
        return MLPCostReceipt(
            projection_values=prices,
            bias_values=self.down_bias.numel(),
            total_stored_values=sum(prices.values()) + self.down_bias.numel(),
            token_table_values=0,
            native_calls_per_forward=0,
            total_input_support=True,
        )

    def multiply_adds(self, *, batch: int, sequence: int) -> dict[str, int]:
        if min(batch, sequence) <= 0:
            raise ValueError("batch and sequence must be positive")

        def price(layer: StoredLinear) -> int:
            if layer.weight is not None:
                return math.prod(layer.weight.shape)
            return layer.stored_values

        linear = batch * sequence * sum(price(layer) for layer in (
            self.left, self.right, self.down,
        ))
        bilinear = batch * sequence * self.hidden
        return {"linear_multiply_adds": linear, "bilinear_multiplies": bilinear}


class TensorMLPBank(nn.Module):
    """Owned ordered MLP program stack with no native-module references."""

    def __init__(self, programs: Sequence[TensorPreservingBilinearMLP]) -> None:
        super().__init__()
        if not programs or not all(
            isinstance(program, TensorPreservingBilinearMLP) for program in programs
        ):
            raise ValueError("tensor MLP bank is malformed")
        if len({program.width for program in programs}) != 1 or len({
            program.hidden for program in programs
        }) != 1:
            raise ValueError("tensor MLP bank topology is inconsistent")
        self.programs = nn.ModuleList(programs)

    @classmethod
    def from_model(
        cls, model: nn.Module, *, ranks: dict[str, int | None] | None = None,
    ) -> "TensorMLPBank":
        blocks = tuple(model.transformer.h)
        if len(blocks) != 18:
            raise ValueError("production tensor MLP bank requires 18 blocks")
        bank = cls([
            TensorPreservingBilinearMLP.from_native(block.mlp, ranks=ranks)
            for block in blocks
        ])
        try:
            parameter = next(model.parameters())
        except StopIteration as error:
            raise ValueError("model has no device-bearing parameter") from error
        return bank.to(device=parameter.device, dtype=parameter.dtype)

    def begin(self, blocks: Sequence[nn.Module]) -> "TensorMLPTransaction":
        return TensorMLPTransaction(self, blocks)

    def cost_receipt(self) -> dict[str, object]:
        layers = [asdict(program.cost_receipt()) for program in self.programs]
        return {
            "layers": layers,
            "total_stored_values": sum(int(row["total_stored_values"]) for row in layers),
            "token_table_values": 0,
            "native_calls_per_forward": 0,
            "total_input_support": all(bool(row["total_input_support"]) for row in layers),
        }


class TensorMLPTransaction(AbstractContextManager):
    """One ordered synchronous MLP dispatch with alias revocation on close."""

    def __init__(self, bank: TensorMLPBank, blocks: Sequence[nn.Module]) -> None:
        self._bank: TensorMLPBank | None = bank
        self._blocks = tuple(blocks)
        if len(self._blocks) != len(bank.programs):
            raise ValueError("MLP transaction block count changed")
        self._next_site = 0
        self._block_identity = True
        self._closed = False
        self._closure: MLPBankClosure | None = None

    def __enter__(self) -> "TensorMLPTransaction":
        if self._closed:
            raise RuntimeError("MLP transaction is closed")
        return self

    @property
    def closure(self) -> MLPBankClosure:
        if not self._closed or self._closure is None:
            raise RuntimeError("MLP transaction has no completed closure")
        return self._closure

    def __call__(self, event) -> torch.Tensor:
        if self._closed or self._bank is None:
            raise RuntimeError("MLP transaction is closed")
        site = self._next_site
        if int(getattr(event, "site", -1)) != site:
            raise RuntimeError("MLP program dispatch is missing, repeated, or reordered")
        if getattr(event, "block", None) is not self._blocks[site]:
            self._block_identity = False
            raise RuntimeError("MLP program received the wrong block identity")
        write = self._bank.programs[site](event.state)
        self._next_site += 1
        return write

    def close(self) -> MLPBankClosure:
        if self._closed:
            raise RuntimeError("MLP transaction is already closed")
        complete = self._bank is not None and self._next_site == len(self._blocks)
        closure = MLPBankClosure(
            sites=tuple(
                (site, 1 if site < self._next_site else 0)
                for site in range(len(self._blocks))
            ),
            ordered=complete,
            block_identity=self._block_identity,
            closed=True,
        )
        self._blocks = ()
        self._bank = None
        self._closed = True
        self._closure = closure
        if not complete:
            raise RuntimeError("MLP transaction did not dispatch every site exactly once")
        return closure

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._closed:
            return None
        if exc_type is None:
            self.close()
        else:
            self._blocks = ()
            self._bank = None
            self._closed = True
        return None
