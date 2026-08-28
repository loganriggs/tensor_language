from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensor_preserving_mlp import (
    TensorMLPBank, TensorPreservingBilinearMLP,
)


class FakeBilinear(nn.Module):
    def __init__(self, width: int = 6, hidden: int = 12) -> None:
        super().__init__()
        self.Left = nn.Linear(width, hidden, bias=False)
        self.Right = nn.Linear(width, hidden, bias=False)
        self.Down = nn.Linear(hidden, width, bias=False)
        self.Down_bias = nn.Parameter(torch.randn(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.Down(self.Left(value) * self.Right(value)) + self.Down_bias


def test_dense_program_is_exact_and_clones_bias() -> None:
    torch.manual_seed(81)
    native = FakeBilinear()
    value = torch.randn(3, 5, 6)
    expected = native(value)
    program = TensorPreservingBilinearMLP.from_native(native)
    torch.testing.assert_close(program(value), expected)
    with torch.no_grad():
        native.Down_bias.add_(100)
        native.Left.weight.zero_()
    torch.testing.assert_close(program(value), expected)
    assert program.down_bias.untyped_storage().data_ptr() != (
        native.Down_bias.untyped_storage().data_ptr()
    )


def test_bias_is_load_bearing_and_fully_priced() -> None:
    torch.manual_seed(82)
    native = FakeBilinear(width=4, hidden=9)
    program = TensorPreservingBilinearMLP.from_native(native)
    value = torch.zeros(2, 3, 4)
    torch.testing.assert_close(program(value), native.Down_bias.expand_as(value))
    receipt = program.cost_receipt()
    assert receipt.bias_values == 4
    assert receipt.total_stored_values == 2 * 4 * 9 + 9 * 4 + 4
    operations = program.multiply_adds(batch=2, sequence=3)
    assert operations["linear_multiply_adds"] == 6 * (2 * 4 * 9 + 9 * 4)
    assert operations["bilinear_multiplies"] == 6 * 9


def test_factored_program_has_expected_complete_cost() -> None:
    torch.manual_seed(83)
    native = FakeBilinear(width=6, hidden=12)
    program = TensorPreservingBilinearMLP.from_native(
        native, ranks={"left": 3, "right": 4, "down": 5},
    )
    receipt = program.cost_receipt()
    assert receipt.projection_values == {
        "left": 3 * (6 + 12),
        "right": 4 * (6 + 12),
        "down": 5 * (12 + 6),
    }
    assert receipt.total_input_support and receipt.native_calls_per_forward == 0


def test_bank_transaction_requires_order_and_block_identity() -> None:
    torch.manual_seed(84)
    blocks = tuple(SimpleNamespace(mlp=FakeBilinear()) for _ in range(3))
    bank = TensorMLPBank([
        TensorPreservingBilinearMLP.from_native(block.mlp) for block in blocks
    ])
    with bank.begin(blocks) as transaction:
        for site, block in enumerate(blocks):
            event = SimpleNamespace(site=site, block=block, state=torch.randn(2, 4, 6))
            torch.testing.assert_close(transaction(event), block.mlp(event.state))
    assert transaction.closure.ordered
    assert transaction.closure.block_identity
    assert transaction.closure.sites == ((0, 1), (1, 1), (2, 1))
    with pytest.raises(RuntimeError, match="closed"):
        transaction(SimpleNamespace(site=0, block=blocks[0], state=torch.randn(2, 4, 6)))


def test_program_rejects_missing_down_bias_and_malformed_state() -> None:
    native = FakeBilinear()
    program = TensorPreservingBilinearMLP.from_native(native)
    with pytest.raises(ValueError, match="state shape"):
        program(torch.randn(2, 6))
    del native.Down_bias
    with pytest.raises(ValueError, match="Down bias"):
        TensorPreservingBilinearMLP.from_native(native)
