from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlp2_cmr_v1_physical_program import (
    PhysicalRetainedBilinearMLP, zero_mlp_write,
)


class FakeBilinear(nn.Module):
    def __init__(self, width: int = 5, hidden: int = 11) -> None:
        super().__init__()
        self.Left = nn.Linear(width, hidden, bias=False, dtype=torch.float64)
        self.Right = nn.Linear(width, hidden, bias=False, dtype=torch.float64)
        self.Down = nn.Linear(hidden, width, bias=False, dtype=torch.float64)
        self.Down_bias = nn.Parameter(torch.randn(width, dtype=torch.float64))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.Down(self.Left(state) * self.Right(state)) + self.Down_bias


def test_physical_program_equals_explicit_mean_replacement() -> None:
    torch.manual_seed(203)
    native = FakeBilinear()
    state = torch.randn(3, 7, 5, dtype=torch.float64)
    mean = torch.randn(11, dtype=torch.float64)
    support = torch.tensor([9, 1, 7, 4])
    program = PhysicalRetainedBilinearMLP.from_native(native, mean, support)
    product = native.Left(state) * native.Right(state)
    omitted = torch.tensor([0, 2, 3, 5, 6, 8, 10])
    product[..., omitted] = mean[omitted]
    expected = native.Down(product) + native.Down_bias
    torch.testing.assert_close(program(state), expected, atol=1e-12, rtol=1e-12)
    assert program.left.shape == (4, 5)
    assert program.right.shape == (4, 5)
    assert program.down.shape == (5, 4)
    assert torch.equal(program.support.cpu(), support)


def test_program_owns_coefficients_and_prices_only_retained_products() -> None:
    torch.manual_seed(204)
    native = FakeBilinear(width=6, hidden=12)
    mean = torch.randn(12, dtype=torch.float64)
    support = torch.tensor([11, 3, 1, 8, 4])
    state = torch.randn(2, 4, 6, dtype=torch.float64)
    expected = PhysicalRetainedBilinearMLP.from_native(native, mean, support)(state)
    program = PhysicalRetainedBilinearMLP.from_native(native, mean, support)
    with torch.no_grad():
        native.Left.weight.zero_()
        native.Right.weight.zero_()
        native.Down.weight.zero_()
        native.Down_bias.add_(100)
    torch.testing.assert_close(program(state), expected)
    assert program.receipt_dict() == {
        "input_width": 6,
        "output_width": 6,
        "native_products": 12,
        "retained_products": 5,
        "stored_scalar_values": 2 * 6 * 5 + 6 * 5 + 6,
        "support_index_values": 5,
        "bilinear_products_per_token": 5,
        "native_mlp_calls_per_forward": 0,
    }


def test_production_dimensions_have_exact_frozen_price() -> None:
    left = torch.zeros(512, 1152, dtype=torch.bfloat16)
    right = torch.zeros_like(left)
    down = torch.zeros(1152, 512, dtype=torch.bfloat16)
    bias = torch.zeros(1152, dtype=torch.bfloat16)
    support = torch.arange(512)
    program = PhysicalRetainedBilinearMLP(
        left, right, down, bias, support, native_products=4608,
    )
    receipt = program.receipt()
    assert receipt.stored_scalar_values == 1_770_624
    assert receipt.bilinear_products_per_token == 512
    assert receipt.native_mlp_calls_per_forward == 0


def test_zero_arm_deletes_bias_and_variable_write() -> None:
    state = torch.randn(2, 3, 5)
    output = zero_mlp_write(state)
    assert torch.equal(output, torch.zeros_like(state))
    assert output.shape == state.shape and output.dtype == state.dtype


def test_program_rejects_duplicate_support_and_wrong_state() -> None:
    native = FakeBilinear()
    mean = torch.zeros(11, dtype=torch.float64)
    with pytest.raises(ValueError, match="support"):
        PhysicalRetainedBilinearMLP.from_native(native, mean, torch.tensor([1, 1]))
    program = PhysicalRetainedBilinearMLP.from_native(native, mean, torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="state shape"):
        program(torch.zeros(2, 5, dtype=torch.float64))
