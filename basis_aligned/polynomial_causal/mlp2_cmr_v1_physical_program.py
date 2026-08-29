"""Owned physical K-channel bilinear program for the frozen MLP2 CMR assay."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicalProgramReceipt:
    input_width: int
    output_width: int
    native_products: int
    retained_products: int
    stored_scalar_values: int
    support_index_values: int
    bilinear_products_per_token: int
    native_mlp_calls_per_forward: int


def _owned_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point() or not bool(
        torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite floating tensor")
    return value.detach().clone().contiguous()


class PhysicalRetainedBilinearMLP(nn.Module):
    """Compute only retained native products, with omitted means folded into bias.

    No reference to the native module is retained.  The forward pass materializes a
    K-dimensional product vector, never the native H-dimensional product vector.
    """

    def __init__(
        self, left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
        folded_bias: torch.Tensor, support: torch.Tensor, *, native_products: int,
    ) -> None:
        super().__init__()
        left = _owned_finite(left, "left")
        right = _owned_finite(right, "right")
        down = _owned_finite(down, "down")
        folded_bias = _owned_finite(folded_bias, "folded_bias")
        support = torch.as_tensor(support, dtype=torch.long).detach().clone().contiguous()
        if left.ndim != 2 or right.shape != left.shape or down.ndim != 2:
            raise ValueError("retained bilinear matrices are malformed")
        retained, input_width = left.shape
        output_width = down.shape[0]
        if down.shape[1] != retained or folded_bias.shape != (output_width,):
            raise ValueError("retained bilinear topology changed")
        if support.shape != (retained,) or torch.unique(support).numel() != retained:
            raise ValueError("support must contain one unique index per retained product")
        if type(native_products) is not int or not retained <= native_products or (
            retained and (int(support.min()) < 0 or int(support.max()) >= native_products)
        ):
            raise ValueError("support is outside the native product range")
        if not (left.dtype == right.dtype == down.dtype == folded_bias.dtype):
            raise ValueError("physical coefficients must share one declared dtype")
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.register_buffer("down", down)
        self.register_buffer("folded_bias", folded_bias)
        self.register_buffer("support", support)
        self.input_width = input_width
        self.output_width = output_width
        self.native_products = native_products
        self.retained_products = retained

    @classmethod
    def from_native(
        cls, native: nn.Module, mean: torch.Tensor, support: torch.Tensor,
    ) -> "PhysicalRetainedBilinearMLP":
        required = ("Left", "Right", "Down", "Down_bias")
        if any(not hasattr(native, name) for name in required):
            raise ValueError("native module is not an ungated bilinear MLP")
        left = native.Left.weight.detach()
        right = native.Right.weight.detach()
        down = native.Down.weight.detach()
        bias = native.Down_bias.detach()
        if left.ndim != 2 or right.shape != left.shape or down.shape != (
            bias.numel(), left.shape[0]
        ) or bias.shape != (down.shape[0],):
            raise ValueError("native bilinear topology changed")
        if not (left.dtype == right.dtype == down.dtype == bias.dtype):
            raise ValueError("native physical coefficients do not share one dtype")
        mean = torch.as_tensor(mean, device=down.device)
        if mean.shape != (left.shape[0],) or not mean.is_floating_point() or not bool(
            torch.isfinite(mean).all()
        ):
            raise ValueError("omitted-product mean is malformed")
        support = torch.as_tensor(support, dtype=torch.long, device=down.device)
        if support.ndim != 1 or not 0 < support.numel() <= left.shape[0] or (
            torch.unique(support).numel() != support.numel()
        ) or int(support.min()) < 0 or int(support.max()) >= left.shape[0]:
            raise ValueError("retained support is malformed")
        omitted_mask = torch.ones(left.shape[0], dtype=torch.bool, device=down.device)
        omitted_mask[support] = False
        omitted = torch.nonzero(omitted_mask, as_tuple=False).flatten()
        # Compute the one-time fold accurately, then publish in the native runtime dtype.
        folded_bias = (
            bias.double() + down[:, omitted].double() @ mean[omitted].double()
        ).to(dtype=bias.dtype)
        return cls(
            left[support], right[support], down[:, support], folded_bias,
            support.cpu(), native_products=left.shape[0],
        ).to(device=left.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != self.input_width:
            raise ValueError("physical MLP state shape changed")
        if state.device != self.left.device:
            raise ValueError("physical MLP state device changed")
        left = F.linear(state, self.left.to(dtype=state.dtype))
        right = F.linear(state, self.right.to(dtype=state.dtype))
        product = left * right
        if product.shape[-1] != self.retained_products:
            raise RuntimeError("physical program materialized the wrong product width")
        return F.linear(
            product, self.down.to(dtype=state.dtype),
            self.folded_bias.to(dtype=state.dtype),
        )

    def receipt(self) -> PhysicalProgramReceipt:
        stored = sum(tensor.numel() for tensor in (
            self.left, self.right, self.down, self.folded_bias,
        ))
        return PhysicalProgramReceipt(
            input_width=self.input_width,
            output_width=self.output_width,
            native_products=self.native_products,
            retained_products=self.retained_products,
            stored_scalar_values=stored,
            support_index_values=self.support.numel(),
            bilinear_products_per_token=self.retained_products,
            native_mlp_calls_per_forward=0,
        )

    def receipt_dict(self) -> dict[str, int]:
        return asdict(self.receipt())


def zero_mlp_write(state: torch.Tensor) -> torch.Tensor:
    """The preregistered ZERO arm: no variable write and no Down bias."""
    if not torch.is_tensor(state) or state.ndim != 3:
        raise ValueError("ZERO arm state is malformed")
    return torch.zeros_like(state)
