"""Exact RMS-typed polarization of an attention-to-bilinear-MLP block.

For residual h, attention write a, RMS scale gamma(v), and bilinear MLP

    Q(z) = Down[(Left z) * (Right z)] + bias,

the post-attention MLP input is

    rms(h+a) = alpha * rms(h) + beta * a,
    alpha = gamma(h+a) / gamma(h), beta = gamma(h+a).

Expanding Q by bilinearity gives five typed output terms.  The identity is exact;
it is not a local Taylor approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def rms_scale(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point() or value.ndim < 1 or (
        value.shape[-1] <= 0
    ) or not torch.isfinite(value).all() or not isinstance(eps, float) or eps <= 0:
        raise ValueError("RMS input or epsilon is malformed")
    return torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


def bilinear_output(
    left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
    first: torch.Tensor, second: torch.Tensor,
) -> torch.Tensor:
    if left.ndim != 2 or right.shape != left.shape or down.ndim != 2 or (
        down.shape[1] != left.shape[0]
    ) or first.shape != second.shape or first.shape[-1] != left.shape[1] or any(
        value.dtype != first.dtype or value.device != first.device
        for value in (left, right, down, second)
    ):
        raise ValueError("bilinear factors or states are incompatible")
    return torch.nn.functional.linear(
        torch.nn.functional.linear(first, left)
        * torch.nn.functional.linear(second, right),
        down,
    )


def native_mlp(
    left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
    bias: torch.Tensor, value: torch.Tensor,
) -> torch.Tensor:
    output = bilinear_output(left, right, down, value, value)
    if bias.shape != (down.shape[0],) or bias.dtype != output.dtype or (
        bias.device != output.device
    ):
        raise ValueError("Down bias is incompatible")
    return output + bias


@dataclass(frozen=True, slots=True)
class PolarizedTerms:
    scaled_base_quadratic: torch.Tensor
    left_residual_right_attention: torch.Tensor
    left_attention_right_residual: torch.Tensor
    attention_quadratic: torch.Tensor
    bias: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor

    @property
    def output(self) -> torch.Tensor:
        return (
            self.scaled_base_quadratic
            + self.left_residual_right_attention
            + self.left_attention_right_residual
            + self.attention_quadratic
            + self.bias
        )


def polarized_terms(
    left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
    bias: torch.Tensor, residual: torch.Tensor, attention_write: torch.Tensor,
    *, eps: float,
) -> PolarizedTerms:
    if residual.shape != attention_write.shape:
        raise ValueError("residual and attention write must share a typed residual port")
    gamma0 = rms_scale(residual, eps=eps)
    gamma1 = rms_scale(residual + attention_write, eps=eps)
    normalized = gamma0 * residual
    alpha = gamma1 / gamma0
    beta = gamma1
    base = bilinear_output(left, right, down, normalized, normalized)
    cross_lr = bilinear_output(left, right, down, normalized, attention_write)
    cross_rl = bilinear_output(left, right, down, attention_write, normalized)
    attention_square = bilinear_output(
        left, right, down, attention_write, attention_write,
    )
    broadcast = (1,) * (residual.ndim - 1) + (len(bias),)
    return PolarizedTerms(
        scaled_base_quadratic=alpha.square() * base,
        left_residual_right_attention=alpha * beta * cross_lr,
        left_attention_right_residual=alpha * beta * cross_rl,
        attention_quadratic=beta.square() * attention_square,
        bias=bias.reshape(broadcast).expand_as(base),
        alpha=alpha,
        beta=beta,
    )


def exact_replay_error(
    left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
    bias: torch.Tensor, residual: torch.Tensor, attention_write: torch.Tensor,
    *, eps: float,
) -> float:
    observed = native_mlp(
        left, right, down, bias,
        rms_scale(residual + attention_write, eps=eps) * (residual + attention_write),
    )
    replay = polarized_terms(
        left, right, down, bias, residual, attention_write, eps=eps,
    ).output
    denominator = torch.linalg.vector_norm(observed.double()).clamp_min(
        torch.finfo(torch.float64).tiny,
    )
    return float(torch.linalg.vector_norm((observed - replay).double()) / denominator)
