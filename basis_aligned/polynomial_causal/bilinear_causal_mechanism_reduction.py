"""Gauge-invariant mechanism replacement primitives for bilin18 MLP products.

The hidden variables of one bilin18 MLP are

    a_j(x) = (L_j x) (R_j x),       y(x) = D a(x) + b.

Each channel has the exact gauge

    L_j -> s_j L_j,  R_j -> t_j R_j,  D_:j -> D_:j / (s_j t_j),

for nonzero scales.  Scores used to select mechanisms must respect this gauge.
This module adapts causal mechanism reduction (CMR) constant/affine folding,
joint logit distortion, and the margin certificate to this physical interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FoldedAffine:
    """Dense compiled replacement for a subset of product channels."""

    kept: torch.Tensor
    down: torch.Tensor
    bias: torch.Tensor


def _validate_weights(
    left: torch.Tensor, right: torch.Tensor, down: torch.Tensor,
) -> tuple[int, int, int]:
    if left.ndim != 2 or right.ndim != 2 or down.ndim != 2:
        raise ValueError("left, right, and down must be matrices")
    if left.shape != right.shape or down.shape[1] != left.shape[0]:
        raise ValueError("incompatible bilinear factor shapes")
    products, input_width = left.shape
    output_width = down.shape[0]
    return products, input_width, output_width


def product_activations(
    states: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact bilinear product variables on leading-batch states."""
    _validate_weights(left, right, torch.empty(
        (1, left.shape[0]), dtype=left.dtype, device=left.device,
    ))
    if states.shape[-1] != left.shape[1]:
        raise ValueError("state width does not match bilinear input width")
    return (states @ left.T) * (states @ right.T)


def bilinear_write(
    states: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``D ((Lx) odot (Rx)) + b``."""
    _, _, output_width = _validate_weights(left, right, down)
    if bias.shape != (output_width,):
        raise ValueError("bias must match bilinear output width")
    return product_activations(states, left, right) @ down.T + bias


def apply_channel_gauge(
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    left_scales: torch.Tensor,
    right_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the exact per-product two-scalar gauge without changing the map."""
    products, _, _ = _validate_weights(left, right, down)
    if left_scales.shape != (products,) or right_scales.shape != (products,):
        raise ValueError("one nonzero left/right scale is required per product")
    if torch.any(left_scales == 0) or torch.any(right_scales == 0):
        raise ValueError("bilinear gauge scales must be nonzero")
    product_scale = left_scales * right_scales
    return (
        left * left_scales[:, None],
        right * right_scales[:, None],
        down / product_scale[None, :],
    )


def cmr_logit_scores(activations: torch.Tensor, outgoing: torch.Tensor) -> torch.Tensor:
    """Per-channel constant-replacement score under affine logit distortion.

    This is ``Var(a_j) * ||D_:j||^2``.  It is exactly invariant under the
    bilinear channel gauge, unlike activation variance or a single factor norm.
    """
    if activations.ndim != 2 or outgoing.ndim != 2:
        raise ValueError("activations and outgoing must be matrices")
    if activations.shape[1] != outgoing.shape[1]:
        raise ValueError("activation channels must match outgoing columns")
    centered = activations - activations.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    return variance * outgoing.square().sum(dim=0)


def joint_logit_distortion(
    activations: torch.Tensor, outgoing: torch.Tensor,
) -> torch.Tensor:
    """Exact mean squared affine-output error for joint mean replacement."""
    if activations.ndim != 2 or outgoing.ndim != 2:
        raise ValueError("activations and outgoing must be matrices")
    if activations.shape[1] != outgoing.shape[1]:
        raise ValueError("activation channels must match outgoing columns")
    centered = activations - activations.mean(dim=0, keepdim=True)
    return (centered @ outgoing.T).square().sum(dim=1).mean()


def off_diagonal_fraction(
    activations: torch.Tensor, outgoing: torch.Tensor,
) -> torch.Tensor:
    """Relative failure of an additive sum of single-channel CMR scores.

    Zero means the joint distortion equals the sum of the diagonal scores.
    Positive or negative values expose cross-channel curvature/covariance terms.
    """
    joint = joint_logit_distortion(activations, outgoing)
    diagonal = cmr_logit_scores(activations, outgoing).sum()
    return (joint - diagonal) / joint.clamp_min(torch.finfo(joint.dtype).tiny)


def compile_constant_replacement(
    down: torch.Tensor,
    bias: torch.Tensor,
    replaced: torch.Tensor,
    constants: torch.Tensor,
) -> FoldedAffine:
    """Remove product channels and fold their constants into the output bias."""
    if down.ndim != 2 or bias.shape != (down.shape[0],):
        raise ValueError("invalid down/bias shapes")
    replaced = torch.as_tensor(replaced, dtype=torch.long, device=down.device)
    constants = torch.as_tensor(constants, dtype=down.dtype, device=down.device)
    if replaced.ndim != 1 or constants.shape != replaced.shape:
        raise ValueError("replaced indices and constants must be matching vectors")
    if replaced.numel() and (
        int(replaced.min()) < 0 or int(replaced.max()) >= down.shape[1]
    ):
        raise ValueError("replaced index out of range")
    if torch.unique(replaced).numel() != replaced.numel():
        raise ValueError("replaced indices must be unique")
    keep_mask = torch.ones(down.shape[1], dtype=torch.bool, device=down.device)
    keep_mask[replaced] = False
    kept = torch.nonzero(keep_mask, as_tuple=False).flatten()
    folded_bias = bias + down[:, replaced] @ constants
    return FoldedAffine(kept=kept, down=down[:, kept], bias=folded_bias)


def compile_affine_replacement(
    down: torch.Tensor,
    bias: torch.Tensor,
    replaced: torch.Tensor,
    intercept: torch.Tensor,
    coefficients: torch.Tensor,
) -> FoldedAffine:
    """Compile ``a_S := intercept + coefficients @ a_K`` into a smaller map."""
    constant = compile_constant_replacement(down, bias, replaced, intercept)
    coefficients = torch.as_tensor(
        coefficients, dtype=down.dtype, device=down.device,
    )
    if coefficients.shape != (replaced.numel(), constant.kept.numel()):
        raise ValueError("affine coefficients must map kept channels to replaced")
    folded_down = constant.down + down[:, replaced] @ coefficients
    return FoldedAffine(kept=constant.kept, down=folded_down, bias=constant.bias)


def certified_iia_lower_bound(
    margins: torch.Tensor, squared_logit_errors: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    """Empirical CMR margin-certificate lower bound on interchange agreement."""
    if margins.ndim != 1 or squared_logit_errors.ndim != 1 or (
        margins.shape != squared_logit_errors.shape
    ):
        raise ValueError("margins and squared errors must be matching vectors")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    small_margin = (margins <= 2.0 * epsilon).to(margins.dtype).mean()
    distortion = squared_logit_errors.mean()
    return (1.0 - small_margin - distortion / (epsilon * epsilon)).clamp(0.0, 1.0)

