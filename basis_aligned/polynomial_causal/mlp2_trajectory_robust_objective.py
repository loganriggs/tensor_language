"""Pure objective/checkpoint contract for paired-background MLP2 fitting."""

from __future__ import annotations

import math

import torch


def centered_target_energy(target: torch.Tensor) -> torch.Tensor:
    """Mean squared target deviation, used only from the frozen fit role."""
    if not target.is_floating_point() or target.numel() == 0:
        raise ValueError("target must be a nonempty floating tensor")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")
    flat = target.reshape(-1, target.shape[-1])
    energy = (flat - flat.mean(dim=0, keepdim=True)).square().mean()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("centered target energy must be positive and finite")
    return energy.detach()


def normalized_mse(
    prediction: torch.Tensor, target: torch.Tensor, target_energy: torch.Tensor | float,
) -> torch.Tensor:
    """Scalar MSE divided by a precomputed fit-only centered target energy."""
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must have one equal nonempty shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must be floating tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    energy = torch.as_tensor(target_energy, device=prediction.device,
                             dtype=prediction.dtype)
    if energy.numel() != 1 or not torch.isfinite(energy) or energy <= 0:
        raise ValueError("target_energy must be one positive finite scalar")
    return (prediction - target).square().mean() / energy


def balanced_background_loss(
    native_prediction: torch.Tensor,
    native_target: torch.Tensor,
    c512_prediction: torch.Tensor,
    c512_target: torch.Tensor,
    native_energy: torch.Tensor | float,
    c512_energy: torch.Tensor | float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Give native and C512 backgrounds exactly equal normalized loss weight."""
    native = normalized_mse(native_prediction, native_target, native_energy)
    c512 = normalized_mse(c512_prediction, c512_target, c512_energy)
    total = 0.5 * (native + c512)
    return total, {"native_normalized_mse": native, "c512_normalized_mse": c512}


def robust_checkpoint_key(native_normalized_mse: float,
                          c512_normalized_mse: float) -> tuple[float, float]:
    """Minimize worst background first and their mean second."""
    values = (native_normalized_mse, c512_normalized_mse)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("checkpoint losses must be finite and nonnegative")
    return max(values), 0.5 * sum(values)


def retain_checkpoint(
    observed_native: float,
    observed_c512: float,
    best_native: float,
    best_c512: float,
) -> bool:
    """Lexicographic minimax rule frozen by the preregistration."""
    return robust_checkpoint_key(observed_native, observed_c512) < robust_checkpoint_key(
        best_native, best_c512,
    )

