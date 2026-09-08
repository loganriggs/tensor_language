"""Exact and tangent transport through gain-free RMS normalization."""
from __future__ import annotations

import torch


class RMSNormTransportError(ValueError):
    pass


def _validate(x, delta, eps):
    x, delta = torch.as_tensor(x), torch.as_tensor(delta)
    if x.shape != delta.shape or x.ndim < 1 or x.shape[-1] < 1:
        raise RMSNormTransportError("x and delta must have equal nonempty final dimensions")
    if not torch.is_floating_point(x) or not torch.is_floating_point(delta):
        raise RMSNormTransportError("x and delta must be floating point")
    if not torch.isfinite(x).all() or not torch.isfinite(delta).all():
        raise RMSNormTransportError("x and delta must be finite")
    eps = float(eps)
    if not eps > 0:
        raise RMSNormTransportError("eps must be positive")
    return x, delta, eps


def rmsnorm_value(x, eps):
    x = torch.as_tensor(x)
    if not torch.is_floating_point(x) or x.ndim < 1 or not torch.isfinite(x).all():
        raise RMSNormTransportError("x must be a finite floating tensor")
    eps = float(eps)
    if not eps > 0:
        raise RMSNormTransportError("eps must be positive")
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def exact_delta(x, delta, eps):
    """Return RMSNorm(x + delta) - RMSNorm(x) without linearization."""
    x, delta, eps = _validate(x, delta, eps)
    return rmsnorm_value(x + delta, eps) - rmsnorm_value(x, eps)


def tangent_delta(x, delta, eps):
    """Apply the exact Jacobian of RMSNorm at x to delta."""
    x, delta, eps = _validate(x, delta, eps)
    dimension = x.shape[-1]
    scale = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    radial = (x * delta).sum(dim=-1, keepdim=True) / dimension
    return delta / scale - x * radial / scale.pow(3)


def transport_split(x, delta, eps):
    """Return exact, tangent, and finite-dose remainder with an exact closure."""
    exact = exact_delta(x, delta, eps)
    tangent = tangent_delta(x, delta, eps)
    return {"exact": exact, "tangent": tangent, "remainder": exact - tangent}
