"""Exact nonlinear hybrid-chain accounting for compiled-program drift.

Input ``logits[k]`` is the same batch evaluated with a compiled prefix ending at cut
k and the native suffix thereafter.  No linearization is used: adjacent differences
telescope exactly for arbitrary nonlinear blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HybridCertificate:
    cuts: int
    samples: int
    telescope_max_abs_error: float
    max_step_sum_bound: float
    max_end_to_end_logit_error: float
    median_cancellation_ratio: float
    p95_cancellation_ratio: float
    mean_ce_bound: float
    top1_certified_fraction: float
    top1_observed_unchanged_fraction: float
    maximum_ce_bound_violation: float


def _quantile_no_interpolation(values: torch.Tensor, q: float) -> float:
    ordered = values.flatten().sort().values
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def certify(logits: torch.Tensor, targets: torch.Tensor) -> HybridCertificate:
    """Certify a complete hybrid chain on fixed samples.

    ``logits`` has shape ``[cuts + 1, ..., vocabulary]`` and ``targets`` has the
    matching sample shape. Cut zero is native; the last cut is fully compiled.
    """

    value = torch.as_tensor(logits).detach().cpu().double()
    target = torch.as_tensor(targets).detach().cpu().long()
    if value.ndim < 3 or value.shape[0] < 2 or tuple(value.shape[1:-1]) != tuple(
        target.shape
    ):
        raise ValueError("logits/targets shapes do not define a hybrid chain")
    if value.shape[-1] < 2 or not torch.isfinite(value).all():
        raise ValueError("logits must be finite with vocabulary size at least two")
    if bool(((target < 0) | (target >= value.shape[-1])).any()):
        raise ValueError("target index is out of vocabulary range")

    steps = value[1:] - value[:-1]
    telescope = steps.sum(dim=0)
    endpoint = value[-1] - value[0]
    telescope_error = float((telescope - endpoint).abs().max())
    step_sum = steps.abs().amax(dim=-1).sum(dim=0)
    endpoint_inf = endpoint.abs().amax(dim=-1)
    ratio = step_sum / endpoint_inf.clamp_min(torch.finfo(torch.float64).tiny)

    native_top2 = value[0].topk(2, dim=-1)
    native_margin = native_top2.values[..., 0] - native_top2.values[..., 1]
    certified = native_margin > 2.0 * step_sum
    observed_unchanged = value[0].argmax(dim=-1) == value[-1].argmax(dim=-1)

    flat_target = target.reshape(-1)
    native_ce = F.cross_entropy(
        value[0].reshape(-1, value.shape[-1]), flat_target, reduction="none",
    ).reshape(target.shape)
    compiled_ce = F.cross_entropy(
        value[-1].reshape(-1, value.shape[-1]), flat_target, reduction="none",
    ).reshape(target.shape)
    ce_change = (compiled_ce - native_ce).abs()
    ce_bound = 2.0 * step_sum
    violation = (ce_change - ce_bound).clamp_min(0)

    return HybridCertificate(
        cuts=value.shape[0] - 1,
        samples=target.numel(),
        telescope_max_abs_error=telescope_error,
        max_step_sum_bound=float(step_sum.max()),
        max_end_to_end_logit_error=float(endpoint_inf.max()),
        median_cancellation_ratio=float(ratio.median()),
        p95_cancellation_ratio=_quantile_no_interpolation(ratio, 0.95),
        mean_ce_bound=float(ce_bound.mean()),
        top1_certified_fraction=float(certified.double().mean()),
        top1_observed_unchanged_fraction=float(observed_unchanged.double().mean()),
        maximum_ce_bound_violation=float(violation.max()),
    )
