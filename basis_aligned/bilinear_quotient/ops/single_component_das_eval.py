#!/usr/bin/env python3
"""Closure-gated constrained DAS inside one native component.

Unlike the invalid cross-layer concatenation, all selected coordinates here are simultaneous
slices of one attention module or one MLP.  Therefore q=I must reproduce exact interchange.
Learned axes are restricted to the empirical donor-minus-base delta span, and DIM is an explicit
initial candidate, so optimization can never return a worse fit objective than that baseline.
"""

# BQGATE: LIBRARY
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import circuit_unit_greedy as g


class SingleComponentDASError(RuntimeError):
    pass


def validate_units(units: Sequence[str]):
    units = tuple(units)
    if not units or len(units) != len(set(units)):
        raise SingleComponentDASError("units must be a nonempty unique sequence")
    layers = {g.unit_layer(unit) for unit in units}
    kinds = {"head" if ":head:" in unit else "neuron" if ":neuron:" in unit else "mlp"
             for unit in units}
    if len(layers) != 1 or len(kinds) != 1:
        raise SingleComponentDASError(
            "DAS units must be simultaneous coordinates inside one native component")
    if "mlp" in kinds and len(units) != 1:
        raise SingleComponentDASError("a whole-MLP output is already one complete component")
    return next(iter(layers)), next(iter(kinds))


def cached_delta_matrix(backend, prep, units):
    validate_units(units)
    torch = backend.torch

    def cached(cache, row_id, unit):
        if ":neuron:" in unit:
            index = int(unit.rsplit(":", 1)[1])
            return torch.as_tensor(cache[(row_id, g.hidden_key(g.unit_layer(unit)))])[index:index + 1]
        return torch.as_tensor(cache[(row_id, unit)])

    return torch.stack([
        torch.cat([
            cached(prep.donor_cache, row_id, unit).float()
            - cached(prep.base_cache, row_id, unit).float()
            for unit in units
        ])
        for row_id in prep.base_batch.row_ids
    ]).to(backend.device)


def empirical_span(delta, relative_tolerance=1e-6):
    """Return orthonormal rows spanning observed intervention deltas."""
    torch = __import__("torch")
    if delta.ndim != 2 or not delta.shape[0] or not delta.shape[1]:
        raise SingleComponentDASError("delta matrix must be nonempty and two-dimensional")
    _u, singular, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    if not singular.numel() or float(singular[0]) <= 0:
        raise SingleComponentDASError("intervention deltas have zero empirical rank")
    rank = int((singular > singular[0] * relative_tolerance).sum())
    return vh[:rank], singular, rank


def _axis(out):
    return -(out[:, 0] - out[:, 1])


def normalized_objective(backend, prep, units, q, exact_axis=None, complement_weight=1.0):
    """Causal match plus complement-inertness in per-row donor-recovery units."""
    torch = backend.torch
    if exact_axis is None:
        exact_axis = _axis(g.forward_units(
            backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
            base_cache=prep.base_cache))
    base = torch.tensor(prep.base_axis, device=backend.device)
    donor = torch.tensor(prep.donor_axis, device=backend.device)
    denominator = donor - base
    if bool((denominator <= 1e-6).any()):
        raise SingleComponentDASError("fit rows require positive donor-oriented denominators")
    sub = _axis(g.forward_units(
        backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
        base_cache=prep.base_cache, q=q, grad=True))
    comp = _axis(g.forward_units(
        backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
        base_cache=prep.base_cache, q=q, grad=True, complement=True))
    match = ((((sub - base) - (exact_axis - base)) / denominator) ** 2).mean()
    inert = (((comp - base) / denominator) ** 2).mean()
    return match + complement_weight * inert, match, inert


def identity_closure(backend, prep, units):
    validate_units(units)
    torch = backend.torch
    exact = g.forward_units(
        backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
        base_cache=prep.base_cache)
    dimension = sum(g.unit_dim(unit) for unit in units)
    identity = torch.eye(dimension, device=backend.device)
    full = g.forward_units(
        backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
        base_cache=prep.base_cache, q=identity)
    return {
        "dimension": dimension,
        "max_abs_logit_error": float((full - exact).abs().max()),
        "mean_abs_logit_error": float((full - exact).abs().mean()),
    }


@dataclass
class FitResult:
    q: object
    span_rank: int
    singular_values: list[float]
    dim_objective: dict[str, float]
    best_objective: dict[str, float]
    selected_start: str
    restarts: list[dict]


def fit(backend, prep, units, *, rank=1, steps=200, lr=0.03, random_seeds=(1, 2),
        complement_weight=1.0):
    """Fit in the observed delta span, always retaining DIM as a feasible baseline."""
    validate_units(units)
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    delta = cached_delta_matrix(backend, prep, units)
    span, singular, span_rank = empirical_span(delta)
    if rank > span_rank:
        raise SingleComponentDASError("requested rank exceeds empirical delta rank")
    exact_axis = _axis(g.forward_units(
        backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
        base_cache=prep.base_cache)).detach()
    q_dim = g.diff_in_means_direction(backend, prep, units)

    with torch.no_grad():
        loss, match, inert = normalized_objective(
            backend, prep, units, q_dim, exact_axis, complement_weight)
    dim_metrics = {"joint": float(loss), "match": float(match), "inert": float(inert)}
    global_q, global_metrics, global_start = q_dim.detach().clone(), dict(dim_metrics), "dim_unoptimized"
    reports = []
    starts = [("dim", None)] + [(f"random_seed_{seed}", seed) for seed in random_seeds]
    for name, seed in starts:
        if seed is None:
            coefficients0 = span @ q_dim
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            coefficients0 = torch.randn(span_rank, rank, generator=generator).to(backend.device)
        raw = coefficients0.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=lr)
        best_q, best = None, None
        trace = []
        for step in range(steps + 1):
            optimizer.zero_grad()
            coefficients, _ = torch.linalg.qr(raw)
            q = span.T @ coefficients[:, :rank]
            loss, match, inert = normalized_objective(
                backend, prep, units, q, exact_axis, complement_weight)
            metrics = {"joint": float(loss.detach()), "match": float(match.detach()),
                       "inert": float(inert.detach())}
            if best is None or metrics["joint"] < best["joint"]:
                best, best_q = metrics, q.detach().clone()
            if step % 50 == 0 or step == steps:
                trace.append({"step": step, **metrics})
            if step == steps:
                break
            loss.backward()
            optimizer.step()
        reports.append({"start": name, "best": best, "trace": trace,
                        "cosine_to_dim": float((best_q[:, 0] @ q_dim[:, 0]).abs())})
        if best["joint"] < global_metrics["joint"]:
            global_q, global_metrics, global_start = best_q, dict(best), name
    return FitResult(
        q=global_q, span_rank=span_rank,
        singular_values=[float(value) for value in singular.detach().cpu()],
        dim_objective=dim_metrics, best_objective=global_metrics,
        selected_start=global_start, restarts=reports,
    )
