"""Gauge-invariant cut analysis for finite-horizon tangent response operators.

The production consumer will supply block responses H[j, i] from an intervention
injected at depth i to a registered downstream test at depth j.  This module performs
only the finite-dimensional linear algebra: cut assembly, operator-Schmidt spectra,
optimal truncated factorization, and orthogonal gauge replay.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


BlockKey = tuple[int, int]


def _validate_dims(name: str, dims: Mapping[int, int]) -> dict[int, int]:
    if not dims:
        raise ValueError(f"{name} dimensions must be nonempty")
    checked: dict[int, int] = {}
    for site, width in dims.items():
        if type(site) is not int or type(width) is not int or width <= 0:
            raise ValueError(f"{name} dimensions are malformed")
        checked[site] = width
    return checked


def assemble_cut(
    blocks: Mapping[BlockKey, torch.Tensor],
    input_dims: Mapping[int, int],
    output_dims: Mapping[int, int],
    cut: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assemble downstream-tests by upstream-interventions across one depth cut.

    Every causal rectangle entry must be explicit, including structural zero blocks.
    This fail-closed rule prevents a missing measurement from becoming a zero response.
    """
    inputs = _validate_dims("input", input_dims)
    outputs = _validate_dims("output", output_dims)
    if type(cut) is not int:
        raise ValueError("cut must be an integer depth")
    upstream = tuple(site for site in sorted(inputs) if site < cut)
    downstream = tuple(site for site in sorted(outputs) if site >= cut)
    if not upstream or not downstream:
        raise ValueError("cut must separate at least one input and output site")
    rows = []
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for output_site in downstream:
        row = []
        for input_site in upstream:
            key = (output_site, input_site)
            if key not in blocks:
                raise ValueError(f"missing explicit response block {key}")
            block = blocks[key]
            expected = (outputs[output_site], inputs[input_site])
            if (
                not torch.is_tensor(block) or tuple(block.shape) != expected
                or not block.is_floating_point() or not bool(torch.isfinite(block).all())
            ):
                raise ValueError(f"response block {key} is malformed")
            if dtype is None:
                dtype, device = block.dtype, block.device
            elif block.dtype != dtype or block.device != device:
                raise ValueError("response blocks must share dtype and device")
            row.append(block)
        rows.append(torch.cat(row, dim=1))
    matrix = torch.cat(rows, dim=0).double()
    receipt = {
        "cut": cut,
        "upstream_input_sites": list(upstream),
        "downstream_output_sites": list(downstream),
        "row_dimension": matrix.shape[0],
        "column_dimension": matrix.shape[1],
        "explicit_blocks": len(upstream) * len(downstream),
    }
    return matrix, receipt


def analyze_cut(
    matrix: torch.Tensor, *, energy_fraction: float = 0.95,
    gap_ratio: float = 2.0, support_rtol: float = 1e-12,
) -> dict[str, Any]:
    """Return the exact cut rank and a prospective energy-plus-gap truncation."""
    if (
        not torch.is_tensor(matrix) or matrix.ndim != 2 or matrix.numel() == 0
        or not matrix.is_floating_point() or not bool(torch.isfinite(matrix).all())
    ):
        raise ValueError("cut response matrix is malformed")
    if not 0 < energy_fraction <= 1 or gap_ratio <= 1 or support_rtol <= 0:
        raise ValueError("cut analysis constants are malformed")
    singular = torch.linalg.svdvals(matrix.double())
    leading = float(singular[0]) if len(singular) else 0.0
    threshold = support_rtol * max(leading, 1.0)
    support_rank = int((singular > threshold).sum())
    energy = singular.square()
    total = float(energy.sum())
    selected: int | None = None
    selected_gap: float | None = None
    if support_rank > 1 and total > 0:
        cumulative = torch.cumsum(energy, dim=0) / total
        candidate = int(torch.searchsorted(
            cumulative, torch.tensor(energy_fraction, dtype=cumulative.dtype),
        )) + 1
        if 0 < candidate < support_rank:
            denominator = float(singular[candidate])
            gap = float(singular[candidate - 1]) / denominator if denominator > 0 else math.inf
            if gap >= gap_ratio:
                selected, selected_gap = candidate, gap
    tails = [float(energy[rank:].sum()) for rank in range(len(singular) + 1)]
    return {
        "exact_cut_rank": support_rank,
        "singular_values": [float(value) for value in singular],
        "squared_frobenius_energy": total,
        "optimal_squared_frobenius_tail_by_rank": tails,
        "energy_fraction_rule": energy_fraction,
        "gap_ratio_rule": gap_ratio,
        "selected_rank": selected,
        "selected_gap_ratio": selected_gap,
        "certified_compression_knee": selected is not None,
    }


def truncated_factorization(
    matrix: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Return the optimal rank-r cut factorization L @ R by Eckart--Young."""
    if type(rank) is not int or rank <= 0 or rank > min(matrix.shape):
        raise ValueError("factorization rank is outside the cut dimensions")
    u, singular, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    root = singular[:rank].clamp_min(0).sqrt()
    left = u[:, :rank] * root
    right = root[:, None] * vh[:rank]
    residual = matrix.double() - left @ right
    return left, right, {
        "rank": rank,
        "squared_frobenius_error": float(residual.square().sum()),
        "maximum_absolute_error": float(residual.abs().max()),
    }


def transform_blocks_orthogonal(
    blocks: Mapping[BlockKey, torch.Tensor],
    input_gauges: Mapping[int, torch.Tensor],
    output_gauges: Mapping[int, torch.Tensor],
) -> dict[BlockKey, torch.Tensor]:
    """Replay independent orthogonal coordinate gauges at every typed boundary."""
    transformed: dict[BlockKey, torch.Tensor] = {}
    for (output_site, input_site), block in blocks.items():
        if input_site not in input_gauges or output_site not in output_gauges:
            raise ValueError("gauge map is incomplete")
        q_in = input_gauges[input_site].to(dtype=block.dtype, device=block.device)
        q_out = output_gauges[output_site].to(dtype=block.dtype, device=block.device)
        if q_in.shape != (block.shape[1], block.shape[1]) or q_out.shape != (
            block.shape[0], block.shape[0]
        ):
            raise ValueError("gauge shape does not match response block")
        identity_in = torch.eye(q_in.shape[0], dtype=q_in.dtype, device=q_in.device)
        identity_out = torch.eye(q_out.shape[0], dtype=q_out.dtype, device=q_out.device)
        if not torch.allclose(q_in.T @ q_in, identity_in, atol=1e-10, rtol=1e-10) or not (
            torch.allclose(q_out.T @ q_out, identity_out, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("gauge replay requires orthogonal matrices")
        transformed[(output_site, input_site)] = q_out @ block @ q_in.T
    return transformed


def analyze_all_cuts(
    blocks: Mapping[BlockKey, torch.Tensor],
    input_dims: Mapping[int, int], output_dims: Mapping[int, int],
    cuts: tuple[int, ...], **analysis_kwargs: Any,
) -> dict[str, Any]:
    if not cuts or len(set(cuts)) != len(cuts):
        raise ValueError("cuts must be a nonempty unique tuple")
    result: dict[str, Any] = {}
    for cut in cuts:
        matrix, receipt = assemble_cut(blocks, input_dims, output_dims, cut)
        result[str(cut)] = {**receipt, **analyze_cut(matrix, **analysis_kwargs)}
    return result
