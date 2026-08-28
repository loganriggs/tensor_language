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


def compare_split_cuts(
    primary_blocks: Mapping[BlockKey, torch.Tensor],
    replication_blocks: Mapping[BlockKey, torch.Tensor],
    input_dims: Mapping[int, int], output_dims: Mapping[int, int],
    cuts: tuple[int, ...], *, energy_fraction: float = 0.95,
    gap_ratio: float = 2.0, maximum_rank_difference: int = 2,
    maximum_spectrum_l1: float = 0.10,
    maximum_projector_distance: float = 0.15,
) -> dict[str, Any]:
    """Compare independently measured cut operators in their common input gauge.

    Context rows differ between splits, so left singular vectors are incomparable.
    The right projectors live in the shared registered intervention coordinates and
    are the correct split-stability object.
    """
    if not cuts or len(set(cuts)) != len(cuts) or maximum_rank_difference < 0 or (
        maximum_spectrum_l1 < 0 or maximum_projector_distance < 0
    ):
        raise ValueError("split-cut comparison constants are malformed")
    reports: dict[str, Any] = {}
    for cut in cuts:
        primary, _ = assemble_cut(primary_blocks, input_dims, output_dims, cut)
        replication, _ = assemble_cut(replication_blocks, input_dims, output_dims, cut)
        if primary.shape[1] != replication.shape[1]:
            raise ValueError("split cut operators do not share intervention coordinates")
        singular_primary = torch.linalg.svdvals(primary)
        singular_replication = torch.linalg.svdvals(replication)
        energy_primary = singular_primary.square()
        energy_replication = singular_replication.square()
        trace_primary = float(energy_primary.sum())
        trace_replication = float(energy_replication.sum())
        mean_trace = (trace_primary + trace_replication) / 2
        relative_trace = (
            abs(trace_primary - trace_replication) / mean_trace
            if mean_trace > 0 else math.inf
        )
        if trace_primary <= 0 or trace_replication <= 0:
            spectrum_l1 = math.inf
        else:
            size = max(len(energy_primary), len(energy_replication))
            left = torch.zeros(size, dtype=torch.float64)
            right = torch.zeros(size, dtype=torch.float64)
            left[:len(energy_primary)] = energy_primary / trace_primary
            right[:len(energy_replication)] = energy_replication / trace_replication
            spectrum_l1 = float(torch.sum(torch.abs(left - right)))
        primary_analysis = analyze_cut(
            primary, energy_fraction=energy_fraction, gap_ratio=gap_ratio,
        )
        replication_analysis = analyze_cut(
            replication, energy_fraction=energy_fraction, gap_ratio=gap_ratio,
        )
        combined = torch.cat((primary, replication), dim=0)
        combined_analysis = analyze_cut(
            combined, energy_fraction=energy_fraction, gap_ratio=gap_ratio,
        )
        selected_primary = primary_analysis["selected_rank"]
        selected_replication = replication_analysis["selected_rank"]
        comparison_rank = combined_analysis["selected_rank"]
        ranks_exist = all(value is not None for value in (
            selected_primary, selected_replication, comparison_rank,
        ))
        rank_difference = (
            abs(int(selected_primary) - int(selected_replication))
            if ranks_exist else None
        )
        projector_distance = None
        if ranks_exist:
            rank = int(comparison_rank)
            _, _, vh_primary = torch.linalg.svd(primary, full_matrices=False)
            _, _, vh_replication = torch.linalg.svd(replication, full_matrices=False)
            projector_primary = vh_primary[:rank].T @ vh_primary[:rank]
            projector_replication = vh_replication[:rank].T @ vh_replication[:rank]
            projector_distance = float(torch.linalg.matrix_norm(
                projector_primary - projector_replication,
            ) / math.sqrt(2 * rank))
        gates = {
            "positive_stable_trace": bool(
                math.isfinite(relative_trace) and relative_trace <= 0.05
            ),
            "normalized_spectrum_l1": bool(
                math.isfinite(spectrum_l1) and spectrum_l1 <= maximum_spectrum_l1
            ),
            "selected_ranks_exist_and_agree": bool(
                ranks_exist and rank_difference is not None
                and rank_difference <= maximum_rank_difference
            ),
            "right_projector_stability": bool(
                projector_distance is not None
                and projector_distance <= maximum_projector_distance
            ),
        }
        reports[str(cut)] = {
            "primary_selected_rank": selected_primary,
            "replication_selected_rank": selected_replication,
            "combined_selected_rank": comparison_rank,
            "rank_difference": rank_difference,
            "relative_trace_difference": relative_trace,
            "normalized_squared_spectrum_l1": spectrum_l1,
            "normalized_right_projector_chordal_distance": projector_distance,
            "gates": gates,
            "passes": all(gates.values()),
        }
    return reports


def analyze_contextwise_cuts(
    blocks: Mapping[BlockKey, torch.Tensor], input_dims: Mapping[int, int],
    output_dims: Mapping[int, int], cuts: tuple[int, ...], *,
    probes_per_context: int, support_rtol: float = 1e-12,
) -> dict[str, Any]:
    """Report ranks before contexts are stacked into one shared-interface problem.

    A per-context rank lower-bounds a general tangent state on that context.  The rank
    after vertical context stacking instead lower-bounds only a context-independent
    linear encoder shared across contexts; the two currencies must not be conflated.
    """
    if type(probes_per_context) is not int or probes_per_context <= 0 or (
        support_rtol <= 0
    ):
        raise ValueError("contextwise cut constants are malformed")
    reports: dict[str, Any] = {}
    for cut in cuts:
        matrix, _ = assemble_cut(blocks, input_dims, output_dims, cut)
        if matrix.shape[0] % probes_per_context:
            raise ValueError("cut rows do not partition into complete probe contexts")
        contexts = matrix.reshape(-1, probes_per_context, matrix.shape[1])
        singular = torch.linalg.svdvals(contexts)
        leading = singular[:, :1]
        thresholds = support_rtol * torch.maximum(
            leading, torch.ones_like(leading),
        )
        ranks = (singular > thresholds).sum(dim=1).to(torch.int64)
        reports[str(cut)] = {
            "contexts": len(contexts),
            "probes_per_context": probes_per_context,
            "minimum_rank": int(ranks.min()),
            "median_rank": float(torch.median(ranks.double())),
            "maximum_rank": int(ranks.max()),
            "ranks": ranks.tolist(),
            "interpretation": (
                "per-context lower bound; stacked rank is only a shared-linear-encoder bound"
            ),
        }
    return reports
