"""Context-normalized diagnostics for finite-horizon tangent response bundles.

For a context response ``H_c`` (Fisher probes by intervention directions), define

    D_c = H_c.T @ H_c / ||H_c||_F^2.

``D_c`` is a positive semidefinite, trace-one density on the registered intervention
coordinates.  It discards context-to-context response intensity while retaining the
causal direction geometry.  Averaging these densities therefore tests whether one
small, context-independent intervention dictionary transports across documents.  A
large gap between local and barycenter ranks is evidence for a rotating bundle, not a
certificate of a nonlinear state.

Only scalar summaries are returned.  Raw responses, densities, eigenvectors, and
projectors never escape this module.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _context_densities(
    responses: torch.Tensor, probes_per_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not torch.is_tensor(responses) or responses.ndim != 2
        or not responses.is_floating_point() or not bool(torch.isfinite(responses).all())
    ):
        raise ValueError("responses must be a finite floating matrix")
    if type(probes_per_context) is not int or probes_per_context <= 0 or (
        responses.shape[0] % probes_per_context
    ):
        raise ValueError("responses must partition into complete probe contexts")
    contexts = responses.double().reshape(
        -1, probes_per_context, responses.shape[1],
    )
    energies = contexts.square().sum(dim=(1, 2))
    if not bool((energies > 0).all()):
        raise ValueError("every context must have positive tangent energy")
    densities = torch.bmm(contexts.transpose(1, 2), contexts)
    densities = densities / energies[:, None, None]
    return densities, energies


def _rank_for_energy(eigenvalues: torch.Tensor, fraction: float) -> int:
    if eigenvalues.ndim != 1 or len(eigenvalues) == 0:
        raise ValueError("eigenvalue spectrum is malformed")
    total = eigenvalues.sum()
    if not bool(total > 0):
        raise ValueError("eigenvalue spectrum has zero energy")
    cumulative = torch.cumsum(eigenvalues, dim=0) / total
    target = torch.tensor(fraction, dtype=cumulative.dtype, device=cumulative.device)
    return int(torch.searchsorted(cumulative, target)) + 1


def analyze_context_normalized_bundle(
    primary_responses: torch.Tensor,
    replication_responses: torch.Tensor,
    *,
    probes_per_context: int,
    energy_fraction: float = 0.95,
    maximum_fractional_rank: float = 0.5,
    minimum_replication_capture: float = 0.90,
    maximum_capture_gap: float = 0.05,
    maximum_barycenter_distance: float = 0.20,
) -> dict[str, Any]:
    """Test a shared intervention dictionary after normalizing context intensity.

    The candidate rank is learned only from the primary barycenter.  Its projector is
    then evaluated on the replication barycenter.  This is a directional OOD test;
    replication data do not choose the primary candidate rank or basis.

    The returned ``bundle_rotation_gap`` is descriptive: a positive gap means local
    response operators need fewer dimensions than their pooled density barycenter.
    It can motivate a context-conditioned model, but does not by itself prove one.
    """
    constants = (
        energy_fraction, maximum_fractional_rank, minimum_replication_capture,
        maximum_capture_gap, maximum_barycenter_distance,
    )
    if (
        not 0 < energy_fraction <= 1
        or not 0 < maximum_fractional_rank < 1
        or not 0 < minimum_replication_capture <= 1
        or maximum_capture_gap < 0
        or maximum_barycenter_distance < 0
        or any(not math.isfinite(value) for value in constants)
    ):
        raise ValueError("bundle-analysis constants are malformed")
    if primary_responses.shape[1] != replication_responses.shape[1]:
        raise ValueError("splits must share registered intervention coordinates")

    primary, primary_energy = _context_densities(
        primary_responses, probes_per_context,
    )
    replication, replication_energy = _context_densities(
        replication_responses, probes_per_context,
    )
    width = primary.shape[-1]
    primary_mean = primary.mean(dim=0)
    replication_mean = replication.mean(dim=0)
    primary_values, primary_vectors = torch.linalg.eigh(primary_mean)
    replication_values = torch.linalg.eigvalsh(replication_mean)
    primary_values = primary_values.flip(0).clamp_min(0)
    primary_vectors = primary_vectors.flip(1)
    replication_values = replication_values.flip(0).clamp_min(0)

    selected_rank = _rank_for_energy(primary_values, energy_fraction)
    projector = (
        primary_vectors[:, :selected_rank] @ primary_vectors[:, :selected_rank].T
    )
    primary_capture = float(torch.trace(projector @ primary_mean))
    replication_capture = float(torch.trace(projector @ replication_mean))
    capture_gap = abs(primary_capture - replication_capture)
    mean_norm = (
        float(torch.linalg.matrix_norm(primary_mean))
        + float(torch.linalg.matrix_norm(replication_mean))
    ) / 2
    barycenter_distance = float(torch.linalg.matrix_norm(
        primary_mean - replication_mean,
    )) / mean_norm

    local_primary_values = torch.linalg.eigvalsh(primary).flip(1).clamp_min(0)
    local_replication_values = torch.linalg.eigvalsh(replication).flip(1).clamp_min(0)
    local_primary_ranks = torch.tensor([
        _rank_for_energy(values, energy_fraction) for values in local_primary_values
    ], dtype=torch.int64)
    local_replication_ranks = torch.tensor([
        _rank_for_energy(values, energy_fraction) for values in local_replication_values
    ], dtype=torch.int64)
    median_local_rank = float(torch.median(torch.cat((
        local_primary_ranks.double(), local_replication_ranks.double(),
    ))))
    rank_limit = math.floor(maximum_fractional_rank * width)
    gates = {
        "primary_rank_is_compressive": selected_rank <= rank_limit,
        "replication_energy_capture": replication_capture >= minimum_replication_capture,
        "capture_transport_gap": capture_gap <= maximum_capture_gap,
        "barycenter_stability": barycenter_distance <= maximum_barycenter_distance,
    }
    return {
        "status": (
            "shared_context_normalized_dictionary_candidate"
            if all(gates.values()) else "no_shared_context_normalized_dictionary"
        ),
        "contexts": {
            "primary": len(primary), "replication": len(replication),
        },
        "probes_per_context": probes_per_context,
        "intervention_width": width,
        "normalization": "D_c = H_c^T H_c / ||H_c||_F^2",
        "energy_fraction_rule": energy_fraction,
        "maximum_fractional_rank_rule": maximum_fractional_rank,
        "rank_limit": rank_limit,
        "primary_selected_rank": selected_rank,
        "replication_selected_rank": _rank_for_energy(
            replication_values, energy_fraction,
        ),
        "primary_barycenter_eigenvalues": [float(value) for value in primary_values],
        "replication_barycenter_eigenvalues": [
            float(value) for value in replication_values
        ],
        "primary_capture": primary_capture,
        "replication_capture_by_primary_dictionary": replication_capture,
        "absolute_capture_gap": capture_gap,
        "normalized_barycenter_frobenius_distance": barycenter_distance,
        "local_rank_summary": {
            "primary_minimum": int(local_primary_ranks.min()),
            "primary_median": float(torch.median(local_primary_ranks.double())),
            "primary_maximum": int(local_primary_ranks.max()),
            "replication_minimum": int(local_replication_ranks.min()),
            "replication_median": float(torch.median(local_replication_ranks.double())),
            "replication_maximum": int(local_replication_ranks.max()),
        },
        "bundle_rotation_gap": selected_rank - median_local_rank,
        "raw_trace_summary": {
            "primary_minimum": float(primary_energy.min()),
            "primary_median": float(torch.median(primary_energy)),
            "primary_maximum": float(primary_energy.max()),
            "replication_minimum": float(replication_energy.min()),
            "replication_median": float(torch.median(replication_energy)),
            "replication_maximum": float(replication_energy.max()),
        },
        "thresholds": {
            "minimum_replication_capture": minimum_replication_capture,
            "maximum_capture_gap": maximum_capture_gap,
            "maximum_barycenter_distance": maximum_barycenter_distance,
        },
        "gates": gates,
        "passes": all(gates.values()),
        "interpretation": (
            "tests one shared linear intervention dictionary after removing context "
            "intensity; a rotation gap is descriptive and is not a nonlinear-state proof"
        ),
        "raw_responses_returned": False,
        "density_matrices_returned": False,
        "projectors_returned": False,
    }


def analyze_site_bundles(
    primary_blocks: dict[tuple[int, int], torch.Tensor],
    replication_blocks: dict[tuple[int, int], torch.Tensor],
    *,
    target_site: int,
    source_sites: tuple[int, ...],
    probes_per_context: int,
    **analysis_kwargs: Any,
) -> dict[str, Any]:
    """Apply the held-out bundle test separately to registered source sites."""
    if (
        type(target_site) is not int or not source_sites
        or len(set(source_sites)) != len(source_sites)
        or any(type(site) is not int or site >= target_site for site in source_sites)
    ):
        raise ValueError("target or source-site registry is malformed")
    expected = {(target_site, site) for site in source_sites}
    if set(primary_blocks) != expected or set(replication_blocks) != expected:
        raise ValueError("every and only registered site blocks must be supplied")
    return {
        str(site): analyze_context_normalized_bundle(
            primary_blocks[(target_site, site)],
            replication_blocks[(target_site, site)],
            probes_per_context=probes_per_context,
            **analysis_kwargs,
        )
        for site in source_sites
    }
