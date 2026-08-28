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


def _physical_frame(
    responses: torch.Tensor, directions: torch.Tensor, rank: int,
) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(responses.double(), full_matrices=False)
    frame = directions.double().T @ vh[:rank].T
    q, r = torch.linalg.qr(frame, mode="reduced")
    if int((torch.abs(torch.diagonal(r)) > 1e-12).sum()) != rank:
        raise ValueError("direction map collapses a registered response frame")
    return q


def _frame_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("physical frames must have equal shape")
    rank = left.shape[1]
    overlap = float((left.T @ right).square().sum())
    return math.sqrt(max(0.0, rank - overlap) / rank)


def analyze_repeated_probe_physical_bundle(
    first_responses: torch.Tensor,
    second_responses: torch.Tensor,
    directions: torch.Tensor,
    *,
    probes_per_half: int,
    fixed_ranks: tuple[int, ...] = (8, 16, 24),
    energy_fraction: float = 0.95,
    gap_ratio: float = 2.0,
    local_rank_limit: int = 16,
    maximum_local_rank_difference: int = 2,
    maximum_same_context_distance: float = 0.15,
    minimum_context_fraction: float = 0.75,
    minimum_bundle_distance_lcb: float = 0.05,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    """Separate same-context probe noise from cross-context physical variation.

    Both response matrices must contain the same ordered contexts but independent
    Fisher-probe halves.  Right response frames are mapped through ``directions`` into
    physical residual-write space before distances are computed.  The document-level
    paired bootstrap compares cross-context and same-context distances at each fixed
    rank.  One context per document is an external preregistration requirement.
    """
    if (
        not torch.is_tensor(first_responses) or not torch.is_tensor(second_responses)
        or first_responses.ndim != 2 or second_responses.ndim != 2
        or first_responses.shape != second_responses.shape
        or not first_responses.is_floating_point()
        or not second_responses.is_floating_point()
        or not bool(torch.isfinite(first_responses).all())
        or not bool(torch.isfinite(second_responses).all())
    ):
        raise ValueError("probe halves must be equal finite floating matrices")
    if (
        not torch.is_tensor(directions) or directions.ndim != 2
        or directions.shape[0] != first_responses.shape[1]
        or directions.shape[1] < directions.shape[0]
        or not directions.is_floating_point() or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("physical direction map is malformed")
    if (
        type(probes_per_half) is not int or probes_per_half <= 0
        or first_responses.shape[0] % probes_per_half
        or first_responses.shape[0] // probes_per_half < 3
        or not fixed_ranks or len(set(fixed_ranks)) != len(fixed_ranks)
        or any(type(rank) is not int or rank <= 0 or rank > min(
            probes_per_half, first_responses.shape[1],
        ) for rank in fixed_ranks)
        or not 0 < energy_fraction <= 1 or gap_ratio <= 1
        or type(local_rank_limit) is not int or local_rank_limit <= 0
        or type(maximum_local_rank_difference) is not int
        or maximum_local_rank_difference < 0
        or not 0 <= maximum_same_context_distance <= 1
        or not 0 < minimum_context_fraction <= 1
        or not 0 <= minimum_bundle_distance_lcb <= 1
        or type(bootstrap_repetitions) is not int or bootstrap_repetitions < 100
        or type(bootstrap_seed) is not int or bootstrap_seed < 0
    ):
        raise ValueError("repeated-probe analysis constants are malformed")
    if int(torch.linalg.matrix_rank(directions.double())) != directions.shape[0]:
        raise ValueError("physical direction map must have full row rank")

    contexts = first_responses.shape[0] // probes_per_half
    first = first_responses.double().reshape(
        contexts, probes_per_half, first_responses.shape[1],
    )
    second = second_responses.double().reshape_as(first)
    if not bool((first.square().sum(dim=(1, 2)) > 0).all()) or not bool(
        (second.square().sum(dim=(1, 2)) > 0).all()
    ):
        raise ValueError("every probe half must have positive tangent energy")

    spectra_first = torch.linalg.svdvals(first)
    spectra_second = torch.linalg.svdvals(second)

    def spectrum_report(values: torch.Tensor) -> tuple[int, int, float | None]:
        support = int((values > 1e-12 * values[0]).sum())
        energy = values.square()
        rank95 = _rank_for_energy(energy, energy_fraction)
        gap = (
            float(values[rank95 - 1] / values[rank95])
            if rank95 < support and values[rank95] > 0 else None
        )
        selected = rank95 if rank95 == support or (
            gap is not None and gap >= gap_ratio
        ) else 0
        return support, rank95, float(selected)

    spectrum_rows = [
        (spectrum_report(a), spectrum_report(b))
        for a, b in zip(spectra_first, spectra_second, strict=True)
    ]
    probe_limited = [
        a[0] >= min(24, probes_per_half, first.shape[2])
        and b[0] >= min(24, probes_per_half, first.shape[2])
        and a[1] > local_rank_limit and b[1] > local_rank_limit
        for a, b in spectrum_rows
    ]

    fixed: dict[str, Any] = {}
    frame_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    for rank in fixed_ranks:
        for context in range(contexts):
            frame_cache[(0, context, rank)] = _physical_frame(
                first[context], directions, rank,
            )
            frame_cache[(1, context, rank)] = _physical_frame(
                second[context], directions, rank,
            )
        same = torch.tensor([
            _frame_distance(
                frame_cache[(0, context, rank)], frame_cache[(1, context, rank)],
            ) for context in range(contexts)
        ], dtype=torch.float64)
        cross = torch.full((contexts, contexts), float("nan"), dtype=torch.float64)
        for left in range(contexts):
            for right in range(left + 1, contexts):
                value = 0.5 * (
                    _frame_distance(
                        frame_cache[(0, left, rank)], frame_cache[(1, right, rank)],
                    ) + _frame_distance(
                        frame_cache[(1, left, rank)], frame_cache[(0, right, rank)],
                    )
                )
                cross[left, right] = cross[right, left] = value
        generator = torch.Generator(device="cpu").manual_seed(
            bootstrap_seed + 1000003 * rank,
        )
        differences = []
        attempts = 0
        while len(differences) < bootstrap_repetitions and attempts < (
            10 * bootstrap_repetitions
        ):
            attempts += 1
            sample = torch.randint(contexts, (contexts,), generator=generator)
            same_mean = float(same[sample].mean())
            pairs = [
                float(cross[int(sample[i]), int(sample[j])])
                for i in range(contexts) for j in range(i + 1, contexts)
                if int(sample[i]) != int(sample[j])
            ]
            if pairs:
                differences.append(sum(pairs) / len(pairs) - same_mean)
        if len(differences) != bootstrap_repetitions:
            raise RuntimeError("document bootstrap produced incomplete contrasts")
        difference_tensor = torch.tensor(differences, dtype=torch.float64)
        fixed[str(rank)] = {
            "same_context_mean_distance": float(same.mean()),
            "same_context_median_distance": float(torch.median(same)),
            "cross_context_mean_distance": float(cross[torch.isfinite(cross)].mean()),
            "cross_minus_same_mean": float(cross[torch.isfinite(cross)].mean() - same.mean()),
            "cross_minus_same_bootstrap_lcb_95": float(torch.quantile(
                difference_tensor, 0.025,
            )),
            "bootstrap_repetitions": bootstrap_repetitions,
        }

    local_stable = []
    for context, (a, b) in enumerate(spectrum_rows):
        selected_a, selected_b = int(a[2]), int(b[2])
        if not selected_a or not selected_b:
            local_stable.append(False)
            continue
        comparison_rank = max(selected_a, selected_b)
        same_distance = _frame_distance(
            _physical_frame(first[context], directions, comparison_rank),
            _physical_frame(second[context], directions, comparison_rank),
        )
        local_stable.append(bool(
            selected_a <= local_rank_limit and selected_b <= local_rank_limit
            and abs(selected_a - selected_b) <= maximum_local_rank_difference
            and same_distance <= maximum_same_context_distance
        ))

    probe_limited_fraction = sum(probe_limited) / contexts
    local_stable_fraction = sum(local_stable) / contexts
    comparison_rank = str(max(rank for rank in fixed_ranks if rank <= local_rank_limit))
    response_bundle = bool(
        local_stable_fraction >= minimum_context_fraction
        and fixed[comparison_rank]["cross_minus_same_bootstrap_lcb_95"]
        >= minimum_bundle_distance_lcb
    )
    return {
        "status": (
            "probe_limited_high_rank" if probe_limited_fraction >= minimum_context_fraction
            else "stable_context_varying_response_bundle" if response_bundle
            else "no_admitted_local_bundle"
        ),
        "contexts": contexts,
        "probes_per_half": probes_per_half,
        "physical_write_width": directions.shape[1],
        "coefficient_width": directions.shape[0],
        "fixed_rank_physical_projectors": fixed,
        "probe_limited_high_rank_fraction": probe_limited_fraction,
        "stable_local_low_rank_fraction": local_stable_fraction,
        "response_bundle_gate": response_bundle,
        "thresholds": {
            "energy_fraction": energy_fraction,
            "gap_ratio": gap_ratio,
            "local_rank_limit": local_rank_limit,
            "maximum_local_rank_difference": maximum_local_rank_difference,
            "maximum_same_context_distance": maximum_same_context_distance,
            "minimum_context_fraction": minimum_context_fraction,
            "minimum_bundle_distance_lcb": minimum_bundle_distance_lcb,
        },
        "physical_mapping": "U_cr = orth(direction_matrix^T V_cr)",
        "one_context_per_document_required": True,
        "raw_responses_returned": False,
        "physical_frames_returned": False,
        "projectors_returned": False,
        "interpretation": (
            "identifies response geometry only; H_c = D_c E_c does not identify an "
            "encoder gauge without an intermediate-state or composition experiment"
        ),
    }
