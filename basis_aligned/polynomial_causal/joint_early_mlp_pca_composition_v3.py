"""Pure contracts for the authoritative mixed early-MLP composition lattice."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping, Sequence

import torch


ARM_STATES: tuple[tuple[str, str, str], ...] = tuple(
    product(("N", "P", "E"), ("N", "P", "E"), ("N", "E"))
)
BASELINE_ARM = ("N", "N", "N")
RETENTION_FRACTION = 0.40


def arm_name(arm: tuple[str, str, str]) -> str:
    if arm not in ARM_STATES:
        raise ValueError(f"unregistered mixed-lattice arm: {arm}")
    return "".join(arm)


NO_FREE_RIDER: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    "p0_given_n1_m2n": (("P", "N", "N"), ("N", "N", "N")),
    "p0_given_n1_m2e": (("P", "N", "E"), ("N", "N", "E")),
    "p0_given_p1_m2n": (("P", "P", "N"), ("N", "P", "N")),
    "p0_given_p1_m2e": (("P", "P", "E"), ("N", "P", "E")),
    "p0_given_e1_m2n": (("P", "E", "N"), ("N", "E", "N")),
    "p0_given_e1_m2e": (("P", "E", "E"), ("N", "E", "E")),
    "p1_given_n0_m2n": (("N", "P", "N"), ("N", "N", "N")),
    "p1_given_n0_m2e": (("N", "P", "E"), ("N", "N", "E")),
    "p1_given_p0_m2n": (("P", "P", "N"), ("P", "N", "N")),
    "p1_given_p0_m2e": (("P", "P", "E"), ("P", "N", "E")),
    "p1_given_e0_m2n": (("E", "P", "N"), ("E", "N", "N")),
    "p1_given_e0_m2e": (("E", "P", "E"), ("E", "N", "E")),
}

# name -> (projected arm, shared background, exact arm)
SAME_BACKGROUND_RETENTION: dict[
    str, tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]
] = {
    "p0_given_n1_m2n": (("P", "N", "N"), ("N", "N", "N"), ("E", "N", "N")),
    "p0_given_n1_m2e": (("P", "N", "E"), ("N", "N", "E"), ("E", "N", "E")),
    "p0_given_p1_m2n": (("P", "P", "N"), ("N", "P", "N"), ("E", "P", "N")),
    "p0_given_p1_m2e": (("P", "P", "E"), ("N", "P", "E"), ("E", "P", "E")),
    "p0_given_e1_m2n": (("P", "E", "N"), ("N", "E", "N"), ("E", "E", "N")),
    "p0_given_e1_m2e": (("P", "E", "E"), ("N", "E", "E"), ("E", "E", "E")),
    "p1_given_n0_m2n": (("N", "P", "N"), ("N", "N", "N"), ("N", "E", "N")),
    "p1_given_n0_m2e": (("N", "P", "E"), ("N", "N", "E"), ("N", "E", "E")),
    "p1_given_p0_m2n": (("P", "P", "N"), ("P", "N", "N"), ("P", "E", "N")),
    "p1_given_p0_m2e": (("P", "P", "E"), ("P", "N", "E"), ("P", "E", "E")),
    "p1_given_e0_m2n": (("E", "P", "N"), ("E", "N", "N"), ("E", "E", "N")),
    "p1_given_e0_m2e": (("E", "P", "E"), ("E", "N", "E"), ("E", "E", "E")),
}


def _ci95(values: torch.Tensor) -> list[float]:
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]


def _summary(point: float, draws: torch.Tensor) -> dict[str, Any]:
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(draws.mean()),
        "ci95": _ci95(draws),
    }


def paired_document_cluster_lattice(
    row_ce_by_arm: Mapping[tuple[str, str, str], Sequence[float]],
    document_ids: Sequence[str],
    *,
    draws: int = 2000,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the complete lattice with shared document multiplicities."""

    if set(row_ce_by_arm) != set(ARM_STATES):
        missing = sorted(set(ARM_STATES).difference(row_ce_by_arm))
        extra = sorted(set(row_ce_by_arm).difference(ARM_STATES))
        raise ValueError(f"bootstrap requires complete 18-arm lattice; missing={missing} extra={extra}")
    if draws < 2:
        raise ValueError("bootstrap requires at least two draws")
    values = torch.tensor(
        [[float(value) for value in row_ce_by_arm[arm]] for arm in ARM_STATES],
        dtype=torch.float64,
    )
    if values.ndim != 2 or values.shape[1] <= 0 or not torch.isfinite(values).all():
        raise ValueError("arm row CE must be a finite nonempty paired matrix")
    if len(document_ids) != values.shape[1] or not all(
        isinstance(document, str) and document for document in document_ids
    ):
        raise ValueError("document IDs must align one-to-one with rows")
    unique_documents = list(dict.fromkeys(document_ids))
    if len(unique_documents) < 2:
        raise ValueError("document-cluster bootstrap requires at least two documents")

    arm_index = {arm: index for index, arm in enumerate(ARM_STATES)}
    baseline = values[arm_index[BASELINE_ARM]]
    gain_rows = baseline.unsqueeze(0) - values
    point_gain = gain_rows.mean(dim=1)
    document_index = {document: index for index, document in enumerate(unique_documents)}
    cluster = torch.tensor(
        [document_index[document] for document in document_ids], dtype=torch.long
    )
    document_sums = torch.zeros(
        len(ARM_STATES), len(unique_documents), dtype=torch.float64
    )
    document_sums.index_add_(1, cluster, gain_rows)
    document_row_counts = torch.bincount(
        cluster, minlength=len(unique_documents)
    ).to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    sampled_documents = torch.randint(
        len(unique_documents), (draws, len(unique_documents)), generator=generator
    )
    sampled_numerators = document_sums[:, sampled_documents].sum(dim=2).T
    sampled_denominators = document_row_counts[sampled_documents].sum(dim=1)
    boot_gain = sampled_numerators / sampled_denominators.unsqueeze(1)

    def point(arm: tuple[str, str, str]) -> torch.Tensor:
        return point_gain[arm_index[arm]]

    def boot(arm: tuple[str, str, str]) -> torch.Tensor:
        return boot_gain[:, arm_index[arm]]

    def difference(
        candidate: tuple[str, str, str], background: tuple[str, str, str]
    ) -> tuple[float, torch.Tensor]:
        return float(point(candidate) - point(background)), boot(candidate) - boot(background)

    no_free_rider: dict[str, Any] = {}
    for name, (candidate, background) in NO_FREE_RIDER.items():
        estimate, sampled = difference(candidate, background)
        no_free_rider[name] = _summary(estimate, sampled)

    exact_effects: dict[str, Any] = {}
    retention_margins: dict[str, Any] = {}
    for name, (projected_arm, background, exact_arm) in SAME_BACKGROUND_RETENTION.items():
        projected_point, projected_boot = difference(projected_arm, background)
        exact_point, exact_boot = difference(exact_arm, background)
        margin_point = projected_point - RETENTION_FRACTION * exact_point
        margin_boot = projected_boot - RETENTION_FRACTION * exact_boot
        exact_effects[name] = _summary(exact_point, exact_boot)
        retention_margins[name] = {
            **_summary(margin_point, margin_boot),
            "projected_effect_point_estimate": projected_point,
            "exact_effect_point_estimate": exact_point,
            "descriptive_ratio_if_identified": (
                projected_point / exact_point if exact_point > 0.0 else None
            ),
        }

    package_n_point = float(point(("P", "P", "N")) - torch.maximum(
        point(("P", "N", "N")), point(("N", "P", "N"))
    ))
    package_n_boot = boot(("P", "P", "N")) - torch.maximum(
        boot(("P", "N", "N")), boot(("N", "P", "N"))
    )
    e_constituents_point = torch.stack([
        point(("P", "N", "E")), point(("N", "P", "E")), point(("N", "N", "E"))
    ])
    e_constituents_boot = torch.stack([
        boot(("P", "N", "E")), boot(("N", "P", "E")), boot(("N", "N", "E"))
    ], dim=1)
    package_e_point = float(point(("P", "P", "E")) - e_constituents_point.max())
    package_e_boot = boot(("P", "P", "E")) - e_constituents_boot.max(dim=1).values
    downstream_point, downstream_boot = difference(("P", "P", "E"), ("P", "P", "N"))
    upstream_n_point = float(
        point(("P", "P", "N")) - RETENTION_FRACTION * point(("E", "E", "N"))
    )
    upstream_n_boot = (
        boot(("P", "P", "N")) - RETENTION_FRACTION * boot(("E", "E", "N"))
    )
    upstream_e_point = float(
        (point(("P", "P", "E")) - point(("N", "N", "E")))
        - RETENTION_FRACTION
        * (point(("E", "E", "E")) - point(("N", "N", "E")))
    )
    upstream_e_boot = (
        (boot(("P", "P", "E")) - boot(("N", "N", "E")))
        - RETENTION_FRACTION
        * (boot(("E", "E", "E")) - boot(("N", "N", "E")))
    )
    exact_upstream_n_point = float(point(("E", "E", "N")))
    exact_upstream_n_boot = boot(("E", "E", "N"))
    exact_upstream_e_point, exact_upstream_e_boot = difference(
        ("E", "E", "E"), ("N", "N", "E")
    )
    package_and_downstream = {
        "ppn_minus_best_constituent": _summary(package_n_point, package_n_boot),
        "ppe_minus_best_constituent": _summary(package_e_point, package_e_boot),
        "exact_mlp2_after_projected_upstream": _summary(downstream_point, downstream_boot),
        "exact_upstream_effect_mlp2n": _summary(
            exact_upstream_n_point, exact_upstream_n_boot
        ),
        "exact_upstream_effect_mlp2e": _summary(
            exact_upstream_e_point, exact_upstream_e_boot
        ),
        "projected_upstream_40pct_margin_mlp2n": _summary(upstream_n_point, upstream_n_boot),
        "projected_upstream_40pct_margin_mlp2e": _summary(upstream_e_point, upstream_e_boot),
    }
    return {
        "draws": draws,
        "seed": seed,
        "resampling": (
            "sample unique source documents with replacement; retain all chunks; "
            "divide by sampled row count; share multiplicities across all 18 arms"
        ),
        "row_count": values.shape[1],
        "unique_document_count": len(unique_documents),
        "cluster_size_range": [
            int(document_row_counts.min()), int(document_row_counts.max())
        ],
        "arm_gain": {
            arm_name(arm): _summary(float(point(arm)), boot(arm)) for arm in ARM_STATES
        },
        "no_free_rider": no_free_rider,
        "same_background_exact_effect": exact_effects,
        "same_background_40pct_margin": retention_margins,
        "package_and_downstream": package_and_downstream,
    }


def _positive_gate(discovery: Mapping[str, Any], heldout: Mapping[str, Any]) -> bool:
    values = (
        float(discovery["point_estimate"]),
        float(heldout["point_estimate"]),
        float(heldout["ci95"][0]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("registered gate inputs must be finite")
    return all(value > 0.0 for value in values)


def score_registered_predictions(
    analyses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the preregistered conjunctions without outcome-dependent selection."""

    if set(analyses) != {"discovery", "heldout"}:
        raise ValueError("registered scoring requires discovery and heldout analyses")
    discovery, heldout = analyses["discovery"], analyses["heldout"]
    no_free_rider = {
        name: _positive_gate(
            discovery["no_free_rider"][name], heldout["no_free_rider"][name]
        )
        for name in NO_FREE_RIDER
    }
    exact_effect = {
        name: _positive_gate(
            discovery["same_background_exact_effect"][name],
            heldout["same_background_exact_effect"][name],
        )
        for name in SAME_BACKGROUND_RETENTION
    }
    retention_margin = {
        name: _positive_gate(
            discovery["same_background_40pct_margin"][name],
            heldout["same_background_40pct_margin"][name],
        )
        for name in SAME_BACKGROUND_RETENTION
    }
    package_names = (
        "ppn_minus_best_constituent",
        "ppe_minus_best_constituent",
    )
    package = {
        name: _positive_gate(
            discovery["package_and_downstream"][name],
            heldout["package_and_downstream"][name],
        )
        for name in package_names
    }
    downstream = _positive_gate(
        discovery["package_and_downstream"]["exact_mlp2_after_projected_upstream"],
        heldout["package_and_downstream"]["exact_mlp2_after_projected_upstream"],
    )
    upstream_retention = {
        name: _positive_gate(
            discovery["package_and_downstream"][name],
            heldout["package_and_downstream"][name],
        )
        for name in (
            "projected_upstream_40pct_margin_mlp2n",
            "projected_upstream_40pct_margin_mlp2e",
        )
    }
    exact_upstream = {
        name: _positive_gate(
            discovery["package_and_downstream"][name],
            heldout["package_and_downstream"][name],
        )
        for name in (
            "exact_upstream_effect_mlp2n",
            "exact_upstream_effect_mlp2e",
        )
    }
    decisions = {
        "no_free_rider": no_free_rider,
        "same_background_exact_effect_positive": exact_effect,
        "same_background_40pct_margin": retention_margin,
        "package_beats_constituents": package,
        "exact_mlp2_after_projected_upstream": downstream,
        "exact_upstream_effect_positive": exact_upstream,
        "projected_upstream_40pct_margin": upstream_retention,
    }
    predictions = {
        "pred_a_all_twelve_no_free_rider_gates_pass": all(no_free_rider.values()),
        "pred_b_all_twelve_same_background_40pct_retention_gates_pass": (
            all(exact_effect.values()) and all(retention_margin.values())
        ),
        "pred_c_projected_package_beats_constituents_with_and_without_mlp2": (
            all(package.values())
        ),
        "pred_d_exact_mlp2_remains_conditionally_helpful_after_projected_upstream": (
            downstream
        ),
        "pred_e_projected_upstream_retains_40pct_with_and_without_fixed_exact_mlp2": (
            all(exact_upstream.values()) and all(upstream_retention.values())
        ),
    }
    predictions["pred_f_complete_modular_oracle_subspace_gate_passes"] = all(
        predictions.values()
    )
    return {"decisions": decisions, "registered_predictions": predictions}
