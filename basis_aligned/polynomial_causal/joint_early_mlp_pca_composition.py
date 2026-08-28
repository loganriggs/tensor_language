"""Pure analysis contracts for joint projected early-MLP composition."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FRACTION_GATE = 0.40


def analyze_composition(
    exact: Mapping[str, Mapping[str, Any]],
    projected: Mapping[str, Mapping[str, Any]],
    *,
    upstream_heldout_ci95: Sequence[float],
    joint_heldout_ci95: Sequence[float],
) -> dict[str, Any]:
    """Compare same-run exact and PCA0/PCA1-plus-exact2 cubes."""
    splits = ("discovery", "heldout")
    upstream_name = "mlp0+mlp1"
    joint_name = "mlp0+mlp1+mlp2"
    rows: dict[str, Any] = {}
    for split in splits:
        exact_split = exact[split]
        projected_split = projected[split]
        exact_upstream = float(exact_split["gain_by_arm"][upstream_name])
        exact_joint = float(exact_split["gain_by_arm"][joint_name])
        projected_upstream = float(projected_split["gain_by_arm"][upstream_name])
        projected_joint = float(projected_split["gain_by_arm"][joint_name])
        conditional_mlp2 = float(
            projected_split["mlp2_conditional_marginal_after_mlp0_mlp1"]
        )
        values = (exact_upstream, exact_joint, projected_upstream,
                  projected_joint, conditional_mlp2)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("composition inputs must be finite")
        if exact_upstream <= 0.0 or exact_joint <= 0.0:
            raise ValueError("same-run exact denominators must be positive")
        rows[split] = {
            "exact_upstream_gain": exact_upstream,
            "projected_upstream_gain": projected_upstream,
            "projected_upstream_fraction_of_exact": projected_upstream / exact_upstream,
            "exact_joint_gain": exact_joint,
            "projected_joint_gain": projected_joint,
            "projected_joint_fraction_of_exact": projected_joint / exact_joint,
            "exact_mlp2_conditional_after_projected_upstream": conditional_mlp2,
        }

    upstream_ci = [float(value) for value in upstream_heldout_ci95]
    joint_ci = [float(value) for value in joint_heldout_ci95]
    if (len(upstream_ci) != 2 or len(joint_ci) != 2
            or any(not math.isfinite(value) for value in upstream_ci + joint_ci)):
        raise ValueError("composition bootstrap intervals must contain two finite values")
    decisions = {
        "projected_upstream_positive_both_splits": all(
            rows[split]["projected_upstream_gain"] > 0.0 for split in splits
        ),
        "projected_joint_positive_both_splits": all(
            rows[split]["projected_joint_gain"] > 0.0 for split in splits
        ),
        "projected_upstream_heldout_ci95_lower_gt_zero": upstream_ci[0] > 0.0,
        "projected_joint_heldout_ci95_lower_gt_zero": joint_ci[0] > 0.0,
        "projected_upstream_fraction_ge_0p40_both_splits": all(
            rows[split]["projected_upstream_fraction_of_exact"] >= FRACTION_GATE
            for split in splits
        ),
        "projected_joint_fraction_ge_0p40_both_splits": all(
            rows[split]["projected_joint_fraction_of_exact"] >= FRACTION_GATE
            for split in splits
        ),
        "exact_mlp2_conditional_after_projected_upstream_positive_both_splits": all(
            rows[split]["exact_mlp2_conditional_after_projected_upstream"] > 0.0
            for split in splits
        ),
    }
    decisions["complete_composition_gate_passes"] = all(decisions.values())
    predictions = {
        "pred_a_projected_upstream_retains_at_least_40pct_both_splits": (
            decisions["projected_upstream_fraction_ge_0p40_both_splits"]
        ),
        "pred_b_exact_mlp2_conditional_after_projected_upstream_positive_both_splits": (
            decisions[
                "exact_mlp2_conditional_after_projected_upstream_positive_both_splits"
            ]
        ),
        "pred_c_projected_joint_retains_at_least_40pct_both_splits": (
            decisions["projected_joint_fraction_ge_0p40_both_splits"]
        ),
        "pred_d_complete_composition_gate_passes": (
            decisions["complete_composition_gate_passes"]
        ),
    }
    return {
        "currency": "paired same-run CE gain; projected uses PCA0/PCA1 plus exact MLP2",
        "splits": rows,
        "upstream_heldout_bootstrap_ci95": upstream_ci,
        "joint_heldout_bootstrap_ci95": joint_ci,
        "fraction_gate": FRACTION_GATE,
        "decisions": decisions,
        "registered_predictions": predictions,
    }
