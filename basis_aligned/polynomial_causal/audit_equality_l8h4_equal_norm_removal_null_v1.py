#!/usr/bin/env python3
"""Correct the V1 copy-cell predicate without upgrading its real near-miss."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE / "EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_RESULT.json"
PREREG = HERE / "EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_PREREGISTRATION.md"
BINDING = HERE / "EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_BINDING.json"
RUNNER = HERE.parent / "bilinear_quotient/ops/run_equality_l8h4_equal_norm_removal_null_v1.py"
V3_RESULT = HERE / "EQUALITY_L8H4_REVERSIBLE_EDGE_CODE_OOD_V3_RESULT.json"
OUT = HERE / "EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_AUDIT.json"
EXPECTED = {
    "result": "1dd74039b03abd78a9e76aa4fd099095c6c615955b8e84bbc2438c7c28d4dda4",
    "preregistration": "4e807b9eb989c71bbc302e9a0eb21ecb60855466fe70e28d2b273a72a12f0100",
    "binding": "11e55e05d21f34922b112b5cfe5c2d686c73d3b521528bdc23c263f06d982b40",
    "runner": "d7560bc3cdf8c0b0c5ef4d61807efe75b7d997c9785f9f4ac25b204594644a57",
}
COPY_SUBCELLS = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = {"result": RESULT, "preregistration": PREREG, "binding": BINDING, "runner": RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise ValueError("V1 audit authority changed")
    result = json.loads(RESULT.read_text())
    parent = json.loads(V3_RESULT.read_text())
    if result["terminal"] != "valid_equality_l8h4_equal_norm_removal_null" \
            or not result["predictions"]["pred_a_instrument_and_equal_norm"]:
        raise ValueError("V1 is not an instrument-valid null")
    target = result["target"]
    controls = result["equal_norm_controls"]
    corrected_materiality = bool(target["copy_positive_damage_sum_nat"] > 0
        and all(target["copy_cell_damage_sum_nat"][cell] > 0 for cell in COPY_SUBCELLS)
        and min(target["half_damage_sum_nat"]) > 0)
    corrected = dict(result["predictions"])
    corrected["pred_b_target_removal_is_behaviorally_material"] = corrected_materiality

    noncopy_tokens = parent["reports"]["absent"]["all_noncopy"]["tokens"]
    target_noncopy_sum = target["all_noncopy_mean_absolute_nat"] * noncopy_tokens
    target_selectivity_ratio = target["copy_positive_damage_sum_nat"] / max(target_noncopy_sum, 1e-30)
    random_selectivity_ratios = [
        abs(damage) / max(collateral * noncopy_tokens, 1e-30)
        for damage, collateral in zip(controls["copy_positive_damage_sum_nat"],
                                      controls["all_noncopy_mean_absolute_nat"])
    ]
    ordered = sorted(random_selectivity_ratios)
    random_selectivity_median = ordered[(len(ordered) - 1) // 2]
    collateral_ratio = target["all_noncopy_mean_absolute_nat"] \
        / max(controls["all_noncopy_median_absolute_nat"], 1e-30)
    audit = {
        "schema": "equality_l8h4_equal_norm_removal_null_v1_audit",
        "terminal": "valid_equality_l8h4_equal_norm_near_miss",
        "status": "post_execution_predicate_correction_not_new_ood_evidence",
        "bound_files": observed,
        "original_predictions": result["predictions"],
        "corrected_predictions": corrected,
        "correction": {
            "pred_b_originally_iterated_v2_CELLS_including_all_noncopy": True,
            "registered_copy_subcells": list(COPY_SUBCELLS),
            "all_registered_copy_subcells_positive": corrected_materiality,
            "pred_d_remains_false": not corrected["pred_d_target_is_selective"],
            "target_collateral_to_random_median_ratio": collateral_ratio,
        },
        "posthoc_diagnostic_not_a_registered_predicate": {
            "noncopy_tokens": noncopy_tokens,
            "target_copy_to_noncopy_absolute_sum_ratio": target_selectivity_ratio,
            "random_copy_to_noncopy_absolute_sum_ratios": random_selectivity_ratios,
            "random_selectivity_ratio_median": random_selectivity_median,
            "target_to_random_median_selectivity_ratio": target_selectivity_ratio / max(random_selectivity_median, 1e-30),
        },
        "four_trait_status": "not_certified_because_registered_pred_d_failed",
        "finding": "The target removal is material and beats all 16 equal-norm directions by target damage, but its raw noncopy collateral is 13% above the random median. V1's pred_b FALSE was a coding error; the registered overall selectivity claim remains a genuine near-miss through pred_d.",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
