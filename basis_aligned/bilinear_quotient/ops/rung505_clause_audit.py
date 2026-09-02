#!/usr/bin/env python3
"""CPU-only after-outcome clause audit for rung 505."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
RECEIPT = ROOT / "equality_score_gauged_downstream_program_rung505_results.json"
OUT = ROOT / "rung505_clause_audit_results.json"
EXPECTED_RECEIPT_SHA256 = "3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1"
SOURCES = ("N", "P", "Z7", "Z8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_pattern(vector):
    return vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0


def all_negative(vector):
    return all(value < 0 for value in vector)


def source_checks(data, source):
    analysis = data["analysis"]
    pooled = analysis["pooled"]
    metrics = analysis["source_metrics"][source]
    task = pooled["subset_vectors"][source]["7"]
    suppressor = pooled["subset_vectors"][source]["24"]
    union = pooled["subset_vectors"][source]["31"]
    interaction = pooled["interaction_vectors"][source]
    return {
        "task_vector_nat": task,
        "task_sign_pattern_holds": sign_pattern(task),
        "task_norm": metrics["task_to_correction"]["right_norm"],
        "task_norm_at_least_015": metrics["task_to_correction"]["right_norm"] >= .015,
        "task_correction_cosine": metrics["task_to_correction"]["cosine"],
        "task_correction_cosine_at_least_80": metrics["task_to_correction"]["cosine"] >= .80,
        "suppressor_vector_nat": suppressor,
        "suppressor_all_negative": all_negative(suppressor),
        "suppressor_correction_cosine": metrics["suppressor_to_correction"]["cosine"],
        "suppressor_correction_cosine_below_70": metrics["suppressor_to_correction"]["cosine"] < .70,
        "interaction_vector_nat": interaction,
        "interaction_norm": metrics["interaction_norm"],
        "interaction_norm_at_least_005": metrics["interaction_norm"] >= .005,
        "union_vector_nat": union,
        "union_correction_cosine": metrics["all_to_correction"]["cosine"],
        "union_correction_cosine_at_least_80": metrics["all_to_correction"]["cosine"] >= .80,
        "union_projection": metrics["all_to_correction"]["right_projection_on_left"],
        "union_projection_in_40_160": .40 <= metrics["all_to_correction"]["right_projection_on_left"] <= 1.60,
        "half_task_signs": [
            sign_pattern(half["subset_vectors"][source]["7"]) for half in analysis["halves"]
        ],
        "half_suppressor_signs": [
            all_negative(half["subset_vectors"][source]["24"]) for half in analysis["halves"]
        ],
    }


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    receipt_sha = sha256(RECEIPT)
    if EXPECTED_RECEIPT_SHA256 and receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise RuntimeError("rung 505 receipt hash changed")
    data = json.loads(RECEIPT.read_text())
    expected = {
        "pred_a_exact_live_intervention_instrument": True,
        "pred_b_score_actions_calibrated_in_patch_harness": True,
        "pred_c_fixed_program_transfers_code_to_natural": False,
        "pred_d_program_invariant_across_sign_gauge": False,
        "pred_e_correct_gauge_orientation_specific": True,
        "strong_null": False,
        "next_step": "abandon_fixed_five_site_program_as_code_specific",
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise RuntimeError("rung 505 verdict identity changed")
    checks = {source: source_checks(data, source) for source in SOURCES}
    comparisons = data["analysis"]["pooled_source_comparisons"]
    all_correct_orientation_margins = [
        row["margin"] for row in data["analysis"]["orientation_comparisons"]
    ]
    result = {
        "status": "complete",
        "analysis": "after_outcome_cpu_clause_audit_no_new_model_outcomes",
        "rung505_receipt_sha256": receipt_sha,
        "registered_verdict": expected,
        "source_clause_checks": checks,
        "cross_source_headlines": {
            "minimum_N_or_P_to_Z_task_cosine": min(
                comparisons[name]["T"]["cosine"]
                for name in ("N:Z7", "N:Z8", "P:Z7", "P:Z8")
            ),
            "minimum_N_or_P_to_Z_suppressor_cosine": min(
                comparisons[name]["G"]["cosine"]
                for name in ("N:Z7", "N:Z8", "P:Z7", "P:Z8")
            ),
            "minimum_N_or_P_to_Z_interaction_cosine": min(
                comparisons[name]["I"]["cosine"]
                for name in ("N:Z7", "N:Z8", "P:Z7", "P:Z8")
            ),
            "minimum_correct_vs_wrong_orientation_margin": min(all_correct_orientation_margins),
        },
        "interpretation": (
            "The fixed five sites are not one cross-corpus semantic program: the natural task-group "
            "vector is positive in every context, contradicting the code-derived near-/far+/one+/multiple- "
            "role, and its natural magnitude is below the registered floor.  Within natural text, however, "
            "the task, suppressor, and interaction vectors are highly stable across correctly gauged score "
            "sources and reverse under wrong sign.  Preserve this as a source-invariance fragment, not a "
            "cross-corpus identification or permission to weaken C."
        ),
        "new_model_outcomes_opened": False,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "registered_verdict": expected,
        "cross_source_headlines": result["cross_source_headlines"],
        "failed_task_signs": {source: checks[source]["task_sign_pattern_holds"] for source in SOURCES},
        "task_norms": {source: checks[source]["task_norm"] for source in SOURCES},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
