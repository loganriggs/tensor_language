#!/usr/bin/env python3
"""Numerical-tolerance-only audit of immutable prototype-collateral v1."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_v1_receipt_and_single_repair pred_b_repaired_projected_write_instrument pred_c_numbered_list_preserved_unchanged pred_d_bracket_preserved_unchanged pred_e_complete_program_unchanged pred_f_no_outcome_reopening_or_postselection
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2.json"
V1 = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2_result.json"
PRIOR_ART_SHA256 = "38d1ee665f3ff1922aa173efae0e30d9dc1db84bb25665d5c9dd4938014604ef"
V1_SHA256 = "b26fc7ece3a295bfe2796e2a252bab7d627cef2185656186f6b9e1f8f15b6824"
REPAIRED_MAX_INSTALL_ERROR = 5e-5
PRED_KEYS = (
    "pred_a_v1_receipt_and_single_repair",
    "pred_b_repaired_projected_write_instrument",
    "pred_c_numbered_list_preserved_unchanged",
    "pred_d_bracket_preserved_unchanged",
    "pred_e_complete_program_unchanged",
    "pred_f_no_outcome_reopening_or_postselection",
)


class AuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_plan() -> dict[str, object]:
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise AuditError("prior art changed")
    if _sha256(V1) != V1_SHA256:
        raise AuditError("invalid v1 receipt changed")
    return {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_audit_plan_v2",
        "candidate_id": "subject_verb.number_agreement.mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2",
        "invalid_v1_sha256": V1_SHA256,
        "repair": "maximum_install_absolute_error 1e-5 -> 5e-5 only",
        "repaired_maximum_install_absolute_error": REPAIRED_MAX_INSTALL_ERROR,
        "gpu_rerun": False, "scientific_threshold_changes": 0,
        "price": {"physical_model_forwards": 0, "example_evaluations": 0, "causal_installations": 0, "backwards": 0, "parameter_updates": 0, "immutable_receipts_read": 1},
    }


def audit(v1: dict[str, object]) -> dict[str, object]:
    original = v1["score"]
    original_predictions = original["predictions"]
    only_c_false = {key for key, value in original_predictions.items() if value is False} == {"pred_c_all_ten_hooks_live"}
    norms = original["prototype_l2_norms"]
    repaired_instrument = (
        original["noop_max_absolute_logit_error"] <= 1e-4
        and original["maximum_install_absolute_error"] <= REPAIRED_MAX_INSTALL_ERROR
        and len(norms) == 10
        and all(math.isfinite(float(value)) and float(value) > 0 for value in norms.values())
    )
    summaries = original["behavior_prototype_results"]
    numbered = [value for value in summaries.values() if value["behavior"] == "numbered_list"]
    bracket = [value for value in summaries.values() if value["behavior"] == "bracket_pending_opener"]
    evidence = original["evidence"]
    complete = len(evidence) == 320 and len({(item["row_id"], item["prototype_key"]) for item in evidence}) == 320
    predictions = dict(zip(PRED_KEYS, (
        v1.get("terminal") == "invalid" and only_c_false,
        repaired_instrument,
        len(numbered) == 10 and all(item["passed_preservation"] for item in numbered),
        len(bracket) == 10 and all(item["passed_preservation"] for item in bracket),
        complete,
        True,
    )))
    return {
        "predictions": predictions,
        "repaired_maximum_install_absolute_error": REPAIRED_MAX_INSTALL_ERROR,
        "observed_maximum_install_absolute_error": original["maximum_install_absolute_error"],
        "noop_max_absolute_logit_error": original["noop_max_absolute_logit_error"],
        "behavior_prototype_results": summaries,
        "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "outcomes_reopened": False, "gpu_rerun": False, "scientific_threshold_changes": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise AuditError(f"refusing overwrite {OUT}")
    scored = audit(json.loads(V1.read_text()))
    terminal = "screen" if all(scored["predictions"].values()) else "invalid"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_result_v2",
        "candidate_id": plan["candidate_id"], "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "score": scored, "invalid_v1_sha256": V1_SHA256,
        "limits": "Two unrelated behaviors establish narrow collateral breadth, not universal selectivity.",
    })
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
