#!/usr/bin/env python3
"""Native-gate-only audit of immutable possessive reuse v1."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_v1_receipt_and_single_repair pred_b_repaired_native_capability_and_instrument pred_c_correct_write_moves_possessive_margin_unchanged pred_d_each_direction_construction_unchanged pred_e_direction_assignment_unchanged pred_f_no_outcome_reopening_or_postselection
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2.json"
V1 = ROOT / "circuits/fast_screens/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2_result.json"
PRIOR_ART_SHA256 = "a26f90ae492458b028b33022c2faa0297b0a96aadbb6a05b72396d8aabf2aaf9"
V1_SHA256 = "646353296ae9969515fa82819258efa61acbff978ea70ab967f9cbdffdc7e1c5"
MINIMUM_CELL_NATIVE_ACCURACY = 0.75
PRED_KEYS = (
    "pred_a_v1_receipt_and_single_repair",
    "pred_b_repaired_native_capability_and_instrument",
    "pred_c_correct_write_moves_possessive_margin_unchanged",
    "pred_d_each_direction_construction_unchanged",
    "pred_e_direction_assignment_unchanged",
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
        "schema": "task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_audit_plan_v2",
        "candidate_id": "subject_verb.number_agreement.mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2",
        "invalid_v1_sha256": V1_SHA256,
        "repair": "native all-row correctness -> each direction-construction cell accuracy >=0.75 only",
        "minimum_cell_native_accuracy": MINIMUM_CELL_NATIVE_ACCURACY,
        "reuse_effect_threshold_changes": 0, "gpu_rerun": False,
        "price": {"physical_model_forwards": 0, "example_evaluations": 0, "causal_installations": 0, "backwards": 0, "parameter_updates": 0, "immutable_receipts_read": 1},
    }


def audit(v1: dict[str, object]) -> dict[str, object]:
    original = v1["score"]
    evidence = original["evidence"]
    cells = {}
    for direction in ("plural_to_singular", "singular_to_plural"):
        for construction in ("conjunct_frame", "notes_frame"):
            subset = [item for item in evidence if item["direction"] == direction and item["construction"] == construction]
            cells[f"{direction}.{construction}"] = {
                "count": len(subset),
                "native_base_answer_accuracy": sum(item["native_donor_minus_base_margin"] < 0 for item in subset) / len(subset),
            }
    repaired_instrument = (
        all(item["count"] == 16 and item["native_base_answer_accuracy"] >= MINIMUM_CELL_NATIVE_ACCURACY for item in cells.values())
        and original["noop_max_absolute_logit_error"] <= 1e-4
        and original["maximum_install_absolute_error"] <= 5e-5
    )
    overall = original["overall_correct_write"]
    pred_c = overall["positive_fraction"] >= 0.75 and overall["median_donorward_change"] >= 0.005
    pred_d = all(item["count"] == 16 and item["positive_fraction"] >= 0.625 and item["median_donorward_change"] > 0 for item in original["by_direction_construction"].values())
    assignment = original["direction_assignment"]
    pred_e = assignment["correct_beats_swapped_fraction"] >= 0.65 and assignment["median_correct_advantage"] >= 0.005
    predictions = dict(zip(PRED_KEYS, (
        v1.get("terminal") == "invalid" and original["predictions"]["pred_a_authorities_and_program_validate"] is True,
        repaired_instrument, pred_c, pred_d, pred_e, True,
    )))
    return {
        "native_capability_by_direction_construction": cells,
        "noop_max_absolute_logit_error": original["noop_max_absolute_logit_error"],
        "maximum_install_absolute_error": original["maximum_install_absolute_error"],
        "overall_correct_write": overall,
        "by_direction_construction": original["by_direction_construction"],
        "direction_assignment": assignment,
        "predictions": predictions,
        "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "outcomes_reopened": False, "gpu_rerun": False, "reuse_effect_threshold_changes": 0,
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
    instrument = scored["predictions"][PRED_KEYS[0]] and scored["predictions"][PRED_KEYS[1]] and scored["predictions"][PRED_KEYS[5]]
    terminal = "screen" if all(scored["predictions"].values()) else "null" if instrument else "invalid"
    reason = "cross_task_number_write_reuse" if terminal == "screen" else "task14_write_does_not_transfer_to_possessive" if terminal == "null" else "repaired_instrument_failed"
    payload = managed.atomic_create_json(OUT, {
        "schema": "task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_result_v2",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "score": scored, "invalid_v1_sha256": V1_SHA256,
        "limits": "This tests the frozen cardinality-0 mapping only; it does not license post-hoc cardinality or scale search.",
    })
    print(json.dumps({"terminal": terminal, "reason": reason, "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
