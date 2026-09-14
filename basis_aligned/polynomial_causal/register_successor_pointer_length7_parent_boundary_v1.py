#!/usr/bin/env python3
"""Register the successor typed-mediator length-seven boundary."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, design_key, execution_key, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.successor_pointer"
OLD_CLAIM = "successor_pointer_state.v6"
NEW_CLAIM = "successor_pointer_state.v7"
EVENT_ID = "successor_pointer_typed_group_length7.v1.complete.failed"
ARTIFACTS = {
    "successor_length7_builder_v1": ("basis_aligned/polynomial_causal/build_successor_fixed_pointer_length7_low_rank_holdout_v1_rows.py", "builder"),
    "successor_length7_rows_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_FIXED_POINTER_LENGTH7_LOW_RANK_HOLDOUT_V1_ROWS.json", "rows"),
    "successor_interaction_low_rank_prereg_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_PREREGISTRATION.md", "preregistration"),
    "successor_interaction_low_rank_binding_v3": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V3_BINDING.json", "binding"),
    "successor_interaction_low_rank_runner_v3": ("basis_aligned/bilinear_quotient/ops/run_successor_pointer_interaction_low_rank_v3.py", "runner"),
    "successor_interaction_low_rank_invalid_v3": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V3_RESULT.json", "invalid_result"),
    "successor_length7_parent_audit_v1": ("basis_aligned/polynomial_causal/SUCCESSOR_POINTER_LENGTH7_PARENT_BOUNDARY_V1_AUDIT.json", "audit"),
    "successor_length7_parent_auditor_v1": ("basis_aligned/polynomial_causal/audit_successor_pointer_length7_parent_boundary_v1.py", "auditor"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def bind(record, event):
    event["design_key"] = design_key(record, event); event["execution_key"] = execution_key(record, event); return event


def main():
    audit = json.loads((HERE / "SUCCESSOR_POINTER_LENGTH7_PARENT_BOUNDARY_V1_AUDIT.json").read_text())
    assert audit["terminal"] == "length7_parent_mediator_boundary" and all(audit["checks"].values())
    path = circuit_path(TAG); existing = json.loads(path.read_text())
    if any(e["event_id"] == EVENT_ID for e in existing["evidence_events"]):
        validate_v2(existing); rebuild_registry_v2(); print(json.dumps({"status": "already_registered", "claim_id": NEW_CLAIM}, indent=2)); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value: raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        previous = next(c for c in record["claims"] if c["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({"claim_id": NEW_CLAIM, "revision": 7, "supersedes": OLD_CLAIM,
                      "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
                      "next_missing": (
                          "The negative attention/MLP8-12 interaction is real at length six, but the parent native-oracle mediator is not stable at length seven: month projection falls to .466 with every exactness, capability, target, cosine, and control check passing. The low-rank interaction experiment is therefore unopened. Do not tune the .50 bar, rank, group, or length rows. Reopen only with an independently derived length-conditioned mediator/generator; otherwise move to another circuit."
                      )})
        for site in claim["candidate_sites"]:
            if site["site_id"] == "prefix_conditioned_baseline_branch": site["ceiling_event_ids"] = [*site["ceiling_event_ids"], EVENT_ID]
        record["claims"].append(claim)
        m = audit["metrics"]
        event = {"event_id": EVENT_ID, "claim_id": NEW_CLAIM, "test_type": "ood", "stage": "complete", "verdict": "failed", "failure_kind": "scientific_null",
                 "family_ids": ["prefix_change_final_pointer_preserved"], "site_id": "prefix_conditioned_baseline_branch", "split_plan_id": None,
                 "evaluation_role": "prospective_sequence_length_transfer",
                 "metrics": [
                     {"name": "length7_digit_joint_projection", "estimate": m["digit_joint_projection"], "ci95": None, "bar": ">=0.50"},
                     {"name": "length7_month_joint_projection", "estimate": m["month_joint_projection"], "ci95": None, "bar": ">=0.50"},
                     {"name": "length7_month_joint_cosine", "estimate": m["month_joint_cosine"], "ci95": None, "bar": ">=0.70"},
                     {"name": "length7_month_max_control_fraction", "estimate": max(m["month_late_forward_control_fraction"], m["month_early_backward_control_fraction"]), "ci95": None, "bar": "<=0.50"}],
                 "prereg_artifact_id": "successor_interaction_low_rank_prereg_v1", "result_artifact_id": "successor_length7_parent_audit_v1",
                 "input_artifact_ids": list(ARTIFACTS), "seed": None, "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
                 "sections": ["basis_aligned/polynomial_causal/SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_PREREGISTRATION.md"],
                 "notes": "V3 low-rank terminal is invalid, so no rank result is registered. The independent CPU audit proves the sole parent-gate failure is length-seven month projection; all exactness and capability checks pass."}
        record["evidence_events"].append(bind(record, event)); validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); final = json.loads(path.read_text()); validate_v2(final)
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID}, indent=2))


if __name__ == "__main__": main()
