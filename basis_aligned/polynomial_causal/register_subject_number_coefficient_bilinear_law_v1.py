#!/usr/bin/env python3
"""Register the held four-scalar subject-number coefficient law."""
from copy import deepcopy
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, design_key, execution_key, file_sha256, rebuild_registry_v2, validate_v2

TAG = "task.subject_verb.number_agreement"
OLD, NEW = "grammatical_subject_number.v27", "grammatical_subject_number.v28"
EVENT = "subject_number_coefficient_bilinear_law.v1.complete.held"
RESULT = BQ / "circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json"
AUDIT = HERE / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_AUDIT.json"
ARTIFACTS = {
    "subject_number_coefficient_law_builder_v1": ("basis_aligned/polynomial_causal/build_subject_number_coefficient_bilinear_law_v1.py", "builder"),
    "subject_number_coefficient_law_artifact_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json", "artifact"),
    "subject_number_coefficient_law_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_PREREGISTRATION.md", "preregistration"),
    "subject_number_coefficient_law_binding_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_BINDING.json", "binding"),
    "subject_number_coefficient_law_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_subject_number_coefficient_bilinear_law_v1.py", "runner"),
    "subject_number_coefficient_law_test_v1": ("basis_aligned/bilinear_quotient/ops/test_subject_number_coefficient_bilinear_law_v1.py", "test"),
    "subject_number_coefficient_law_result_v1": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json", "result"),
    "subject_number_coefficient_law_auditor_v1": ("basis_aligned/polynomial_causal/audit_subject_number_coefficient_bilinear_law_v1.py", "audit"),
    "subject_number_coefficient_law_audit_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_AUDIT.json", "audit"),
}


def art(path, kind): return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}
def metric(name, estimate, bar): return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def main():
    result, audit = json.loads(RESULT.read_text()), json.loads(AUDIT.read_text())
    assert result["terminal"] == "coefficient_bilinear_law_held" and all(result["score"]["predictions"].values()) and audit["all_checks_pass"]
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        assert record["claims"][-1]["claim_id"] == OLD
        for artifact_id, spec in ARTIFACTS.items(): record["artifacts"][artifact_id] = art(*spec)
        previous = record["claims"][-1]
        claim = deepcopy(previous)
        claim.update({"claim_id": NEW, "revision": 28, "supersedes": OLD, "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
                      "next_missing": "The held rank-one write now uses a four-scalar bilinear direction-by-cardinality coefficient law instead of a ten-entry table and preserves fresh causal effects at .99982 cosine versus rank one. Next derive direction/cardinality variables or the shared axis from native text state; do not refit behavior, add scalar-table variants, rescue the weak reverse composition direction, or use quantization."})
        record["claims"].append(claim)
        s = result["score"]
        event = {"event_id": EVENT, "claim_id": NEW, "test_type": "compiled_equivalence", "stage": "complete", "verdict": "held", "failure_kind": None,
                 "family_ids": [], "site_id": "L11H3.projected_write.direction_cardinality_rank1_program", "split_plan_id": "task14_fresh_matched_natural_split_v1", "evaluation_role": "fresh_coefficient_law_substitution",
                 "metrics": [metric("law_vs_rank1_cosine", s["law_vs_rank1"]["cosine"], ">=.995"), metric("law_vs_rank1_relative_l2", s["law_vs_rank1"]["relative_l2_error"], "<=.15"), metric("law_vs_rank1_sign", s["law_vs_rank1"]["sign_agreement"], ">=.95"), metric("law_vs_native_cosine", s["law_vs_native"]["cosine"], ">=.75"), metric("law_vs_native_relative_l2", s["law_vs_native"]["relative_l2_error"], "<=.75"), metric("law_vs_native_sign", s["law_vs_native"]["sign_agreement"], ">=.75"), metric("stored_coefficient_scalars", 4, "<10")],
                 "prereg_artifact_id": "subject_number_coefficient_law_prereg_v1", "result_artifact_id": "subject_number_coefficient_law_result_v1", "input_artifact_ids": list(ARTIFACTS), "seed": None, "checkpoint_sha256": result["checkpoint_weights_sha256"], "supersedes_event_id": None, "replicates_event_id": None, "sections": [],
                 "notes": "Four coefficients were frozen by OLS on the ten weights-only rank-one coefficients before new causal outcomes. Fresh intervention used 512 effects, zero gradients/updates/quantization, and exact closure."}
        event["design_key"] = design_key(record, event); event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event); validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__": main()
