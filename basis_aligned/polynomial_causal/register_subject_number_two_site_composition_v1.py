#!/usr/bin/env python3
"""Register the subject-number two-site single-arm boundary."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, design_key, execution_key, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.subject_verb_number_agreement"
OLD = "grammatical_subject_number.v25"
NEW = "grammatical_subject_number.v26"
EVENT = "subject_number_two_site_composition.v1.complete.failed"
RESULT = HERE / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_RESULT.json"
ARTIFACTS = {
    "subject_number_two_site_builder_v1": ("basis_aligned/polynomial_causal/build_subject_number_two_site_composition_v1_rows.py", "builder"),
    "subject_number_two_site_test_v1": ("basis_aligned/polynomial_causal/test_build_subject_number_two_site_composition_v1_rows.py", "test"),
    "subject_number_two_site_rows_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_ROWS.json", "rows"),
    "subject_number_two_site_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_PREREGISTRATION.md", "preregistration"),
    "subject_number_two_site_binding_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_BINDING.json", "binding"),
    "subject_number_two_site_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_subject_number_two_site_composition_v1.py", "runner"),
    "subject_number_two_site_invalid_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_RESULT.json", "invalid_result"),
}


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main():
    result = json.loads(RESULT.read_text())
    assert result["terminal"] == "invalid_or_dead_single_site"
    assert result["predictions"] == {
        "pred_a_exact_instrument_and_capability": True,
        "pred_b_single_site_writes_live_and_selective": False,
        "pred_c_two_site_additive_composition": False,
        "pred_d_two_site_interaction_live": False,
    }
    assert result["price"]["observed_forwards"] == 5 and result["price"]["observed_sequences"] == 80
    assert all(report["passes"] for report in result["composition"].values())

    path = circuit_path(TAG)
    current = json.loads(path.read_text())
    if any(event["event_id"] == EVENT for event in current["evidence_events"]):
        validate_v2(current); rebuild_registry_v2(); print("already registered"); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        for artifact_id, spec in ARTIFACTS.items():
            record["artifacts"][artifact_id] = artifact(*spec)
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW,
            "revision": 26,
            "supersedes": OLD,
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
            "next_missing": "The fresh two-site composition authority is invalid because the immutable cardinality-four plural-to-singular write at site 1 has RMS .00800, below the .01 single-arm liveliness bar. Exactness and all native capability cells pass. Diagnostic two-site additivity is nearly exact (cosine 1.0, relative L2 7.6e-5, norm ratio .999996), with zero anticausal site2-to-site1 effect and unrelated controls .0227 of joint number RMS, but no composition verdict is registered after the parent failure. Do not lower the bar, scale the write, change cardinality, or reuse these opened rows. Move circuits unless an independently derived stronger two-site authority is available.",
        })
        claim["counterfactual_families"].append({
            "family_id": "fresh_two_clause_subject_number_composition_v1",
            "role": "interchange",
            "changes": ["rank-one number write at first subject", "rank-one number write at second subject"],
            "holds_fixed": ["two-clause tokens", "native downstream background", "frozen axis", "cardinality-four direction scalars"],
            "builder_artifact_id": "subject_number_two_site_builder_v1",
            "control_ids": ["zero write replay", "site2 anticausal check", "can-versus-will margin"],
            "split_plan_id": None,
            "status": "validated",
        })
        record["claims"].append(claim)
        weak = result["single_site"]["site1|plural_to_singular"]
        event = {
            "event_id": EVENT,
            "claim_id": NEW,
            "test_type": "composition",
            "stage": "complete",
            "verdict": "failed",
            "failure_kind": "scientific_null",
            "family_ids": ["fresh_two_clause_subject_number_composition_v1"],
            "site_id": "L11H3.projected_write.direction_cardinality_rank1_program",
            "split_plan_id": None,
            "evaluation_role": "prospective_two_site_composition_parent_gate",
            "metrics": [
                {"name": "site1_plural_to_singular_effect_rms", "estimate": weak["effect_rms"], "ci95": None, "bar": ">=0.01"},
                {"name": "diagnostic_composition_relative_l2", "estimate": result["composition"]["overall"]["relative_l2_error"], "ci95": None, "bar": "not opened because single-site parent failed"},
                {"name": "diagnostic_interaction_over_joint_rms", "estimate": result["interaction_over_joint_rms"], "ci95": None, "bar": "not opened because single-site parent failed"},
                {"name": "anticausal_site2_to_site1_max_absolute_effect", "estimate": result["anticausal_site2_to_site1_max_absolute_effect"], "ci95": None, "bar": "<=1e-5"},
            ],
            "prereg_artifact_id": "subject_number_two_site_prereg_v1",
            "result_artifact_id": "subject_number_two_site_invalid_v1",
            "input_artifact_ids": list(ARTIFACTS),
            "seed": None,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["basis_aligned/polynomial_causal/SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V1_PREREGISTRATION.md"],
            "notes": "No composition verdict is registered. The sole parent failure is one weak direction/site arm; all composition metrics are diagnostic only.",
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__":
    main()
