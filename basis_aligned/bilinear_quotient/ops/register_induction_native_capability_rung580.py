#!/usr/bin/env python3
"""Register the gate-compliant R580 induction capability experiment before execution."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    file_sha256, rebuild_registry_v2, validate_v2,
)


TAG = "task.induction.selector_payload"
OLD_CLAIM = "induction_selector_and_payload.v8"
NEW_CLAIM = "induction_selector_and_payload.v9"
OLD_EVENT_ID = "induction_selector_payload_native_capability.r580.preregistered.v1"
EVENT_ID = "induction_selector_payload_native_capability.r580.preregistered.v2"
SPLIT_ID = "induction_selector_payload_three_source_split_r578_v1"
SITE_ID = "native_three_source_final_token_logits"
PATHS = {
    "r580_capability_preregistration_v2": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG580_PREREGISTRATION.md",
        "preregistration"),
    "r580_capability_implementation_v2": (
        "basis_aligned/bilinear_quotient/ops/induction_selector_payload_native_capability_rung580.py",
        "implementation"),
    "r580_capability_test_v2": (
        "basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_native_capability_rung580.py", "test"),
    "r580_capability_dryrun_v2": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung580_dryrun.json",
        "dryrun_receipt"),
}
FAMILIES = [
    "two_valid_sources_selector_swap", "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved", "match_break_payload_preserved",
    "irrelevant_source_edit", "irrelevant_payload_edit", "contrast_target_source_edit",
    "copy_relation_preserved_nuisance_change",
]


def artifact(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    dryrun = json.loads((REPO / PATHS["r580_capability_dryrun_v2"][0]).read_text())
    assert dryrun["status"] == "dryrun_passed"
    assert dryrun["groups"] == 108 and dryrun["rows"] == 3240
    assert dryrun["unique_sequences"] == 3024 and dryrun["literal_expected_forwards"] == 95
    assert dryrun["passing_fixture_verdict"] == "held_capability_screen"
    assert dryrun["null_fixture_verdict"] == "scientific_null"
    assert dryrun["model_loaded"] is False and dryrun["model_forwards"] == 0
    assert dryrun["forbidden_splits_opened"] == []
    artifacts = {key: artifact(*value) for key, value in PATHS.items()}
    path = circuit_path(TAG)
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(event["event_id"] == EVENT_ID for event in record["evidence_events"]):
            # The v1 record was created immediately before the managed preflight found that the
            # otherwise frozen script lacked the runner's top-level pred_a/b/c interface.  The
            # scientific clauses did not change and no model outcome existed.  Its artifact paths
            # point at the same files, so keep that superseded event internally valid by refreshing
            # those path hashes to the pre-outcome corrected files and record why.
            legacy_to_current = {
                "r580_capability_preregistration": "r580_capability_preregistration_v2",
                "r580_capability_implementation": "r580_capability_implementation_v2",
                "r580_capability_test": "r580_capability_test_v2",
                "r580_capability_dryrun": "r580_capability_dryrun_v2",
            }
            for legacy_id, current_id in legacy_to_current.items():
                record["artifacts"][legacy_id] = deepcopy(artifacts[current_id])
            old_event = next(
                event for event in record["evidence_events"]
                if event["event_id"] == OLD_EVENT_ID
            )
            clarification = (
                " Pre-outcome artifact hashes were refreshed after managed preflight required "
                "top-level pred_a/b/c fields; the five frozen scientific clauses did not change."
            )
            if clarification.strip() not in old_event["notes"]:
                old_event["notes"] += clarification
            old_event["execution_key"] = execution_key(record, old_event)
            validate_v2(record)
            _atomic_json(path, record)
            print(json.dumps({"status": "preflight_artifacts_repaired", "event_id": EVENT_ID}, indent=2))
            return
        for artifact_id, value in artifacts.items():
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM,
            "revision": 9,
            "supersedes": OLD_CLAIM,
            "status": "specified",
            "evidence_event_ids": [*previous["evidence_event_ids"], EVENT_ID],
            "next_missing": (
                "execute and independently audit the frozen R580 FIT/SELECT native-capability screen; only a held, "
                "audited result may license adapting the existing R557 score/value algebra and R558 interaction "
                "lattice to R578"),
        })
        site = {"site_id": SITE_ID, "tensor_path": "unmodified final-query model logits for R578 prompts",
                "shape": ["batch", "vocabulary"],
                "intervention": "no intervention; measure native B-versus-D margins, factorial interaction, and controls",
                "ceiling_event_ids": [EVENT_ID]}
        if not any(item["site_id"] == SITE_ID for item in claim["candidate_sites"]):
            claim["candidate_sites"].append(site)
        record["claims"].append(claim)
        event = {
            "event_id": EVENT_ID, "claim_id": NEW_CLAIM, "test_type": "capability",
            "stage": "preregistered", "verdict": "inconclusive", "failure_kind": None,
            "family_ids": FAMILIES, "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
            "evaluation_role": "FIT_and_SELECT_native_selector_by_payload_capability_before_any_site_search",
            "metrics": [
                {"name": "four_factorial_cells", "estimate": None, "ci95": None,
                 "bar": ">=0.75 correct and positive group-bootstrap lower mean B/D margin in every cell and split"},
                {"name": "selector_payload_interaction", "estimate": None, "ci95": None,
                 "bar": "positive group-bootstrap lower mean signed interaction in FIT and SELECT"},
                {"name": "relation_preserving_controls", "estimate": None, "ci95": None,
                 "bar": "all 64 split/control/condition/endpoint cells satisfy accuracy and positive lower margin"},
                {"name": "selected_match_necessity_and_neutral_gap", "estimate": None, "ci95": None,
                 "bar": ">=0.70 positive selected-match drops and positive lower means for drop and paired neutral gap"},
                {"name": "execution_envelope", "estimate": None, "ci95": None,
                 "bar": "3,024 unique prompts exactly once in 95 forwards; zero backwards; FINAL_TEST/OOD closed"},
            ],
            "prereg_artifact_id": "r580_capability_preregistration_v2", "result_artifact_id": None,
            "input_artifact_ids": [
                "r578_three_source_preregistration", "r578_three_source_rows", "r578_three_source_receipt",
                "r578_three_source_builder", "r578_three_source_test", "r580_capability_implementation_v2",
                "r580_capability_test_v2", "r580_capability_dryrun_v2",
            ],
            "seed": 580,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": OLD_EVENT_ID, "replicates_event_id": None,
            "sections": [
                "polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG580_PREREGISTRATION.md"],
            "notes": (
                "Supersedes the preflight-rejected v1 authority before any model outcome. The only scientific-text "
                "clarification groups the same five frozen clauses into the runner-required top-level pred_a/b/c fields. "
                "Contrast-source edits are reported but not gated; scientific failures save raw evidence and return a null."
            ),
        }
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    print(json.dumps({"status": "registered", "claim_id": NEW_CLAIM, "event_id": EVENT_ID,
                      "planned_forwards": 95, "model_outcomes_opened": False}, indent=2))


if __name__ == "__main__":
    main()
