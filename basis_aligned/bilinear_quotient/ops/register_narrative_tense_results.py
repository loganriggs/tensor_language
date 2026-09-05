#!/usr/bin/env python3
"""Publish the narrative-tense causal screens into the registry-v2 authority.

This is intentionally a small, CPU-only publisher.  It creates one behavior
record using the existing registry-v2 schema, refuses artifact drift, and is a
no-op when the exact record is already present.  The fast-screen JSONL cannot
represent the custom head-factorial result without inventing missing timing and
specification fields, so the versioned task record is its canonical home.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import circuit_registry_v2 as registry


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
TAG = "task.narrative_tense.past_vs_present"
OUTPUT_NAME = "task_narrative_tense_past_vs_present.json"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
FIRST_NEXT_MISSING = (
    "split attention block 11 head 3 into exact source-score and cached-value terms, "
    "then test those terms with an output-token confound control that distinguishes "
    "shared copula writing from a shared agreement/tense semantic state"
)
SECOND_NEXT_MISSING = (
    "build a fresh disjoint capable narrative-tense authority before testing the "
    "outcome-selected remaining-source (R) hypothesis; do not repair the no-op "
    "tolerance or control bar on the same rows"
)
NEXT_MISSING = (
    "predeclare A1 template capability selection on FIT, evaluate it before route "
    "outcomes, and freeze an untouched construction holdout; rerun the unchanged-"
    "carrier experiment only if the selected authority is capable"
)

ARTIFACTS = {
    "narrative_tense_authority": (
        "basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_narrative_tense.py",
        "ef6212bd87aba24d2dfaa50f995884c2cb14ad72ecf2b481cbfe18d766bcdf8c",
        "dataset_authority",
    ),
    "past_present_v1_invalid_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_past_vs_present_v1_result.json",
        "5da1b51c762bbca72f210ce6520d5eb04095d0de9a80ee2cbf267ccdef5fa757",
        "screen_result",
    ),
    "past_present_v2_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_past_vs_present_v2_result.json",
        "5466980e1aa0a59538e4e8fcfb29457814c01e91cbe39bf41a2d42140fc7e71a",
        "screen_result",
    ),
    "short_cue_v1_invalid_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_short_cue_distance_v1_result.json",
        "abe10c97d176b28fceebcb1d659d86f0a3103c0b5c04747748c0fedd9b09dd44",
        "screen_result",
    ),
    "short_cue_v2_invalid_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_short_cue_distance_v2_result.json",
        "de6ad70ce25b1918a4fa888e59300286de850300ec5fd1ec071567bea1e106b1",
        "screen_result",
    ),
    "head3_complement_prior_art": (
        "basis_aligned/bilinear_quotient/circuits/prior_art/narrative_tense_attn11_head3_complement_factorial_v1.json",
        "704c1207c110e7f2384db1827d9a4dd5b03385b65586e7f5a87aaa7914b933e2",
        "preregistration",
    ),
    "head3_complement_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_attn11_head3_complement_factorial_v1_result.json",
        "7f31c50639ef20dd35f5bdaa5dcb9024221025db8856043baf104074f5b3c32b",
        "screen_result",
    ),
    "source_route_cross_task_prior_art": (
        "basis_aligned/bilinear_quotient/circuits/prior_art/narrative_tense_attn11_head3_source_route_cross_task_payload_v1.json",
        "18f97382fa020d7da7b0ab35d0c52537d500296cd3c5364cbb9ba68e4e664345",
        "preregistration",
    ),
    "source_route_cross_task_invalid_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_attn11_head3_source_route_cross_task_payload_v1_result.json",
        "4a56ae3c8e3fe5dd375f68f520624d91ada9bdfd07fa0e54f23bff40957933a4",
        "screen_result",
    ),
    "fresh_unchanged_carrier_prior_art": (
        "basis_aligned/bilinear_quotient/circuits/prior_art/narrative_tense_attn11_head3_fresh_unchanged_carrier_value_v1.json",
        "5978ab3cb345aff98b1af8f457db5db2dad05415cbc3c0dd2026e7747770b62c",
        "preregistration",
    ),
    "fresh_unchanged_carrier_invalid_result": (
        "basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_attn11_head3_fresh_unchanged_carrier_value_v1_result.json",
        "c066ed776544e6a540a5f8e7e55c205b93f1d260c07a84932cc4b35a83a2a564",
        "screen_result",
    ),
}


class PublicationError(ValueError):
    pass


def _bind_artifacts() -> dict[str, dict]:
    bound = {}
    for artifact_id, (relative, expected, kind) in ARTIFACTS.items():
        path = REPO / relative
        if not path.is_file():
            raise PublicationError(f"missing artifact: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PublicationError(
                f"artifact hash mismatch for {artifact_id}: {actual} != {expected}"
            )
        bound[artifact_id] = {
            "path": relative, "sha256": actual, "kind": kind, "status": "frozen"
        }
    return bound


def _metric(name: str, estimate: object, bar: str) -> dict:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def build_record() -> dict:
    artifacts = _bind_artifacts()
    claim_id = "narrative_tense_at_final_position.v1"
    revised_claim_id = "narrative_tense_at_final_position.v2"
    fresh_claim_id = "narrative_tense_at_final_position.v3"
    family_ids = ["a1_direct_narration", "a2_relative_clause", "p_surface_rewrite", "c_same_answer_rewrite"]
    families = [
        {
            "family_id": "a1_direct_narration", "role": "interchange",
            "changes": ["past versus present narrative frame", "correct answer was versus is"],
            "holds_fixed": ["direct-narration construction", "final prompt token", "answer position"],
            "builder_artifact_id": "narrative_tense_authority",
            "control_ids": ["A2 independent construction", "P same-state rewrite", "C same-answer rewrite"],
            "split_plan_id": "narrative_tense_fit_v1", "status": "validated",
        },
        {
            "family_id": "a2_relative_clause", "role": "interchange",
            "changes": ["past versus present narrative frame", "correct answer was versus is"],
            "holds_fixed": ["relative-clause construction", "final prompt token", "answer position"],
            "builder_artifact_id": "narrative_tense_authority",
            "control_ids": ["A1 independent construction", "P same-state rewrite", "C same-answer rewrite"],
            "split_plan_id": "narrative_tense_fit_v1", "status": "validated",
        },
        {
            "family_id": "p_surface_rewrite", "role": "invariance",
            "changes": ["surface wording within a fixed narrative tense"],
            "holds_fixed": ["narrative tense", "correct answer", "answer position"],
            "builder_artifact_id": "narrative_tense_authority",
            "control_ids": ["A1 opposite-tense transfer", "A2 opposite-tense transfer"],
            "split_plan_id": "narrative_tense_fit_v1", "status": "validated",
        },
        {
            "family_id": "c_same_answer_rewrite", "role": "invariance",
            "changes": ["unrelated past-description wording"],
            "holds_fixed": ["correct answer was", "answer position", "final prompt token"],
            "builder_artifact_id": "narrative_tense_authority",
            "control_ids": ["A1 opposite-tense transfer", "A2 opposite-tense transfer"],
            "split_plan_id": "narrative_tense_fit_v1", "status": "validated",
        },
    ]
    event_ids = [
        "narrative_tense.past_present.v1.invalid_capability",
        "narrative_tense.past_present.v2.held_localization",
        "narrative_tense.short_cue.v1.invalid_capability",
        "narrative_tense.short_cue.v2.invalid_capability",
        "narrative_tense.attn11_head3_complement.v1.held",
        "narrative_tense.attn11_head3_source_route_cross_task.v1.invalid",
        "narrative_tense.attn11_head3_fresh_unchanged_carrier.v1.invalid_capability",
    ]
    record = {
        "schema_version": 2,
        "tag": TAG,
        "identity": {
            "kind": "behavior_circuit", "instance": None,
            "identity_artifact_id": "narrative_tense_authority",
            "aliases": ["narrative_tense", "past_vs_present"],
        },
        "claims": [{
            "claim_id": claim_id, "revision": 1, "status": "site_live", "supersedes": None,
            "causal_variable": {
                "id": "narrative_tense_at_final_position",
                "domain": "past or present narrative frame across direct narration and relative-clause constructions",
                "read": "earlier lexical and inflectional tense cues",
                "operation": "carry the narrative time frame to the copular continuation",
                "write": "signed evidence for the one-token continuation ' was' versus ' is'",
                "endpoint": "donor-answer versus recipient-answer logit margin and binary-contrast CE under exact donor interchange",
            },
            "alternative_explanations": [
                "construction-specific lexical cue copying",
                "a generic copula or output-token write shared with agreement",
                "a broad attention-block effect not specific to head 3",
            ],
            "counterfactual_families": families,
            "candidate_sites": [
                {
                    "site_id": "residual.final_position.after_block18",
                    "tensor_path": "final-position residual after block 18",
                    "shape": ["batch", 1152],
                    "intervention": "replace the complete residual state with the matched natural donor state",
                    "ceiling_event_ids": [event_ids[1]],
                },
                {
                    "site_id": "attention.block11.output.final_position",
                    "tensor_path": "attention block 11 output at the final position",
                    "shape": ["batch", 1152],
                    "intervention": "replace the complete attention-block output with the matched natural donor output",
                    "ceiling_event_ids": [event_ids[4]],
                },
                {
                    "site_id": "attention.block11.head3.pre_output_projection.final_position",
                    "tensor_path": "head 3's slice before attention block 11 output projection at the final position",
                    "shape": ["batch", 128],
                    "intervention": "replace head 3 alone, with the other eight heads as the exact factorial complement",
                    "ceiling_event_ids": [event_ids[4]],
                },
            ],
            "split_plan_ids": ["narrative_tense_fit_v1"],
            # Keep the historical revision frozen to the five events available
            # when v1 was published.  The invalid follow-up belongs only to v2.
            "evidence_event_ids": event_ids[:5],
            "translation_ids": [],
            "next_missing": FIRST_NEXT_MISSING,
        }],
        "split_plans": [{
            "split_plan_id": "narrative_tense_fit_v1",
            "unit": "frozen linked narrative template row, paired in both causal directions",
            "partition_artifact_id": None,
            "builder_artifact_id": "narrative_tense_authority",
            "seed": None,
            "groups": {"FIT": 128},
            "leakage_group_keys": ["template row", "construction", "causal direction", "answer pair"],
            "sealed_before_outcomes": True,
            "sealed_at": "2026-09-05T08:36:11Z",
        }],
        "evidence_events": [], "translations": [], "artifacts": artifacts,
        "provenance": {
            "parent_fast_screen_result_sha256": ARTIFACTS["past_present_v2_result"][1],
            "head_factorial_result_sha256": ARTIFACTS["head3_complement_result"][1],
        },
    }
    revised_claim = copy.deepcopy(record["claims"][0])
    revised_claim.update({
        "claim_id": revised_claim_id,
        "revision": 2,
        "status": "site_live",
        "supersedes": claim_id,
        "evidence_event_ids": event_ids[:6],
        "next_missing": SECOND_NEXT_MISSING,
    })
    record["claims"].append(revised_claim)
    fresh_claim = copy.deepcopy(revised_claim)
    fresh_claim.update({
        "claim_id": fresh_claim_id,
        "revision": 3,
        "status": "site_live",
        "supersedes": revised_claim_id,
        "evidence_event_ids": event_ids,
        "next_missing": NEXT_MISSING,
    })
    record["claims"].append(fresh_claim)

    raw_events = [
        {
            "event_id": event_ids[0], "test_type": "capability", "stage": "invalid",
            "verdict": "invalid", "failure_kind": "invalid_instrument",
            "site_id": None, "result_artifact_id": "past_present_v1_invalid_result",
            "prereg_artifact_id": None,
            "metrics": [_metric("native_capability_gate_passed", 0.0, "must equal 1")],
            "supersedes_event_id": None,
        },
        {
            "event_id": event_ids[1], "test_type": "full_swap_ceiling", "stage": "complete",
            "verdict": "held", "failure_kind": None,
            "site_id": "residual.final_position.after_block18",
            "result_artifact_id": "past_present_v2_result", "prereg_artifact_id": None,
            "metrics": [
                _metric("A1_target_recovery", 1.0, ">=0.55"),
                _metric("A2_target_recovery", 1.0, ">=0.55"),
                _metric("donor_direction_fraction", 1.0, ">=0.75"),
                _metric("P_normalized_movement", 0.06486810422088089, "<=0.25"),
                _metric("C_normalized_movement", 0.14185669001905646, "<=0.35"),
            ],
            "supersedes_event_id": None,
        },
        {
            "event_id": event_ids[2], "test_type": "capability", "stage": "invalid",
            "verdict": "invalid", "failure_kind": "invalid_instrument", "site_id": None,
            "result_artifact_id": "short_cue_v1_invalid_result", "prereg_artifact_id": None,
            "metrics": [_metric("native_capability_gate_passed", 0.0, "must equal 1")],
            "supersedes_event_id": None,
        },
        {
            "event_id": event_ids[3], "test_type": "capability", "stage": "invalid",
            "verdict": "invalid", "failure_kind": "invalid_instrument", "site_id": None,
            "result_artifact_id": "short_cue_v2_invalid_result", "prereg_artifact_id": None,
            "metrics": [_metric("native_capability_gate_passed", 0.0, "must equal 1")],
            "supersedes_event_id": event_ids[2],
        },
        {
            "event_id": event_ids[4], "test_type": "composition", "stage": "complete",
            "verdict": "held", "failure_kind": None,
            "site_id": "attention.block11.head3.pre_output_projection.final_position",
            "result_artifact_id": "head3_complement_result",
            "prereg_artifact_id": "head3_complement_prior_art",
            "metrics": [
                _metric("minimum_head3_fraction_of_full", 0.9114422767818567, ">=0.50 in every A1/A2 direction cell"),
                _metric("minimum_head3_donor_direction_fraction", 1.0, ">=0.75 in every A1/A2 direction cell"),
                _metric("P_head3_normalized_movement", 0.022080523508868587, "<= full-attention movement"),
                _metric("C_head3_normalized_movement", 0.05898047040283231, "<= full-attention movement"),
            ],
            "supersedes_event_id": None,
        },
        {
            "event_id": event_ids[5], "event_claim_id": revised_claim_id,
            "test_type": "composition", "stage": "invalid", "verdict": "invalid",
            "failure_kind": "invalid_instrument",
            "site_id": "attention.block11.head3.pre_output_projection.final_position",
            "result_artifact_id": "source_route_cross_task_invalid_result",
            "prereg_artifact_id": "source_route_cross_task_prior_art",
            "metrics": [
                _metric("installed_noop_max_absolute_error", 0.0000591278076171875, "<=0.00005"),
                _metric("C_R_joint_mean_absolute_normalized_movement", 0.05991756523480643, "<= complete-H3 C movement"),
                _metric("C_complete_H3_mean_absolute_normalized_movement", 0.0589810113272569, "control reference only"),
            ],
            "supersedes_event_id": None,
            "notes": {
                "scientific_status": "invalid instrument; no route or cross-task conclusion",
                "descriptive_R_target_recovery": "retained only in the hash-bound result artifact and explicitly not evidence",
                "cross_task_results": {
                    "is_payload_transfer_passed": False,
                    "was_payload_transfer_passed": False,
                    "status": "descriptive failures under an invalid instrument",
                },
                "repair_prohibition": "no tolerance or control-bar repair on these rows",
            },
        },
        {
            "event_id": event_ids[6], "event_claim_id": fresh_claim_id,
            "test_type": "capability", "stage": "invalid", "verdict": "invalid",
            "failure_kind": "invalid_instrument",
            "site_id": "attention.block11.head3.pre_output_projection.final_position",
            "result_artifact_id": "fresh_unchanged_carrier_invalid_result",
            "prereg_artifact_id": "fresh_unchanged_carrier_prior_art",
            "metrics": [
                _metric("A1_past_minimum_native_capability", 0.75, ">=0.85 on both reciprocal side cells"),
                _metric("source_sum_max_absolute_error", 0.0, "<=0.00005"),
                _metric("same_batch_native_reinstall_max_absolute_error", 0.0, "<=0.00005"),
                _metric("pre_first_change_install_max_absolute_error", 0.0000171661376953125, "<=0.00005"),
            ],
            "supersedes_event_id": None,
            "notes": {
                "scientific_status": "invalid capability; exactness passed but no route or selectivity conclusion",
                "descriptive_only": [
                    "R-joint target recovery",
                    "R effective-value target recovery",
                    "post-last-change effective-value concentration",
                    "P/C selectivity measurements",
                ],
                "repair_prohibition": "do not select rows or weaken capability bars on this authority",
            },
        },
    ]
    for raw in raw_events:
        event_claim_id = raw.pop("event_claim_id", claim_id)
        event = {
            **raw, "claim_id": event_claim_id, "family_ids": family_ids,
            "evaluation_role": "frozen FIT screen",
            "input_artifact_ids": ["narrative_tense_authority"],
            "split_plan_id": "narrative_tense_fit_v1", "seed": None,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "replicates_event_id": None, "sections": [],
        }
        event["design_key"] = registry.design_key(record, event)
        event["execution_key"] = registry.execution_key(record, event)
        record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def apply_record(record: dict | None = None, *, regenerate: bool = True) -> Path:
    value = build_record() if record is None else record
    registry.validate_v2(value)
    path = registry.circuit_path(TAG)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing == value:
            if regenerate:
                registry.rebuild_registry_v2()
            return path
        # The only permitted migration is the exact v2 prefix emitted before
        # the fresh unchanged-carrier capability failure existed.
        base = copy.deepcopy(value)
        base["claims"] = base["claims"][:2]
        base["claims"][-1]["evidence_event_ids"] = base["claims"][-1]["evidence_event_ids"][:6]
        base["evidence_events"] = base["evidence_events"][:6]
        base["artifacts"].pop("fresh_unchanged_carrier_prior_art")
        base["artifacts"].pop("fresh_unchanged_carrier_invalid_result")
        if existing != base:
            raise PublicationError(f"canonical record differs: {path}")
        with registry._lock("registry"):
            current = json.loads(path.read_text())
            if current != base:
                raise PublicationError(f"canonical record moved during publication: {path}")
            registry._atomic_json(path, value)
        if regenerate:
            registry.rebuild_registry_v2()
        return path
    written = registry.write_behavior_circuit(value)
    if not regenerate:
        # write_behavior_circuit always rebuilds; tests may request no additional rebuild.
        pass
    return written


def main() -> None:
    path = apply_record()
    print(json.dumps({"written": str(path.relative_to(REPO)), "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
