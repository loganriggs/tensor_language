#!/usr/bin/env python3
"""Advance Task14 authority over committed post-v22 program evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    _atomic_json, _lock, circuit_path, design_key, execution_key,
    file_sha256, rebuild_registry_v2, validate_v2,
)


TAG = "task_subject_verb_number_agreement"
OLD_CLAIM = "grammatical_subject_number.v22"
NEW_CLAIM = "grammatical_subject_number.v23"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT = "task14_fresh_matched_natural_split_v1"
ARTIFACTS = {
    "post_v22_audit_source": ("basis_aligned/polynomial_causal/audit_subject_number_post_v22_authority_v1.py", "audit_implementation"),
    "post_v22_audit": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_POST_V22_AUTHORITY_AUDIT_V1.json", "audit"),
    "post_v22_ood_tangent": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_v1_result.json", "result"),
    "post_v22_ood_gate": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json", "result"),
    "post_v22_prompt_loo": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_ood_mlp6_7_prompt_level_composition_crossvalidation_v1_result.json", "result"),
    "post_v22_fresh_composition": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_result.json", "result"),
    "post_v22_continuous_edit": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1_result.json", "result"),
    "post_v22_zero_anchor": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1_result.json", "result"),
    "post_v22_fixed_reader": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json", "result"),
    "post_v22_guided_edit": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_fixed_reader_guided_margin_edit_v1_result.json", "result"),
    "post_v22_upstream_program": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json", "result"),
    "post_v22_narrow_collateral": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2_result.json", "audit"),
    "post_v22_possessive_reuse": ("basis_aligned/bilinear_quotient/circuits/fast_screens/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2_result.json", "result"),
    "post_v22_mediator_split": ("basis_aligned/bilinear_quotient/circuits/followups/task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1_result.json", "result"),
}


EVENT_SPECS = (
    ("task14.mlp6_7.ood_tangent_geometry.held.v1", "ood", "held", None, "post_v22_ood_tangent", "MLP6_7.distributed_background_gate_to_L11H3", "OOD midpoint geometry and task prediction transfer; task use is background gated and lexical specificity remains null."),
    ("task14.mlp6_7.ood_distributed_background_gate.held.v1", "composition", "held", None, "post_v22_ood_gate", "MLP6_7.distributed_background_gate_to_L11H3", "Exact E/A/U/W factorial identifies a distributed, signed, nearly first-order background gate."),
    ("task14.mlp6_7.prompt_loo_composition.held.v1", "cross_family_transfer", "held", None, "post_v22_prompt_loo", "MLP6_7.distributed_background_gate_to_L11H3", "Leave-one-prompt-out coefficients predict fourteen omitted background subsets per prompt."),
    ("task14.mlp6_7.fresh_construction_composition.held.v1", "cross_family_transfer", "held", None, "post_v22_fresh_composition", "MLP6_7.distributed_background_gate_to_L11H3", "Frozen composition profile transfers to two new fronted templates."),
    ("task14.mlp6_7.continuous_background_edit.held.v1", "composition", "held", None, "post_v22_continuous_edit", "MLP6_7.distributed_background_gate_to_L11H3", "Continuous interpolation/extrapolation drives the signed task effect predictably."),
    ("task14.mlp6_7.zero_anchor_absolute_coefficients.null.v1", "cross_family_transfer", "null", "scientific_null", "post_v22_zero_anchor", "MLP6_7.distributed_background_gate_to_L11H3", "Untouched text rejects globally shared absolute coefficients; target context supplies baseline and amplitude."),
    ("task14.mlp6_7.fixed_direction_readers.cross_corpus.held.v1", "compiled_equivalence", "held", None, "post_v22_fixed_reader", "L11H3.projected_write.direction_reader", "Two fixed projected-write readers predict a new corpus lattice without target-tail execution or calibration."),
    ("task14.mlp6_7.fixed_reader_guided_edit.held.v1", "composition", "held", None, "post_v22_guided_edit", "L11H3.projected_write.direction_reader", "Frozen reader amplitudes choose gains that hit a requested signed margin effect."),
    ("task14.mlp6_7.direction_cardinality_program.held.v1", "compiled_equivalence", "held", None, "post_v22_upstream_program", "L11H3.projected_write.direction_cardinality_program", "Ten fixed direction-by-cardinality writes prospectively substitute native grouped-source effects on a third corpus."),
    ("task14.mlp6_7.direction_cardinality_program.narrow_collateral.held.v1", "null_control", "held", None, "post_v22_narrow_collateral", "L11H3.projected_write.direction_cardinality_program", "All ten literal writes preserve frozen numbered-list and bracket controls after a numerical-only audit."),
    ("task14.mlp6_7.direction_cardinality_program.possessive_reuse.null.v1", "composition", "null", "scientific_null", "post_v22_possessive_reuse", "L11H3.projected_write.direction_cardinality_program", "The fixed write does not transfer as a generic possessive-number state."),
    ("task14.mlp6_7.direction_cardinality_program.mlp15_17_distributed.held.v1", "composition", "held", None, "post_v22_mediator_split", "L11H3.projected_write.direction_cardinality_program", "MLP15 and MLP17 contribute additively to downstream mediation."),
    ("task14.mlp6_7.direction_cardinality_program.single_mediator.null.v1", "composition", "null", "scientific_null", "post_v22_mediator_split", "L11H3.projected_write.direction_cardinality_program", "Neither MLP15 nor MLP17 is a single dominant mediator."),
)


def artifact(path, kind):
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def predictions(doc): return doc.get("predictions") or doc.get("score", {}).get("predictions", {})


def metric(name, estimate, bar): return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def make_event(record, spec, docs):
    event_id, test_type, verdict, failure, result_id, site_id, notes = spec
    pred = predictions(docs[result_id])
    event = {
        "event_id": event_id, "claim_id": NEW_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": verdict, "failure_kind": failure,
        "family_ids": [], "site_id": site_id, "split_plan_id": SPLIT,
        "evaluation_role": "post_v22_committed_primary_receipt_reconciliation",
        "metrics": [metric(name, value, "frozen registered predicate") for name, value in sorted(pred.items())],
        "prereg_artifact_id": None, "result_artifact_id": result_id,
        "input_artifact_ids": [result_id, "post_v22_audit"], "seed": None,
        "checkpoint_sha256": CHECKPOINT, "supersedes_event_id": None,
        "replicates_event_id": None, "sections": [ARTIFACTS[result_id][0]], "notes": notes,
    }
    event["design_key"] = design_key(record, event)
    event["execution_key"] = execution_key(record, event)
    return event


def main():
    audit = json.loads((HERE / "SUBJECT_NUMBER_POST_V22_AUTHORITY_AUDIT_V1.json").read_text())
    assert audit["audit_conclusion"]["registry_is_stale"] is True
    docs = {artifact_id: json.loads((REPO / path).read_text()) for artifact_id, (path, _) in ARTIFACTS.items() if artifact_id not in ("post_v22_audit_source",)}
    event_ids = [spec[0] for spec in EVENT_SPECS]
    path = circuit_path(TAG)
    existing = json.loads(path.read_text())
    if all(any(item["event_id"] == key for item in existing["evidence_events"]) for key in event_ids):
        validate_v2(existing); rebuild_registry_v2()
        print(json.dumps({"status":"already_registered","claim_id":NEW_CLAIM},indent=2)); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        if any(any(item["event_id"] == key for item in record["evidence_events"]) for key in event_ids):
            raise RuntimeError("partial post-v22 registration exists")
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 23, "supersedes": OLD_CLAIM,
            "status": "weights_translated", "evidence_event_ids": [*previous["evidence_event_ids"], *event_ids],
            "next_missing": (
                "A compact conditional interface is prospectively supported at L11H3: ten direction-by-cardinality "
                "1152D upstream writes plus two fixed 1152D readers predict, compose and manipulate new-text "
                "effects, with narrow bracket/list collateral. Global zero-anchor amplitudes, generic possessive "
                "reuse, a singleton downstream mediator and universal lexical specificity fail. Native text-state "
                "generation, the direction/cardinality selector, suffix execution, broader collateral and selective "
                "necessity remain external. Do not repeat OOD, gain, reader, program or narrow-collateral screens."
            ),
        })
        claim["causal_variable"] = {
            **claim["causal_variable"],
            "read": "a grouped MLP6--7 contextual source response, dispatched by agreement direction and four-factor background cardinality",
            "operation": "install one of ten fixed projected L11H3 writes and score its task effect with one of two fixed direction readers",
        }
        claim["candidate_sites"].extend([
            {"site_id":"MLP6_7.distributed_background_gate_to_L11H3","tensor_path":"grouped MLP6--7 response under E/A/U/W background subsets through exact MLP8 and L11H3","shape":["batch",16,1152],"intervention":"binary-subset and continuous background scaling with grouped source fixed","ceiling_event_ids":[event_ids[3],event_ids[4]]},
            {"site_id":"L11H3.projected_write.direction_reader","tensor_path":"two fixed direction-specific linear readers on the 1152D projected L11H3 write","shape":[2,1152],"intervention":"prediction and gain selection only; target native tail is unopened during prediction","ceiling_event_ids":[event_ids[6],event_ids[7]]},
            {"site_id":"L11H3.projected_write.direction_cardinality_program","tensor_path":"ten fixed direction-by-background-cardinality projected L11H3 write prototypes","shape":[10,1152],"intervention":"install the dispatched write on new text and recompute the native suffix","ceiling_event_ids":[event_ids[8],event_ids[9]]},
        ])
        record["claims"].append(claim)
        for spec in EVENT_SPECS: record["evidence_events"].append(make_event(record, spec, docs))
        validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status":"registered","claim_id":NEW_CLAIM,"events":len(EVENT_SPECS)},indent=2))


if __name__ == "__main__": main()
