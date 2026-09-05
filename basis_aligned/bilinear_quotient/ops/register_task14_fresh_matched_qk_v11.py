#!/usr/bin/env python3
"""Publish the licensed fresh Task14 L11H3 score/value factorial as v11.

This publisher is deliberately outcome-splitting: instrument validity, number
discrimination, lexical selectivity, bidirectional task use, and directional
asymmetry are separate evidence events even though they share one result file.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
import circuit_registry_v2 as registry  # noqa: E402


TAG = "task_subject_verb_number_agreement"
BASE_CLAIM = "grammatical_subject_number.v10"
NEW_CLAIM = "grammatical_subject_number.v11"
BASE_CLAIM_SHA256 = "437c58438a03300e5f959418a5407945025e366326f8b2597470d85e98342280"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"

ARTIFACT_SPECS = {
    "task14_fresh_matched_capability_authority": (
        "ops/circuit_fast_screen_candidate_task14_fresh_matched_natural_native_capability.py",
        "3d02183780bdae4f4b317b3cf04410ac2184af83d69d9c517fba10522d4f3449", "counterfactual_authority"),
    "task14_fresh_matched_capability_prior_art": (
        "circuits/prior_art/task14_fresh_matched_natural_native_capability_v1.json",
        "a0efc20022dfc96611b3bfb02a113c391c4d3b1f8c0533a3e6776c2026e2be5c", "preregistration"),
    "task14_fresh_matched_capability_result": (
        "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_result.json",
        "363d45f17e877ddfff50d99e41d999ced0acc33f91ef93b972023ede78031e18", "screen_result"),
    "task14_fresh_matched_capability_license_result": (
        "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_license_result.json",
        "ece04f3954ab8dd9fd3be73942af63cdf13d76f13854c07ccb02b31ce2d7553b", "capability_result"),
    "task14_fresh_matched_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_natural_native_capability_v1_license.json",
        "12d1835835612ce52629272309cbb49ff0af4d48dcabad45fe7e29e3fea94b4c", "capability_license"),
    "task14_fresh_matched_qk_scientific_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_natural_qk_factorial_v1.json",
        "d87ce6a857d1ac0dd58aee02822dad3a2fa99448bc3bfce4bddedd84a1034c44", "preregistration"),
    "task14_fresh_matched_qk_v1_engineering_failure": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_natural_qk_factorial_engineering_failure_v1.json",
        "34024d08e1d84dc6ad4ad7aca0a75f69ce91b7b971370026b736e019620cb034", "engineering_failure"),
    "task14_fresh_matched_qk_v2_repair": (
        "circuits/prior_art/task14_head11_3_fresh_matched_natural_qk_factorial_numerical_repair_v2.json",
        "1aaf9316c9475e8ca9a7ecc697d1f63d1f32dcecc2e37633dc8551926c0fce2b", "preregistration"),
    "task14_fresh_matched_qk_v2_invalid_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_natural_qk_factorial_v2_result.json",
        "d2655846b44aef36dfea791884ffe6899844a92fcb42e179c8ac3878f8b4d66f", "screen_result"),
    "task14_fresh_matched_qk_v3_repair": (
        "circuits/prior_art/task14_head11_3_fresh_matched_natural_qk_factorial_same_batch_repair_v3.json",
        "9c170ed8db667743bd62d761ebdd1876790d81c0a95898e9268245d653814399", "preregistration"),
    "task14_fresh_matched_qk_v3_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_natural_qk_factorial_v3_result.json",
        "69f954df45ec642cf2aa284f6ed2d20e68e0d9f9eefedc1546140a0579a2c981", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _load_artifacts() -> tuple[dict[str, dict], dict[str, dict]]:
    artifacts, docs = {}, {}
    for artifact_id, (relative, expected, kind) in ARTIFACT_SPECS.items():
        path = BQ / relative
        actual = registry.file_sha256(path)
        if actual != expected:
            raise PublicationError(f"hash mismatch for {path.name}: {actual} != {expected}")
        artifacts[artifact_id] = {
            "path": str(path.relative_to(REPO)), "sha256": actual,
            "kind": kind, "status": "frozen",
        }
        if path.suffix == ".json":
            docs[artifact_id] = json.loads(path.read_text())
    return artifacts, docs


def _base_event(event_id: str, test_type: str, verdict: str, failure_kind: str | None,
                result_id: str, prereg_id: str | None, metrics: list[dict], **extra: Any) -> dict:
    extra.pop("family_ids", None)
    event = {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "invalid" if verdict == "invalid" else "complete",
        "verdict": verdict, "failure_kind": failure_kind,
        # Events bind the exact v10 causal variable.  The v11 revision then
        # names the new families and appends these event IDs, avoiding a
        # claim/event circularity in the append-only registry API.
        "family_ids": [],
        "site_id": extra.pop("site_id", None), "split_plan_id": extra.pop("split_plan_id", SPLIT_ID),
        "evaluation_role": extra.pop("evaluation_role", "FRESH_LICENSED_HOLDOUT"),
        "metrics": metrics, "prereg_artifact_id": prereg_id,
        "result_artifact_id": result_id,
        "input_artifact_ids": extra.pop("input_artifact_ids", [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_result",
            "task14_fresh_matched_capability_license_result",
            "task14_fresh_matched_capability_license",
        ]),
        "seed": None, "checkpoint_sha256": extra.pop("checkpoint_sha256", CHECKPOINT),
        "supersedes_event_id": extra.pop("supersedes_event_id", None),
        "replicates_event_id": None, "sections": [],
        "notes": extra.pop("notes", ""),
    }
    if extra:
        raise PublicationError(f"unused event fields: {sorted(extra)}")
    return event


def build_plan() -> dict[str, Any]:
    artifacts, docs = _load_artifacts()
    record = json.loads(registry.circuit_path(TAG).read_text())
    latest = record["claims"][-1]["claim_id"]
    if latest not in {BASE_CLAIM, NEW_CLAIM}:
        raise PublicationError(f"canonical Task14 latest claim is {latest}, expected v10/v11")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v10 claim is not the exact audited base")

    capability = docs["task14_fresh_matched_capability_result"]
    license_result = docs["task14_fresh_matched_capability_license_result"]
    license_doc = docs["task14_fresh_matched_capability_license"]
    v1_failure = docs["task14_fresh_matched_qk_v1_engineering_failure"]
    v2 = docs["task14_fresh_matched_qk_v2_invalid_result"]
    v3 = docs["task14_fresh_matched_qk_v3_result"]
    if capability.get("terminal") != "licensed" or not capability["fit"]["passed"] or not capability["holdout"]["passed"]:
        raise PublicationError("fresh capability artifact is not the licensed result")
    if license_result.get("terminal") != "pass" or not license_result.get("native_only"):
        raise PublicationError("generic capability result is not a native-only pass")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != v3.get("candidate_id"):
        raise PublicationError("capability license does not bind the scientific candidate")
    if v1_failure.get("scientific_outcome_scored") is not False or v1_failure.get("result_written") is not False:
        raise PublicationError("v1 artifact is not a pre-scoring engineering failure")
    if v2.get("terminal") != "invalid" or v3.get("terminal") != "valid_causal_screen":
        raise PublicationError("v2/v3 terminals changed")
    for result in (v2, v3):
        if result.get("checkpoint_weights_sha256") != CHECKPOINT:
            raise PublicationError("checkpoint changed")
        if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
            raise PublicationError("causal split scope changed")
    score = v3["score"]
    if score["predictions"] != {
        "pred_a_instrument_live": True,
        "pred_b_number_score_discriminative": False,
        "pred_c_lexically_selective": False,
        "pred_d_bidirectional_task_use": False,
        "pred_e_directionally_asymmetric_task_use": True,
    }:
        raise PublicationError("v3 outcome contract changed")
    cells = list(score["cells"].values())
    number_effects = [abs(c[state]["mean"]) for c in cells for state in
                      ("number_score_same_value", "number_score_opposite_value")]
    number_signs = [c[state]["expected_margin_sign_fraction"] for c in cells for state in
                    ("number_score_same_value", "number_score_opposite_value")]
    lexical_ratios = [c[state]["absolute_mean_over_number"] for c in cells for state in
                      ("lexical_score_same_value", "lexical_score_opposite_value")]
    helpful_margin = [c["number_score_opposite_value"]["donor_helpful_margin_fraction"] for c in cells]
    helpful_ce = [c["number_score_opposite_value"]["donor_helpful_CE_fraction"] for c in cells]
    helpful_mean_margin = [c["number_score_opposite_value"]["mean_donor_margin_improvement"] for c in cells]
    helpful_mean_ce = [c["number_score_opposite_value"]["mean_donor_CE_improvement"] for c in cells]
    complete_margin = [c["complete_opposite_head"]["mean_donor_margin_improvement"] for c in cells]
    complete_ce = [c["complete_opposite_head"]["mean_donor_CE_improvement"] for c in cells]
    complete_margin_fraction = [c["complete_opposite_head"]["donor_margin_improvement_fraction"] for c in cells]
    complete_ce_fraction = [c["complete_opposite_head"]["donor_CE_improvement_fraction"] for c in cells]

    capability_event = _base_event(
        "task14_head11_3.fresh_matched_natural.capability.complete.v1", "capability", "held", None,
        "task14_fresh_matched_capability_result", "task14_fresh_matched_capability_prior_art",
        [metric("FIT_minimum_cell_accuracy", min(x["accuracy"] for x in capability["fit"]["cells"].values()), ">=0.875 each cell"),
         metric("HOLDOUT_minimum_cell_accuracy", min(x["accuracy"] for x in capability["holdout"]["cells"].values()), ">=0.875 each cell"),
         metric("model_forwards", capability["active_price"]["model_forwards"], "2"),
         metric("causal_interventions", capability["causal_interventions"], "0")],
        site_id=None, evaluation_role="FRESH_MATCHED_NATURAL_CAPABILITY_FIT_HOLDOUT",
        family_ids=[], notes="Native capability only. This licenses the bound causal candidate but is not QK evidence.")
    failure_event = _base_event(
        "task14_head11_3.fresh_matched_qk.v1_engineering_failure.invalid.v1", "null_control", "invalid", "implementation_failure",
        "task14_fresh_matched_qk_v1_engineering_failure", "task14_fresh_matched_qk_scientific_prior_art",
        [metric("result_written", 0, "must equal 0"), metric("scientific_outcome_scored", 0, "must equal 0")],
        notes="Pre-scoring engineering failure retained as provenance; it contains no scientific outcome.")
    v2_event = _base_event(
        "task14_head11_3.fresh_matched_qk.v2_instrument.invalid.v1", "null_control", "invalid", "invalid_instrument",
        "task14_fresh_matched_qk_v2_invalid_result", "task14_fresh_matched_qk_v2_repair",
        [metric("patched_dispatch_recipient_head_max_absolute_error", v2["score"]["patched_dispatch_recipient_head_max_absolute_error"], "<=0.00007"),
         metric("same_score_same_value_endpoint_max_absolute_error", v2["score"]["same_score_same_value_endpoint_max_absolute_error"], "<=0.00007")],
        supersedes_event_id=failure_event["event_id"],
        notes="Invalid numerical instrument. Scientific scores in this artifact are descriptive only and are not evidence.")
    instrument_event = _base_event(
        "task14_head11_3.fresh_matched_qk.v3_instrument.complete.v1", "null_control", "held", None,
        "task14_fresh_matched_qk_v3_result", "task14_fresh_matched_qk_v3_repair",
        [metric("native_replay_max_absolute_logit_error", score["native_replay_max_absolute_logit_error"], "<=0.00007"),
         metric("same_batch_native_noop_endpoint_max_absolute_error", score["same_batch_native_noop_endpoint_max_absolute_error"], "<=0.00007"),
         metric("source_term_sum_max_absolute_error", score["source_term_sum_max_absolute_error"], "<=0.00005"),
         metric("installed_term_max_absolute_error", score["installed_term_max_absolute_error"], "<=0.00005"),
         metric("complete_head_vector_max_absolute_error", score["complete_head_vector_max_absolute_error"], "<=0.00005"),
         metric("minimum_complete_head_mean_donor_margin_improvement", min(complete_margin), ">=0.05 each cell"),
         metric("minimum_complete_head_mean_donor_CE_improvement", min(complete_ce), ">=0 each cell"),
         metric("minimum_complete_head_row_improvement_fraction", min(complete_margin_fraction + complete_ce_fraction), ">=0.75 each cell")],
        site_id="attention.block11.head3.fronted_subject_self_score.qk_factors.final_position",
        family_ids=["fresh_matched_subject_number_score_value_v1"],
        supersedes_event_id=v2_event["event_id"],
        notes="The corrected same-batch instrument and complete-head positive control passed. This validates the intervention, not the scientific hypotheses by itself.")
    common = dict(site_id="attention.block11.head3.fronted_subject_self_score.qk_factors.final_position",
                  result_id="task14_fresh_matched_qk_v3_result", prereg_id="task14_fresh_matched_qk_scientific_prior_art")
    number_event = _base_event(
        "task14_head11_3.fresh_matched_qk.number_discrimination.complete.v1", "composition", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("minimum_number_score_absolute_mean_margin_effect", min(number_effects), ">=0.05 at both value states in every cell"),
         metric("minimum_expected_row_sign_fraction", min(number_signs), ">=0.75 at both value states in every cell"),
         metric("cells_passing_both_value_state_signs", sum(all(c[s]["expected_margin_sign_fraction"] >= .75 for s in ("number_score_same_value", "number_score_opposite_value")) for c in cells), "4 of 4")],
        site_id=common["site_id"], family_ids=["fresh_matched_subject_number_score_value_v1"],
        notes="The score effect was large enough, but the registered number-dependent signs failed in both plural-to-singular cells.")
    lexical_event = _base_event(
        "task14_head11_3.fresh_matched_qk.lexical_selectivity.complete.v1", "invariance", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("maximum_same_number_lexical_over_number_margin_ratio", max(lexical_ratios), "<=0.25 in every value-state cell"),
         metric("cells_passing_both_value_state_lexical_ratios", sum(all(c[s]["absolute_mean_over_number"] <= .25 for s in ("lexical_score_same_value", "lexical_score_opposite_value")) for c in cells), "4 of 4")],
        site_id=common["site_id"], family_ids=["fresh_matched_same_number_lexical_score_v1"],
        notes="Same-number lemma changes were not uniformly small relative to opposite-number score changes.")
    use_event = _base_event(
        "task14_head11_3.fresh_matched_qk.bidirectional_task_use.complete.v1", "cross_family_transfer", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("minimum_opposite_score_opposite_value_mean_donor_margin_improvement", min(helpful_mean_margin), ">=0.05 each cell"),
         metric("minimum_opposite_score_opposite_value_mean_donor_CE_improvement", min(helpful_mean_ce), ">=0 each cell"),
         metric("minimum_donor_helpful_row_fraction", min(helpful_margin + helpful_ce), ">=0.75 each cell"),
         metric("direction_template_cells_passing_task_use", sum(m >= .75 and c >= .75 and mm >= .05 and mc >= 0 for m, c, mm, mc in zip(helpful_margin, helpful_ce, helpful_mean_margin, helpful_mean_ce)), "4 of 4")],
        site_id=common["site_id"], family_ids=["fresh_matched_subject_number_score_value_v1"],
        notes="Opposite-number score with opposite-number value helped singular-to-plural but harmed plural-to-singular, so bidirectional task use failed.")
    asymmetry_event = _base_event(
        "task14_head11_3.fresh_matched_qk.directional_asymmetry.complete.v1", "composition", "held", None,
        common["result_id"], common["prereg_id"],
        [metric("singular_to_plural_minimum_donor_helpful_margin_fraction", min(c["number_score_opposite_value"]["donor_helpful_margin_fraction"] for k, c in score["cells"].items() if k.startswith("singular_to_plural")), "1.0 in both templates"),
         metric("plural_to_singular_maximum_donor_helpful_margin_fraction", max(c["number_score_opposite_value"]["donor_helpful_margin_fraction"] for k, c in score["cells"].items() if k.startswith("plural_to_singular")), "0.0 in both templates"),
         metric("singular_to_plural_mean_donor_margin_improvement_range", [min(v for k, v in zip(score["cells"], helpful_mean_margin) if k.startswith("singular_to_plural")), max(v for k, v in zip(score["cells"], helpful_mean_margin) if k.startswith("singular_to_plural"))], ">0"),
         metric("plural_to_singular_mean_donor_margin_improvement_range", [min(v for k, v in zip(score["cells"], helpful_mean_margin) if k.startswith("plural_to_singular")), max(v for k, v in zip(score["cells"], helpful_mean_margin) if k.startswith("plural_to_singular"))], "<0")],
        site_id=common["site_id"], family_ids=["fresh_matched_subject_number_score_value_v1"],
        notes="Held alternative: the exact score/value combination has opposite task effects by direction. This does not rescue the separate bidirectional-use null.")
    events = [capability_event, failure_event, v2_event, instrument_event,
              number_event, lexical_event, use_event, asymmetry_event]

    split_plan = {
        "split_plan_id": SPLIT_ID,
        "unit": "16 frozen noun groups; each group keeps two templates and recipient/opposite-number/same-number-lexical endpoints together",
        "partition_artifact_id": "task14_fresh_matched_capability_authority",
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "seed": "task14-fresh-matched-natural-v1|groups-0-7-fit|groups-8-15-holdout",
        "groups": {"FIT": 8, "LICENSED_HOLDOUT": 8},
        "leakage_group_keys": ["noun group and forms", "exact prompt and token ids", "row id", "template", "direction"],
        "sealed_before_outcomes": True, "sealed_at": "2026-09-05T13:42:00Z",
    }
    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 11, "supersedes": BASE_CLAIM,
        "status": "site_live", "split_plan_ids": base["split_plan_ids"] + [SPLIT_ID],
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed fresh matched-natural QK factorial is complete and must not be repeated: its instrument was valid, "
            "but the registered number-discrimination, lexical-selectivity, and bidirectional task-use claims were null. "
            "The supported effect is directional asymmetry: the subject-self score/value combination helped singular-to-plural "
            "and harmed plural-to-singular. Next test a separately preregistered explanation of that asymmetry or a different "
            "causal score basis, with selective removal and unrelated-task controls; do not weaken these rows or bars."
        ),
    })
    families = revision["counterfactual_families"]
    family_ids = {f["family_id"] for f in families}
    additions = [
        {"family_id": "fresh_matched_subject_number_score_value_v1", "role": "interchange",
         "changes": ["the exact token-8 subject-self score, subject effective value, or both between opposite subject numbers"],
         "holds_fixed": ["licensed HOLDOUT template and attractors", "all non-subject L11H3 source terms and answer position"],
         "builder_artifact_id": "task14_fresh_matched_capability_authority",
         "control_ids": ["same-batch native no-op and source algebra", "complete opposite-head positive control", "same-number different-lemma score"],
         "split_plan_id": SPLIT_ID, "status": "validated"},
        {"family_id": "fresh_matched_same_number_lexical_score_v1", "role": "invariance",
         "changes": ["the subject lemma in the exact token-8 self-score while preserving subject number"],
         "holds_fixed": ["subject number and is/are answer", "template, attractors, value state, and non-subject source terms"],
         "builder_artifact_id": "task14_fresh_matched_capability_authority",
         "control_ids": ["matched opposite-number score effect", "native and opposite subject-value states"],
         "split_plan_id": SPLIT_ID, "status": "failed"},
    ]
    for family in additions:
        if family["family_id"] not in family_ids:
            families.append(family)
    return {"schema": "task14_fresh_matched_qk_publication_v11", "canonical_tag": TAG,
            "artifacts": artifacts, "split_plans": [split_plan], "events": events,
            "claim_revision": revision}


def _keyed(record: dict, event: dict) -> dict:
    out = copy.deepcopy(event)
    out["design_key"] = registry.design_key(record, out)
    out["execution_key"] = registry.execution_key(record, out)
    return out


def apply_plan(plan: dict, *, regenerate: bool = True) -> Path:
    path = registry.circuit_path(plan["canonical_tag"])
    preview = json.loads(path.read_text())
    for artifact_id, artifact in plan["artifacts"].items():
        if artifact_id in preview["artifacts"] and preview["artifacts"][artifact_id] != artifact:
            raise PublicationError(f"artifact collision: {artifact_id}")
        preview["artifacts"][artifact_id] = artifact
    for split in plan["split_plans"]:
        found = [x for x in preview["split_plans"] if x["split_plan_id"] == split["split_plan_id"]]
        if found and found != [split]:
            raise PublicationError("split-plan collision")
        if not found:
            preview["split_plans"].append(split)
    revision = plan["claim_revision"]
    found_claim = [x for x in preview["claims"] if x["claim_id"] == NEW_CLAIM]
    if found_claim and found_claim != [revision]:
        raise PublicationError("claim revision collision")
    if not found_claim:
        preview["claims"].append(revision)
    for event in plan["events"]:
        expected = _keyed(preview, event)
        found = [x for x in preview["evidence_events"] if x["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            preview["evidence_events"].append(expected)
    registry.validate_v2(preview)

    registry.append_artifacts(plan["canonical_tag"], plan["artifacts"])
    registry.append_split_plans(plan["canonical_tag"], plan["split_plans"])
    for event in plan["events"]:
        current = json.loads(path.read_text())
        expected = _keyed(current, event)
        found = [x for x in current["evidence_events"] if x["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            registry.append_evidence_event(plan["canonical_tag"], event)
    current = json.loads(path.read_text())
    if not any(x["claim_id"] == NEW_CLAIM for x in current["claims"]):
        registry.append_claim_revision(plan["canonical_tag"], revision)
    registry.validate_v2(json.loads(path.read_text()))
    if regenerate:
        for script in ("make_circuit_coverage.py", "make_circuit_experiment_index.py", "make_circuit_campaign_queue.py"):
            subprocess.run([sys.executable, str(BQ / script)], cwd=REPO, check=True)
        registry.rebuild_registry_v2()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    if args.apply:
        apply_plan(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
