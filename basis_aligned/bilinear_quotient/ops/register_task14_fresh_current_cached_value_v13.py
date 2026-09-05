#!/usr/bin/env python3
"""Publish the corrected licensed Task14 current/cached-value factorial as v13."""

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
BASE_CLAIM = "grammatical_subject_number.v12"
NEW_CLAIM = "grammatical_subject_number.v13"
BASE_CLAIM_SHA256 = "267b2b5e087e618ef96f2ef3e844bb4fdde00dc8884c37c4bf498e31a0f50c3f"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "attention.block11.head3.subject_effective_value.current_and_cached.final_position"

ARTIFACT_SPECS = {
    "task14_current_cached_v1_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_cached_value_factorial_v1_capability_license.json",
        "ce418412936aab8a8df37460dc8ef54b0555980d3625377903e88cb686f8070f", "capability_license"),
    "task14_current_cached_v2_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_cached_value_factorial_v2_capability_license.json",
        "b5ba447870147311b76211b9ff52c4e672f7a095c96488381d8147ef9b403ab4", "capability_license"),
    "task14_current_cached_scientific_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v1.json",
        "9ca110e1614c6bc8e3fb1dd771060ae6ac35c39088eb80728a4eece2160f9b9c", "preregistration"),
    "task14_current_cached_v1_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial.py",
        "d76044585a8d9c7d8aea95bada0ba9adb2aa19aac5e13e2cfbf52675eacc4f5f", "experiment_runner"),
    "task14_current_cached_v1_invalid_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v1_result.json",
        "df090c4e22070b6c151b12a4e80e25ca75c132d7210d3f57dc8de79c818ada8d", "screen_result"),
    "task14_current_cached_v2_correction": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_numerical_repair_v2.json",
        "3516fd368b80676ee0554592f94c2dbacceaea0350e69e7be9a0bbdbb6c81c9c", "preregistration"),
    "task14_current_cached_v2_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2.py",
        "54debcbb3580f30b93402b0719db2d8b57a2c2923ee036d4c176851392bbabdc", "experiment_runner"),
    "task14_current_cached_v2_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2_result.json",
        "5749ccc87aa9631699da79a72504101651f6d9f91da9157d646e723151f52ca9", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, test_type: str, verdict: str, failure_kind: str | None,
           result_id: str, prereg_id: str, metrics: list[dict], notes: str,
           *, supersedes: str | None = None) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "invalid" if verdict == "invalid" else "complete",
        "verdict": verdict, "failure_kind": failure_kind, "family_ids": [],
        "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": prereg_id, "result_artifact_id": result_id,
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_current_cached_v2_capability_license",
            "task14_current_cached_v2_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": supersedes, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def build_plan() -> dict[str, Any]:
    artifacts, docs = {}, {}
    for artifact_id, (relative, expected, kind) in ARTIFACT_SPECS.items():
        path = BQ / relative
        actual = registry.file_sha256(path)
        if actual != expected:
            raise PublicationError(f"hash mismatch for {path.name}: {actual} != {expected}")
        artifacts[artifact_id] = {"path": str(path.relative_to(REPO)), "sha256": actual,
                                  "kind": kind, "status": "frozen"}
        if path.suffix == ".json":
            docs[artifact_id] = json.loads(path.read_text())

    record = json.loads(registry.circuit_path(TAG).read_text())
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM}:
        raise PublicationError("canonical Task14 latest claim is not v12/v13")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v12 is not the exact audited base")

    v1 = docs["task14_current_cached_v1_invalid_result"]
    correction = docs["task14_current_cached_v2_correction"]
    license_doc = docs["task14_current_cached_v2_capability_license"]
    result = docs["task14_current_cached_v2_result"]
    if v1.get("terminal") != "invalid" or v1["score"].get("value_branch_sum_max_absolute_error") != 0.000244140625:
        raise PublicationError("v1 invalid-instrument provenance changed")
    if correction["v1_provenance"].get("invalid_result_sha256") != ARTIFACT_SPECS["task14_current_cached_v1_invalid_result"][1]:
        raise PublicationError("correction does not bind v1")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("v2 license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("v2 terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("v2 split scope changed")
    expected = {
        "pred_a_instrument_live": True, "pred_b_current_branch_carries_task": True,
        "pred_c_cached_branch_carries_task": False, "pred_d_interaction_is_needed": False,
        "pred_e_lexical_leakage": False, "pred_f_number_specific": True,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    cells = list(score["cells"].values())
    current = [c["opposite_current_only"] for c in cells]
    cached = [c["opposite_cached_only"] for c in cells]
    joint = [c["opposite_both"] for c in cells]
    complete = [c["complete_opposite_head"] for c in cells]
    derived = [c["derived"] for c in cells]
    frac = lambda arm, key: [sum(v > 0 for v in c[arm][key]) / len(c[arm][key]) for c in cells]

    invalid_id = "task14_head11_3.fresh_current_cached.v1_instrument.invalid.v1"
    invalid = _event(
        invalid_id, "null_control", "invalid", "implementation_failure",
        "task14_current_cached_v1_invalid_result", "task14_current_cached_scientific_prior_art",
        [metric("value_branch_sum_max_absolute_error", v1["score"]["value_branch_sum_max_absolute_error"], "<=0.00005")],
        "Engineering-only invalid result: separately projecting and then adding the two branches changed floating-point operation order. Its scientific arms are not evidence.")
    # The corrected run binds the repair receipt as its preregistration.
    common = dict(result_id="task14_current_cached_v2_result", prereg_id="task14_current_cached_v2_correction")
    instrument = _event(
        "task14_head11_3.fresh_current_cached.v2_instrument.complete.v1", "null_control", "held", None,
        common["result_id"], common["prereg_id"],
        [metric("native_replay_max_absolute_logit_error", score["native_replay_max_absolute_logit_error"], "<=0.00007"),
         metric("source_term_sum_max_absolute_error", score["source_term_sum_max_absolute_error"], "<=0.00005"),
         metric("raw_effective_value_max_absolute_error", score["raw_effective_value_max_absolute_error"], "<=0.00005"),
         metric("projected_effective_value_max_absolute_error", score["projected_effective_value_max_absolute_error"], "<=0.00005"),
         metric("same_batch_native_noop_endpoint_max_absolute_error", score["same_batch_native_noop_endpoint_max_absolute_error"], "<=0.00007"),
         metric("installed_head_max_absolute_error", score["installed_head_max_absolute_error"], "<=0.00005"),
         metric("complete_head_vector_max_absolute_error", score["complete_head_vector_max_absolute_error"], "<=0.00005"),
         metric("minimum_complete_head_mean_margin", min(x["mean_margin"] for x in complete), ">=0.05 in every cell"),
         metric("minimum_joint_value_mean_margin", min(x["mean_margin"] for x in joint), ">=0.05 in every cell"),
         metric("minimum_joint_positive_row_fraction", min(frac("opposite_both", "margin_values") + frac("opposite_both", "CE_values")), ">=0.75 in every cell")],
        "The corrected native-order branch sum, no-op, complete-head control, and joint-value intervention passed.", supersedes=invalid_id)
    current_event = _event(
        "task14_head11_3.fresh_current_cached.current_branch_use.complete.v1", "composition", "held", None,
        common["result_id"], common["prereg_id"],
        [metric("current_mean_margin_range", [min(x["mean_margin"] for x in current), max(x["mean_margin"] for x in current)], ">=0.05 in every cell"),
         metric("current_mean_CE_range", [min(x["mean_CE"] for x in current), max(x["mean_CE"] for x in current)], ">=0 in every cell"),
         metric("minimum_current_positive_row_fraction", min(frac("opposite_current_only", "margin_values") + frac("opposite_current_only", "CE_values")), ">=0.75 in every cell"),
         metric("current_over_joint_margin_range", [min(x["current_margin_recovery"] for x in derived), max(x["current_margin_recovery"] for x in derived)], ">=0.70 in every cell")],
        "The opposite-number current-state value branch carried the answer-directed task effect in every licensed cell.")
    cached_event = _event(
        "task14_head11_3.fresh_current_cached.cached_branch_use.complete.v1", "composition", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("cached_mean_margin_range", [min(x["mean_margin"] for x in cached), max(x["mean_margin"] for x in cached)], ">=0.05 in every cell"),
         metric("cached_mean_CE_range", [min(x["mean_CE"] for x in cached), max(x["mean_CE"] for x in cached)], ">=0 in every cell"),
         metric("maximum_cached_positive_row_fraction", max(frac("opposite_cached_only", "margin_values") + frac("opposite_cached_only", "CE_values")), ">=0.75 in every cell"),
         metric("cached_over_joint_margin_range", [min(x["cached_margin_recovery"] for x in derived), max(x["cached_margin_recovery"] for x in derived)], ">=0.70 in every cell")],
        "The cached layer-0 value branch did not independently carry the answer-directed task effect.")
    interaction_event = _event(
        "task14_head11_3.fresh_current_cached.interaction_needed.complete.v1", "composition", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("current_over_joint_margin_range", [min(x["current_margin_recovery"] for x in derived), max(x["current_margin_recovery"] for x in derived)], "<=0.50 for each single branch"),
         metric("interaction_over_joint_margin_range", [min(x["interaction_margin_recovery"] for x in derived), max(x["interaction_margin_recovery"] for x in derived)], ">=0.50 in every cell"),
         metric("interaction_mean_margin_range", [min(x["interaction_mean_margin"] for x in derived), max(x["interaction_mean_margin"] for x in derived)], ">0 in every cell"),
         metric("interaction_mean_CE_range", [min(x["interaction_mean_CE"] for x in derived), max(x["interaction_mean_CE"] for x in derived)], ">0 in every cell")],
        "The registered interaction-needed prediction was null: current-only exceeded the joint effect and the interaction was small and negative.")
    lexical_event = _event(
        "task14_head11_3.fresh_current_cached.lexical_leakage.complete.v1", "invariance", "null", "scientific_null",
        common["result_id"], common["prereg_id"],
        [metric("maximum_lexical_over_number_joint_margin_ratio", max(x["maximum_lexical_over_number_joint_margin_ratio"] for x in derived), ">=0.50 in any cell")],
        "The registered lexical-leakage prediction was null: same-number different-lemma effects stayed below half of the number-change effect.")
    specificity_event = _event(
        "task14_head11_3.fresh_current_cached.number_specificity.complete.v1", "invariance", "held", None,
        common["result_id"], common["prereg_id"],
        [metric("maximum_lexical_over_number_joint_margin_ratio", max(x["maximum_lexical_over_number_joint_margin_ratio"] for x in derived), "<=0.25 in every cell")],
        "The registered number-specificity alternative held on these licensed rows: every same-number lexical contrast was small relative to the matched opposite-number joint effect.")
    events = [invalid, instrument, current_event, cached_event, interaction_event, lexical_event, specificity_event]

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 13, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed current-versus-cached subject-value factorial is complete and must not be repeated. With recipient p_8 and the native non-subject complement fixed, the current-state value branch carried the task effect, the cached layer-0 branch did not, interaction-needed was null, lexical leakage was null, and the registered number-specificity alternative held. Next test a finer causal split within the current-state value branch and identify downstream readers. Necessity/selective removal and genuinely new OOD constructions remain open."
        ),
    })
    if not any(s["site_id"] == SITE_ID for s in revision["candidate_sites"]):
        revision["candidate_sites"].append({
            "site_id": SITE_ID,
            "tensor_path": "the current-state and layer-0 cached branches of the exact L11H3 subject effective value u_8",
            "shape": ["batch", 2, 128],
            "intervention": "replace current and cached pre-projection value branches independently, sum in native order, then project once",
            "ceiling_event_ids": [],
        })
    family_id = "fresh_matched_subject_current_cached_value_v1"
    if not any(f["family_id"] == family_id for f in revision["counterfactual_families"]):
        revision["counterfactual_families"].append({
            "family_id": family_id, "role": "interchange",
            "changes": ["the subject current-state value branch, cached layer-0 value branch, or both using matched opposite-number or same-number lexical donors"],
            "holds_fixed": ["licensed HOLDOUT text", "recipient subject score p_8", "native non-subject source-term complement"],
            "builder_artifact_id": "task14_fresh_matched_capability_authority",
            "control_ids": ["native-order branch reconstruction and same-batch no-op", "complete opposite-head and joint-value positive controls"],
            "split_plan_id": SPLIT_ID, "status": "validated",
        })
    return {"schema": "task14_fresh_current_cached_value_publication_v13",
            "canonical_tag": TAG, "artifacts": artifacts, "events": events,
            "claim_revision": revision}


def _keyed(record: dict, event: dict) -> dict:
    out = copy.deepcopy(event)
    out["design_key"] = registry.design_key(record, out)
    out["execution_key"] = registry.execution_key(record, out)
    return out


def apply_plan(plan: dict, *, regenerate: bool = True) -> Path:
    path = registry.circuit_path(TAG)
    preview = json.loads(path.read_text())
    for artifact_id, artifact in plan["artifacts"].items():
        if artifact_id in preview["artifacts"] and preview["artifacts"][artifact_id] != artifact:
            raise PublicationError(f"artifact collision: {artifact_id}")
        preview["artifacts"][artifact_id] = artifact
    revision = plan["claim_revision"]
    found = [x for x in preview["claims"] if x["claim_id"] == NEW_CLAIM]
    if found and found != [revision]:
        raise PublicationError("claim revision collision")
    if not found:
        preview["claims"].append(revision)
    for event in plan["events"]:
        expected = _keyed(preview, event)
        found = [x for x in preview["evidence_events"] if x["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            preview["evidence_events"].append(expected)
    registry.validate_v2(preview)
    registry.append_artifacts(TAG, plan["artifacts"])
    for event in plan["events"]:
        current_record = json.loads(path.read_text())
        expected = _keyed(current_record, event)
        found = [x for x in current_record["evidence_events"] if x["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            registry.append_evidence_event(TAG, event)
    current_record = json.loads(path.read_text())
    if not any(x["claim_id"] == NEW_CLAIM for x in current_record["claims"]):
        registry.append_claim_revision(TAG, revision)
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
