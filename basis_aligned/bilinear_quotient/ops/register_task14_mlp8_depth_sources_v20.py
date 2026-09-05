#!/usr/bin/env python3
"""Publish the valid Task14 MLP8 E/A/U/V depth-source factorial as v20."""

from __future__ import annotations

import argparse
import ast
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
BASE_CLAIM = "grammatical_subject_number.v19"
NEW_CLAIM = "grammatical_subject_number.v20"
BASE_CLAIM_SHA256 = "eedf901a1203473bb99bcd5704e5b0e64885884cf519bdb90b2e8432a830f9ab"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "MLP.block8.subject_input.E_A_U_V_depth_sources.response_to_attention11_head3.final_position"

ARTIFACT_SPECS = {
    "task14_mlp8_depth_sources_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_capability_license.json",
        "b69c9548dc80342d9e374ea4d627630e5558012c6b98c392e757ac11a152e09e", "capability_license"),
    "task14_mlp8_depth_sources_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1.json",
        "03304b189bd8d1c98dc007543dd9f64253a91d23a3808546f8c13949ca8e1bbb", "preregistration"),
    "task14_mlp8_depth_sources_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial.py",
        "c1f132a2be9409157058531445fc1fac8602b8a53735e0dde4cac84c438c42cc", "experiment_runner"),
    "task14_mlp8_depth_sources_invalid_attempt": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_invalid_numeric_grouping_result.json",
        "d88d03097c62c7c9e495d682e224e287d2831404be568b54f26cb350e78ec7db", "screen_result"),
    "task14_mlp8_depth_sources_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_result.json",
        "429812569df68b1581f4f6632c704b8d034f65ed115c0f9f7d78ca8bb37ec817", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, verdict: str, failure_kind: str | None,
           result_id: str, metrics: list[dict], notes: str,
           *, supersedes: str | None = None) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM,
        "test_type": "null_control" if verdict == "invalid" or "instrument" in event_id else "composition",
        "stage": "invalid" if verdict == "invalid" else "complete",
        "verdict": verdict, "failure_kind": failure_kind, "family_ids": [],
        "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": "task14_mlp8_depth_sources_prior_art",
        "result_artifact_id": result_id,
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_mlp8_input_writers_result",
            "task14_mlp8_depth_sources_capability_license",
            "task14_mlp8_depth_sources_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": supersedes, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


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
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM} \
            and not any(c["claim_id"] == NEW_CLAIM for c in record["claims"]):
        raise PublicationError("canonical Task14 contains neither a v19 migration state nor v20")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v19 is not the exact audited base")

    invalid = docs["task14_mlp8_depth_sources_invalid_attempt"]
    license_doc = docs["task14_mlp8_depth_sources_capability_license"]
    result = docs["task14_mlp8_depth_sources_result"]
    if invalid.get("terminal") != "invalid" or invalid.get("score", {}).get("predictions", {}).get("pred_a_instrument_and_parent_closure") is not False:
        raise PublicationError("invalid attempt provenance changed")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_and_parent_closure": True, "pred_b_V_late_dominant": True,
        "pred_c_U_early_dominant": False, "pred_d_distributed_depth": False,
        "pred_e_cross_depth_composition": False, "pred_f_direction_switch": False,
        "pred_g_number_specific": True,
    }
    score = result["score"]
    if score.get("predictions") != expected or score.get("direction_winners") != {
            "plural_to_singular": "V", "singular_to_plural": "V"}:
        raise PublicationError("registered outcomes changed")

    invalid_id = "task14_head11_3.fresh_MLP8_depth_sources.numeric_grouping_attempt.invalid.v1"
    events = [_event(
        invalid_id, "invalid", "implementation_failure",
        "task14_mlp8_depth_sources_invalid_attempt",
        [metric("input_state_closure_max_absolute_error", invalid["score"]["input_state_closure_max_absolute_error"], "<=0.00005"),
         metric("parent_raw_state_max_absolute_error", invalid["score"]["parent_raw_state_max_absolute_error"], "<=0.00005"),
         metric("parent_MLP8_output_max_absolute_error", invalid["score"]["parent_MLP8_output_max_absolute_error"], "<=0.00005"),
         metric("parent_propagated_slot_max_absolute_error", invalid["score"]["parent_propagated_slot_max_absolute_error"], "<=0.00005")],
        "Engineering-only invalid attempt: regrouped float32 E+A+U+V evaluation failed frozen state and parent-reproduction bars. All scientific arms from this artifact are non-evidence.")]

    exact_names = [
        "native_replay_max_absolute_logit_error", "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error", "M_grouping_closure_max_absolute_error",
        "hybrid_endpoint_max_absolute_error", "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error", "output_closure_max_absolute_error",
        "propagated_endpoint_max_absolute_error", "gauge_invariance_max_absolute_error",
        "parent_raw_state_max_absolute_error", "parent_normalized_input_max_absolute_error",
        "parent_MLP8_output_max_absolute_error", "parent_propagated_slot_max_absolute_error",
        "parent_head_endpoint_max_absolute_error", "parent_installed_head_max_absolute_error",
        "parent_downstream_outcome_max_absolute_error", "downstream_state_closure_max_absolute_error",
        "downstream_normalized_closure_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    events.append(_event(
        "task14_head11_3.fresh_MLP8_depth_sources.instrument_parent_closure.complete.v1",
        "held", None, "task14_mlp8_depth_sources_result",
        [metric(n, score[n], "within frozen exactness/parent-reproduction bar") for n in exact_names],
        "The repaired E/A/U/V factorial passed every exactness and parent E/A/M reproduction gate. It supersedes only the invalid instrument attempt; the invalid scientific arms remain non-evidence.",
        supersedes=invalid_id))

    direction_stats = {
        direction: {ast.literal_eval(key): value for key, value in entries.items()}
        for direction, entries in score["direction_aggregate_recovery"].items()}
    def values(depth: str) -> list[float]:
        return [value for entries in direction_stats.values()
                for (component, outcome, aggregate, name), value in entries.items()
                if aggregate in ("M", "EM", "AM") and name == depth]
    v, u, v_only, u_only, uv = (values(name) for name in ("V", "U", "V_only", "U_only", "UV"))
    events.extend([
        _event("task14_head11_3.fresh_MLP8_depth_sources.V_late_dominant.complete.v1", "held", None,
               "task14_mlp8_depth_sources_result",
               [metric("V_aggregate_recovery_range", _range(v), ">=0.70 across directions/components/outcomes/M,EM,AM"),
                metric("U_only_aggregate_recovery_range", _range(u_only), "absolute value <=0.25 across the same cells")],
               "Late prior-MLP writes V=MLP4--7 were dominant under the preregistered aggregate criterion in both number directions."),
        _event("task14_head11_3.fresh_MLP8_depth_sources.U_early_dominant.complete.v1", "null", "scientific_null",
               "task14_mlp8_depth_sources_result",
               [metric("U_aggregate_recovery_range", _range(u), ">=0.70 across all registered aggregates"),
                metric("V_only_aggregate_recovery_range", _range(v_only), "absolute value <=0.25")],
               "Early prior-MLP writes U=MLP0--3 did not satisfy the registered dominance criterion."),
        _event("task14_head11_3.fresh_MLP8_depth_sources.distributed_depth.complete.v1", "null", "scientific_null",
               "task14_mlp8_depth_sources_result",
               [metric("V_aggregate_recovery_range", _range(v), "both depth groups materially contribute without a dominant group"),
                metric("U_aggregate_recovery_range", _range(u), "both depth groups materially contribute without a dominant group")],
               "The distributed-depth alternative was null because V passed the registered dominance criterion."),
        _event("task14_head11_3.fresh_MLP8_depth_sources.cross_depth_composition.complete.v1", "null", "scientific_null",
               "task14_mlp8_depth_sources_result",
               [metric("absolute_UV_interaction_recovery_range", _range([abs(x) for x in uv]), ">=0.25 for one component/aggregate across both directions and outcomes")],
               "Cross-depth U×V composition was null under the all-direction/outcome criterion; some individual interactions were large but no registered component/aggregate passed uniformly."),
        _event("task14_head11_3.fresh_MLP8_depth_sources.direction_switch.complete.v1", "null", "scientific_null",
               "task14_mlp8_depth_sources_result",
               [metric("direction_winners", score["direction_winners"], "different unique depth winners by direction")],
               "Direction switching was null: V was the unique registered depth winner for both plural-to-singular and singular-to-plural."),
        _event("task14_head11_3.fresh_MLP8_depth_sources.number_specificity.complete.v1", "held", None,
               "task14_mlp8_depth_sources_result",
               [metric("maximum_lexical_ratio", score["maximum_lexical_ratio"], "<=0.25 across every subset/component/outcome")],
               "Same-number different-lemma depth-source interventions remained below one quarter of opposite-number effects."),
    ])

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 20, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The valid licensed MLP8 E/A/U/V depth-source factorial is complete and must not be repeated. Its earlier float32 regrouping attempt remains a separate invalid instrument with non-evidence scientific arms. In the repaired exact run, V=MLP4--7 was the dominant source of MLP8's prior-MLP input effect in both number directions; U=MLP0--3 dominance, distributed depth, cross-depth composition, and direction switching were null, while number specificity held. E/A/U/V are operational native writer groups, not a unique semantic basis. Individual MLP4--7 identities, within-group semantic units, OOD replication, independent data, downstream readers, and necessity outside the fixed L11H3 interface remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "MLP8's subject-position input split into embedding/skip E, attention A0--8, early MLP U=MLP0--3, and late MLP V=MLP4--7, with exact response propagation to L11H3",
        "shape": ["batch", 4, 1152],
        "intervention": "factorially swap all 15 nonempty E/A/U/V subsets, compute exact cross/quadratic/full MLP8 responses, and install only through the fixed L11H3 subject-value interface",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_MLP8_E_A_U_V_depth_sources_v1",
        "role": "interchange",
        "changes": ["embedding/skip E", "attention writes A0--8", "early MLP writes U=MLP0--3", "late MLP writes V=MLP4--7", "all pair, triple, and four-way combinations"],
        "holds_fixed": ["licensed HOLDOUT text and subject position 8", "recipient other MLP4--10 downstream background", "recipient L11H3 p_8 and cached value", "native non-subject L11H3 source complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["exact E+A+U+V input closure", "exact U+V reproduction of parent M corners", "hybrid response and propagated endpoint closure", "gauge invariance and same-batch no-op", "same-number lexical subsets"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_mlp8_depth_sources_publication_v20",
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
        current = json.loads(path.read_text())
        expected = _keyed(current, event)
        found = [x for x in current["evidence_events"] if x["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            registry.append_evidence_event(TAG, event)
    current = json.loads(path.read_text())
    if not any(x["claim_id"] == NEW_CLAIM for x in current["claims"]):
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
