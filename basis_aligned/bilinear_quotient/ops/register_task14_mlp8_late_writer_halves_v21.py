#!/usr/bin/env python3
"""Publish the valid Task14 MLP8 E/A/U/W/X late-writer factorial as v21."""

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
BASE_CLAIM = "grammatical_subject_number.v20"
NEW_CLAIM = "grammatical_subject_number.v21"
BASE_CLAIM_SHA256 = "90f2296b9812614157a72c329a0d3cde08a0585977bf041a774cc6c12adee747"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "MLP.block8.subject_input.E_A_U_W_X_late_writer_halves.response_to_attention11_head3.final_position"

ARTIFACT_SPECS = {
    "task14_mlp8_late_writer_halves_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_mlp8_mlp4_7_source_factorial_v1_capability_license.json",
        "8fa2a31c03b34354f669e52eb6f26ec8d3ae754c1fc363cd7f2cb0ae5135f420", "capability_license"),
    "task14_mlp8_late_writer_halves_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial_v1.json",
        "85be30474a62b09b9d067a5ba2a4f526d2633196df11392653ca1a9edc4d3e85", "preregistration"),
    "task14_mlp8_late_writer_halves_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial.py",
        "45cc8396a26135e6f9cfd22b73e62a9e8ae7a6e0d0720baff606ad8c2a5988f5", "experiment_runner"),
    "task14_mlp8_late_writer_halves_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial_v1_result.json",
        "11d64cb3f3dca1b4d0d3bf50a1288c5503335e23eeb8c10754bc2907d8ee637f", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, verdict: str, failure_kind: str | None,
           metrics: list[dict], notes: str) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM,
        "test_type": "null_control" if "instrument" in event_id else "composition",
        "stage": "complete", "verdict": verdict, "failure_kind": failure_kind,
        "family_ids": [], "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": "task14_mlp8_late_writer_halves_prior_art",
        "result_artifact_id": "task14_mlp8_late_writer_halves_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_mlp8_input_writers_result",
            "task14_mlp8_depth_sources_result",
            "task14_mlp8_late_writer_halves_capability_license",
            "task14_mlp8_late_writer_halves_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
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
        artifacts[artifact_id] = {
            "path": str(path.relative_to(REPO)), "sha256": actual,
            "kind": kind, "status": "frozen",
        }
        if path.suffix == ".json":
            docs[artifact_id] = json.loads(path.read_text())

    record = json.loads(registry.circuit_path(TAG).read_text())
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM} \
            and not any(c["claim_id"] == NEW_CLAIM for c in record["claims"]):
        raise PublicationError("canonical Task14 contains neither a v20 migration state nor v21")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v20 is not the exact audited base")

    license_doc = docs["task14_mlp8_late_writer_halves_capability_license"]
    result = docs["task14_mlp8_late_writer_halves_result"]
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_X_mlp6_7_dominant": False,
        "pred_c_W_mlp4_5_dominant": False,
        "pred_d_distributed_within_V": True,
        "pred_e_WX_composition": False,
        "pred_f_direction_switch": False,
        "pred_g_number_specific": True,
    }
    score = result["score"]
    if score.get("predictions") != expected or score.get("direction_winners") != {
            "plural_to_singular": None, "singular_to_plural": None}:
        raise PublicationError("registered outcomes changed")

    direction_stats = {
        direction: {ast.literal_eval(key): value for key, value in entries.items()}
        for direction, entries in score["direction_aggregate_recovery"].items()
    }

    def values(source: str) -> list[float]:
        return [value for entries in direction_stats.values()
                for (component, outcome, aggregate, name), value in entries.items()
                if aggregate in ("M", "EM", "AM") and name == source]

    w, x, w_only, x_only, wx = (values(name) for name in ("W", "X", "W_only", "X_only", "WX"))
    exact_names = [
        "native_replay_max_absolute_logit_error", "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error", "M_grouping_closure_max_absolute_error",
        "V_grouping_closure_max_absolute_error", "hybrid_endpoint_max_absolute_error",
        "source_term_sum_max_absolute_error", "product_closure_max_absolute_error",
        "output_closure_max_absolute_error", "propagated_endpoint_max_absolute_error",
        "gauge_invariance_max_absolute_error", "parent_lattice_mobius_max_absolute_error",
        "parent_raw_state_max_absolute_error", "parent_normalized_input_max_absolute_error",
        "parent_MLP8_output_max_absolute_error", "parent_propagated_slot_max_absolute_error",
        "parent_head_endpoint_max_absolute_error", "parent_installed_head_max_absolute_error",
        "parent_downstream_outcome_max_absolute_error", "downstream_state_closure_max_absolute_error",
        "downstream_normalized_closure_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    events = [
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.instrument_parent_closure.complete.v1",
               "held", None,
               [metric(n, score[n], "within frozen exactness/parent-regrouping bar") for n in exact_names],
               "The E/A/U/W/X factorial exactly reproduced its native computation and the parent E/A/U/V lattice, where W=MLP4--5, X=MLP6--7, and V=W+X."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.X_mlp6_7_dominant.complete.v1",
               "null", "scientific_null",
               [metric("X_aggregate_recovery_range", _range(x), ">=0.70 across registered aggregates"),
                metric("W_only_aggregate_recovery_range", _range(w_only), "absolute value <=0.25")],
               "MLP6--7 did not satisfy the preregistered all-cell dominance criterion."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.W_mlp4_5_dominant.complete.v1",
               "null", "scientific_null",
               [metric("W_aggregate_recovery_range", _range(w), ">=0.70 across registered aggregates"),
                metric("X_only_aggregate_recovery_range", _range(x_only), "absolute value <=0.25")],
               "MLP4--5 did not satisfy the preregistered all-cell dominance criterion."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.distributed_within_V.complete.v1",
               "held", None,
               [metric("W_aggregate_recovery_range", _range(w), ">=0.25 for a registered aggregate in both metrics and directions"),
                metric("X_aggregate_recovery_range", _range(x), ">=0.25 for a registered aggregate in both metrics and directions")],
               "Both MLP4--5 and MLP6--7 made material contributions in both metrics and number directions, while neither half passed the dominance criterion. This is distribution across operational writer groups, not a semantic decomposition."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.WX_composition.complete.v1",
               "null", "scientific_null",
               [metric("WX_interaction_recovery_range", _range(wx), ">=0.25 for a registered aggregate in both metrics and directions")],
               "W×X interaction did not pass the preregistered cross-direction and cross-metric composition criterion, despite some sizable individual cells."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.direction_switch.complete.v1",
               "null", "scientific_null",
               [metric("direction_winners", score["direction_winners"], "different identified dominant halves by direction")],
               "No dominant half was identified in either direction, so there was no registered direction switch."),
        _event("task14_head11_3.fresh_MLP8_late_writer_halves.number_specificity.complete.v1",
               "held", None,
               [metric("maximum_lexical_ratio", score["maximum_lexical_ratio"], "<=0.25 across every subset/component/outcome")],
               "Same-number different-lemma interventions stayed below one quarter of opposite-number effects."),
    ]

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 21, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The valid licensed MLP8 E/A/U/W/X factorial is complete and must not be repeated. Splitting the previously dominant V=MLP4--7 group into W=MLP4--5 and X=MLP6--7 found neither half dominant, while distributed contribution across the two halves held. W×X composition and direction switching were null, and number specificity held. W and X are operational native writer groups, not unique semantic units. Individual MLP4--7 identities, within-MLP semantic units, OOD replication, independent data, downstream readers, and necessity outside the fixed L11H3 interface remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "MLP8 subject-position input split into E, A0--8, U=MLP0--3, W=MLP4--5, and X=MLP6--7, with exact response propagation to L11H3",
        "shape": ["batch", 5, 1152],
        "intervention": "factorially swap all 31 nonempty E/A/U/W/X subsets, compute exact cross/quadratic/full MLP8 responses, and install only through the fixed L11H3 subject-value interface",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_MLP8_E_A_U_W_X_late_writer_halves_v1",
        "role": "interchange",
        "changes": ["embedding/skip E", "attention writes A0--8", "early MLP writes U=MLP0--3", "middle MLP writes W=MLP4--5", "late MLP writes X=MLP6--7", "all pair and higher-order combinations"],
        "holds_fixed": ["licensed HOLDOUT text and subject position 8", "recipient other MLP4--10 downstream background", "recipient L11H3 p_8 and cached value", "native non-subject L11H3 source complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["exact E+A+U+W+X input closure", "exact W+X regrouping to the parent V lattice", "hybrid response and propagated endpoint closure", "gauge invariance and same-batch no-op", "same-number lexical subsets"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_mlp8_late_writer_halves_publication_v21",
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
