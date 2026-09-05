#!/usr/bin/env python3
"""Publish the licensed Task14 conditional MLP4--10 layer screen as v16."""

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
BASE_CLAIM = "grammatical_subject_number.v15"
NEW_CLAIM = "grammatical_subject_number.v16"
BASE_CLAIM_SHA256 = "90736cba7225c05030931fd2c3fd561df18eb8df1a8672c4f12a9cc48927b98d"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "attention.block11.head3.subject_current_value_input.upstream_MLP4_10_layers.final_position"

ARTIFACT_SPECS = {
    "task14_mlp4_10_conditional_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1_capability_license.json",
        "672c7b309d5ed623d141d2bd6e1673ad7055133f715f1cb192acad34ca2a769c", "capability_license"),
    "task14_mlp4_10_conditional_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1.json",
        "de8d1dc9ba4fe13b200540ee1df43f43d20a80c89361c1c61e0eb5903905b312", "preregistration"),
    "task14_mlp4_10_conditional_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen.py",
        "a7555a33150a024730f44b1dc7d7a54beff46058e032a216e0abab0e855b02f9", "experiment_runner"),
    "task14_mlp4_10_conditional_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1_result.json",
        "5ca8b1ee5b23aad32e5fda9a3b4650c20c230228989ebd310c0804dfc695cba2", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, verdict: str, failure_kind: str | None,
           metrics: list[dict], notes: str, *, test_type: str = "composition") -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": verdict, "failure_kind": failure_kind,
        "family_ids": [], "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": "task14_mlp4_10_conditional_prior_art",
        "result_artifact_id": "task14_mlp4_10_conditional_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_mlp_depth_result",
            "task14_mlp4_10_conditional_capability_license",
            "task14_mlp4_10_conditional_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def _layer_ranges(cells: list[dict], layer: int) -> dict[str, list[float]]:
    key = str(layer)
    return {
        f"standalone_{out}": _range([
            c["derived"]["layers"][key]["standalone_recovery"][out] for c in cells])
        for out in ("margin", "CE")
    } | {
        f"conditional_{out}": _range([
            c["derived"]["layers"][key]["conditional_recovery"][out] for c in cells])
        for out in ("margin", "CE")
    } | {
        f"context_difference_{out}": _range([
            c["derived"]["layers"][key]["context_difference"][out] for c in cells])
        for out in ("margin", "CE")
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
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM} \
            and not any(c["claim_id"] == NEW_CLAIM for c in record["claims"]):
        raise PublicationError("canonical Task14 contains neither a v15 migration state nor v16")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v15 is not the exact audited base")

    license_doc = docs["task14_mlp4_10_conditional_capability_license"]
    result = docs["task14_mlp4_10_conditional_result"]
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_live": True,
        "pred_b_at_least_one_standalone_layer": True,
        "pred_c_at_least_one_conditional_layer": True,
        "pred_d_same_layer_is_stable": True,
        "pred_e_context_dependence": False,
        "pred_f_number_specific": True,
        "pred_g_lexical_collateral": False,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    expected_layers = {str(i): {"standalone": i == 8, "conditional": i == 8}
                       for i in range(4, 11)}
    if score.get("layer_predictions") != expected_layers:
        raise PublicationError("per-layer outcomes changed")
    cells = list(score["cells"].values())
    exact_names = [
        "native_replay_max_absolute_logit_error", "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error", "source_term_sum_max_absolute_error",
        "recipient_high_group_max_absolute_error", "donor_high_group_max_absolute_error",
        "full_donor_current_head_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    full_margin = _range([c["opposite_full_M4_10"]["mean_margin"] for c in cells])
    full_ce = _range([c["opposite_full_M4_10"]["mean_CE"] for c in cells])
    events = [_event(
        "task14_head11_3.fresh_MLP4_10_conditional.instrument.complete.v1", "held", None,
        [metric(n, score[n], "within frozen exactness bar") for n in exact_names] + [
            metric("full_donor_mean_margin_range", full_margin, ">=0.05 in every cell"),
            metric("full_donor_mean_CE_range", full_ce, ">=0 in every cell")],
        "All exact replay, state/group closure, source summation, no-op, installation, and full-donor task controls passed.",
        test_type="null_control")]

    r8 = _layer_ranges(cells, 8)
    events.extend([
        _event("task14_head11_3.fresh_MLP4_10_conditional.at_least_one_standalone.complete.v1", "held", None,
               [metric("MLP8_standalone_margin_recovery_range", r8["standalone_margin"], ">=0.25 in every cell"),
                metric("MLP8_standalone_CE_recovery_range", r8["standalone_CE"], ">=0.25 in every cell")],
               "MLP8 alone passed the registered 25% recovery and row-sign bars for both task outcomes in every licensed cell."),
        _event("task14_head11_3.fresh_MLP4_10_conditional.at_least_one_conditional.complete.v1", "held", None,
               [metric("MLP8_conditional_margin_recovery_range", r8["conditional_margin"], ">=0.25 in every cell"),
                metric("MLP8_conditional_CE_recovery_range", r8["conditional_CE"], ">=0.25 in every cell")],
               "Removing MLP8 from the full donor MLP4--10 set also passed the registered conditional-effect bars in every cell."),
        _event("task14_head11_3.fresh_MLP4_10_conditional.same_layer_stable.complete.v1", "held", None,
               [metric("MLP8_standalone_margin_recovery_range", r8["standalone_margin"], ">=0.25 in every cell"),
                metric("MLP8_conditional_margin_recovery_range", r8["conditional_margin"], ">=0.25 in every cell"),
                metric("MLP8_standalone_CE_recovery_range", r8["standalone_CE"], ">=0.25 in every cell"),
                metric("MLP8_conditional_CE_recovery_range", r8["conditional_CE"], ">=0.25 in every cell")],
               "MLP8 was the same passing native-layer localization handle in standalone and full-minus-layer views."),
    ])

    max_context_margin = max(max(_layer_ranges(cells, i)["context_difference_margin"])
                             for i in range(4, 11))
    max_context_ce = max(max(_layer_ranges(cells, i)["context_difference_CE"])
                         for i in range(4, 11))
    events.append(_event(
        "task14_head11_3.fresh_MLP4_10_conditional.context_dependence.complete.v1", "null", "scientific_null",
        [metric("maximum_context_difference_margin_fraction", max_context_margin, ">=0.25 in every cell for one layer"),
         metric("maximum_context_difference_CE_fraction", max_context_ce, ">=0.25 in every cell for one layer")],
        "No layer's standalone-versus-conditional difference met the registered context-dependence bar."))

    lexical_margin = [c["derived"]["maximum_lexical_ratio"]["margin"] for c in cells]
    lexical_ce = [c["derived"]["maximum_lexical_ratio"]["CE"] for c in cells]
    events.extend([
        _event("task14_head11_3.fresh_MLP4_10_conditional.number_specificity.complete.v1", "held", None,
               [metric("maximum_lexical_margin_ratio", max(lexical_margin), "<=0.25 in every cell"),
                metric("maximum_lexical_CE_ratio", max(lexical_ce), "<=0.25 in every cell")],
               "Same-number different-lemma interventions stayed small relative to opposite-number MLP4--10 transfer.", test_type="invariance"),
        _event("task14_head11_3.fresh_MLP4_10_conditional.lexical_collateral.complete.v1", "null", "scientific_null",
               [metric("maximum_lexical_margin_ratio", max(lexical_margin), ">=0.50 in any cell"),
                metric("maximum_lexical_CE_ratio", max(lexical_ce), ">=0.50 in any cell")],
               "The preregistered lexical-collateral alternative was null.", test_type="invariance"),
    ])

    for layer in range(4, 11):
        ranges = _layer_ranges(cells, layer)
        if layer == 8:
            verdict, failure = "held", None
            note = "MLP8 passed both registered views across every licensed cell. This localizes a native layer handle only; it does not identify a within-MLP semantic unit."
        elif layer == 10:
            verdict, failure = "inconclusive", None
            note = "MLP10 crossed the 25% bar in some cells or outcomes but not every cell in either view, so its evidence is mixed rather than a passing localization."
        else:
            verdict, failure = "null", "scientific_null"
            note = f"MLP{layer} did not pass the registered all-cell criterion in either standalone or conditional view."
        events.append(_event(
            f"task14_head11_3.fresh_MLP4_10_conditional.MLP{layer}.complete.v1", verdict, failure,
            [metric(f"MLP{layer}_{name}_recovery_range", value,
                    ">=0.25 in every cell" if not name.startswith("context") else "difference diagnostic")
             for name, value in ranges.items()], note))

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 16, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed conditional MLP4--10 screen is complete and must not be repeated. MLP8 was the only native-layer handle that passed both standalone and full-minus-layer criteria in every cell. MLP10 showed mixed sub-threshold/cell-dependent evidence; MLP4--7 and MLP9 were null under the registered localization bars. Context-dependence and lexical-collateral predictions were null, while number-specificity held. This is native-layer localization, not a claim that all of MLP8 is one semantic circuit. Within-MLP splitting/grouping, necessity, OOD syntax, and downstream readers remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "the exact propagated writes of MLP4 through MLP10 in the pre-attention-11 subject residual",
        "shape": ["batch", 7, 1152],
        "intervention": "swap one MLP write alone or measure its full-donor-minus-leave-one effect while recipient E+A+R, MLP0--3+MR, p_8, cached value, and non-subject terms remain fixed",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_current_upstream_MLP4_10_conditional_layers_v1",
        "role": "interchange",
        "changes": ["one exact propagated MLP4--10 write alone", "the full opposite-number MLP4--10 set with one layer left recipient"],
        "holds_fixed": ["licensed HOLDOUT text", "recipient embedding/skip, attention, and state remainder", "recipient MLP0--3 plus grouping remainder", "recipient p_8, cached value, and native non-subject complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["exact recipient/donor high-group closure", "same-batch no-op", "full-donor current-head reproduction", "same-number different-lemma controls"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_fresh_mlp4_10_conditional_publication_v16",
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
