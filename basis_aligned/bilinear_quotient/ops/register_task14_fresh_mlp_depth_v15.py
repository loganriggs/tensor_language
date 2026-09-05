#!/usr/bin/env python3
"""Publish the licensed Task14 upstream-MLP depth factorial as v15."""

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
BASE_CLAIM = "grammatical_subject_number.v14"
NEW_CLAIM = "grammatical_subject_number.v15"
BASE_CLAIM_SHA256 = "0df74b01fbe981bc2ad85360e70af4ca4beddf09eb6727f11ff557da3ad18de4"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "attention.block11.head3.subject_current_value_input.upstream_MLP_depth_groups.final_position"

ARTIFACT_SPECS = {
    "task14_mlp_depth_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_mlp_depth_factorial_v1_capability_license.json",
        "33679fc9b7a2ec9f0e59c640975ebe7bae3372a31f143b6ac7f31c8797e72f82", "capability_license"),
    "task14_mlp_depth_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1.json",
        "7b7243e9f94a35b376f3838f0353f1a9e09db87d631675dede07472d232ee7b5", "preregistration"),
    "task14_mlp_depth_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial.py",
        "e57cd41a10e567a21b956f0095232d6c53da85eddaf1d37a853d3953b7e32dbf", "experiment_runner"),
    "task14_mlp_depth_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1_result.json",
        "f18ce285152a1b3388ca3083de344dd9dd3675e4932b27bf44ea1f9e8e3f3073", "screen_result"),
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
        "prereg_artifact_id": "task14_mlp_depth_prior_art",
        "result_artifact_id": "task14_mlp_depth_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_upstream_writers_v2_result",
            "task14_mlp_depth_capability_license",
            "task14_mlp_depth_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def _mean_range(cells: list[dict], arm: str, key: str) -> list[float]:
    return _range([c[arm][f"mean_{key}"] for c in cells])


def _minimum_positive_fraction(cells: list[dict], arm: str, key: str) -> float:
    return min(sum(v > 0 for v in c[arm][f"{key}_values"]) /
               len(c[arm][f"{key}_values"]) for c in cells)


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
        raise PublicationError("canonical Task14 contains neither a v14 migration state nor v15")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v14 is not the exact audited base")

    license_doc = docs["task14_mlp_depth_capability_license"]
    result = docs["task14_mlp_depth_result"]
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_live": True,
        "pred_b_G0_carries_task": False,
        "pred_c_G1_carries_task": False,
        "pred_d_G2_carries_task": False,
        "pred_e_distributed_across_depth_groups": False,
        "pred_f_interaction_is_needed": False,
        "pred_g_number_specific": True,
        "pred_h_lexical_collateral": False,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    cells = list(score["cells"].values())

    exact_names = [
        "native_replay_max_absolute_logit_error", "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error", "source_term_sum_max_absolute_error",
        "recipient_grouped_M_max_absolute_error", "all_donor_grouped_M_max_absolute_error",
        "all_donor_current_head_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    all_donor = "opposite_G012"
    events = [_event(
        "task14_head11_3.fresh_MLP_depth.instrument.complete.v1", "held", None,
        [metric(name, score[name], "within frozen exactness bar") for name in exact_names] + [
            metric("all_donor_mean_margin_range", _mean_range(cells, all_donor, "margin"), ">=0.05 in every cell"),
            metric("all_donor_mean_CE_range", _mean_range(cells, all_donor, "CE"), ">=0 in every cell"),
            metric("minimum_all_donor_positive_margin_row_fraction", _minimum_positive_fraction(cells, all_donor, "margin"), ">=0.75 in every cell"),
            metric("minimum_all_donor_positive_CE_row_fraction", _minimum_positive_fraction(cells, all_donor, "CE"), ">=0.75 in every cell"),
        ],
        "The exact MLP0--10 depth grouping passed every replay, state-closure, grouped-write, no-op, installation, and live all-donor control.",
        test_type="null_control")]

    for group, layers in (("G0", "MLP0--3"), ("G1", "MLP4--7"), ("G2", "MLP8--10")):
        margin_recovery = [c["derived"]["group_recovery"][group]["margin"] for c in cells]
        ce_recovery = [c["derived"]["group_recovery"][group]["CE"] for c in cells]
        events.append(_event(
            f"task14_head11_3.fresh_MLP_depth.{group}.complete.v1", "null", "scientific_null",
            [
                metric(f"{group}_mean_margin_range", _mean_range(cells, f"opposite_{group}", "margin"), ">=0.05 in every cell"),
                metric(f"{group}_mean_CE_range", _mean_range(cells, f"opposite_{group}", "CE"), ">=0 in every cell"),
                metric(f"{group}_margin_recovery_range", _range(margin_recovery), ">=0.70 in every cell"),
                metric(f"{group}_CE_recovery_range", _range(ce_recovery), ">=0.70 in every cell"),
                metric(f"minimum_{group}_positive_margin_row_fraction", _minimum_positive_fraction(cells, f"opposite_{group}", "margin"), ">=0.75 in every cell"),
                metric(f"minimum_{group}_positive_CE_row_fraction", _minimum_positive_fraction(cells, f"opposite_{group}", "CE"), ">=0.75 in every cell"),
            ],
            f"{group} ({layers}) did not meet the registered 70% recovery bar in every licensed cell; this is a null for coarse-group sufficiency, not a claim that the layers are unused."))

    events.append(_event(
        "task14_head11_3.fresh_MLP_depth.distributed_groups.complete.v1", "null", "scientific_null",
        [
            metric("minimum_G1_margin_recovery", min(c["derived"]["group_recovery"]["G1"]["margin"] for c in cells), ">=0.25 in every cell for at least two groups"),
            metric("minimum_G1_CE_recovery", min(c["derived"]["group_recovery"]["G1"]["CE"] for c in cells), ">=0.25 in every cell for at least two groups"),
            metric("minimum_G2_margin_recovery", min(c["derived"]["group_recovery"]["G2"]["margin"] for c in cells), ">=0.25 in every cell for at least two groups"),
            metric("minimum_G2_CE_recovery", min(c["derived"]["group_recovery"]["G2"]["CE"] for c in cells), ">=0.25 in every cell for at least two groups"),
        ],
        "The preregistered distributed-depth account was null: only G2 cleared 25% on both outcomes in every cell; G1 missed that all-cell condition."))

    interaction_margin = [c["derived"]["total_interaction_recovery"]["margin"] for c in cells]
    interaction_ce = [c["derived"]["total_interaction_recovery"]["CE"] for c in cells]
    events.append(_event(
        "task14_head11_3.fresh_MLP_depth.interaction_needed.complete.v1", "null", "scientific_null",
        [
            metric("total_interaction_margin_recovery_range", _range(interaction_margin), ">=0.50 in every cell"),
            metric("total_interaction_CE_recovery_range", _range(interaction_ce), ">=0.50 in every cell"),
        ],
        "The preregistered interaction-needed account was null: non-additive terms recovered only a small fraction of the complete effect."))

    lexical_margin = [c["derived"]["maximum_lexical_ratio"]["margin"] for c in cells]
    lexical_ce = [c["derived"]["maximum_lexical_ratio"]["CE"] for c in cells]
    events.extend([
        _event(
            "task14_head11_3.fresh_MLP_depth.number_specificity.complete.v1", "held", None,
            [metric("maximum_lexical_margin_ratio", max(lexical_margin), "<=0.25 in every cell"),
             metric("maximum_lexical_CE_ratio", max(lexical_ce), "<=0.25 in every cell")],
            "Same-number different-lemma depth-group effects stayed small relative to the opposite-number all-group effect.",
            test_type="invariance"),
        _event(
            "task14_head11_3.fresh_MLP_depth.lexical_collateral.complete.v1", "null", "scientific_null",
            [metric("maximum_lexical_margin_ratio", max(lexical_margin), ">=0.50 in any cell"),
             metric("maximum_lexical_CE_ratio", max(lexical_ce), ">=0.50 in any cell")],
            "The preregistered lexical-collateral alternative was null.", test_type="invariance"),
    ])

    g12_margin = [c["opposite_G12"]["mean_margin"] / c[all_donor]["mean_margin"] for c in cells]
    g12_ce = [c["opposite_G12"]["mean_CE"] / c[all_donor]["mean_CE"] for c in cells]
    events.append(_event(
        "task14_head11_3.fresh_MLP_depth.G1_G2_exploratory.complete.v1", "inconclusive", None,
        [metric("G1_G2_margin_recovery_range", _range(g12_margin), "exploratory; no preregistered decision bar"),
         metric("G1_G2_CE_recovery_range", _range(g12_ce), "exploratory; no preregistered decision bar")],
        "Mixed evidence only: the observed G1+G2 corner recovered 93.6--95.2% of the full effect, but G1+G2 was not registered as a carries-task prediction. This is motivation for a new conditional split, not a retroactive held claim."))

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 15, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed upstream-MLP depth factorial is complete and must not be repeated. No single coarse group was sufficient under the registered all-cell bar; the distributed and interaction-needed accounts were also null, while number-specificity held and lexical collateral was null. G1+G2 recovery of 0.936--0.952 is exploratory because that pair was not registered as a carries-task claim. The screen supports a mostly additive MLP4--10 path dominated by MLP8--10, not an individual-MLP circuit. Next preregister conditional per-layer effects within MLP4--10. Necessity, OOD syntax, downstream readers, and individual MLPs remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "the exact MLP0--10 writes in the pre-attention-11 subject residual, grouped as MLP0--3, MLP4--7, and MLP8--10",
        "shape": ["batch", 3, 1152],
        "intervention": "take recipient or matched donor depth-group writes factorially, with the numerical grouping remainder assigned to G0, then run the native RMSNorm, V11H3 current branch, and suffix",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_current_upstream_MLP_depth_groups_v1",
        "role": "interchange",
        "changes": ["the exact propagated writes of MLP0--3, MLP4--7, and MLP8--10, singly and factorially"],
        "holds_fixed": ["licensed HOLDOUT text", "recipient embedding/skip, accumulated attention, and state remainder", "recipient subject score p_8", "recipient cached value and native non-subject complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["exact recipient and donor grouped-MLP closure", "same-batch no-op", "all-donor current-head reproduction", "same-number different-lemma controls"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_fresh_mlp_depth_publication_v15", "canonical_tag": TAG,
            "artifacts": artifacts, "events": events, "claim_revision": revision}


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
