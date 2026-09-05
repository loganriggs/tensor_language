#!/usr/bin/env python3
"""Publish the corrected licensed Task14 upstream-writer factorial as v14."""

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
BASE_CLAIM = "grammatical_subject_number.v13"
NEW_CLAIM = "grammatical_subject_number.v14"
BASE_CLAIM_SHA256 = "8e8f39b688e79141e8663fe97e78d42469ea7bd0e72a881ce273ff8953f0aca2"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "attention.block11.head3.subject_current_value_input.upstream_writer_families.final_position"

ARTIFACT_SPECS = {
    "task14_upstream_writers_v1_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_upstream_writer_factorial_v1_capability_license.json",
        "f0d1ebba9a7bc8ad5ec9936913a1d2dc5daa4e6f1f72a0571746b4aacc5b9f40", "capability_license"),
    "task14_upstream_writers_v2_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_current_upstream_writer_factorial_v2_capability_license.json",
        "bf8f9a3b71846a06ebaf577421547eb19149345d9011e3bf5417bcbd7338abc6", "capability_license"),
    "task14_upstream_writers_scientific_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v1.json",
        "999884e00a3e730ccba9f60ae84aa99e55305cc9bcf537c77dbb2f971b6c7ea9", "preregistration"),
    "task14_upstream_writers_v1_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial.py",
        "1f9f8580350fe4f59c0f69ce329800d7390c7848d0954e0a5905a26cdf7af8ae", "experiment_runner"),
    "task14_upstream_writers_v1_invalid_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v1_result.json",
        "72e8ea00ea82e3f76e45c098f1e758ced0bf74c457d1432ed886eeedb8990518", "screen_result"),
    "task14_upstream_writers_v2_correction": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_numerical_repair_v2.json",
        "99befade2c755168e6eff45f2c3b58f1df556b709925eeff772bcf1d355606f4", "preregistration"),
    "task14_upstream_writers_v2_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2.py",
        "7966a5d46a76e7bdf87c04958647d2ba88792eb71ae402f84755d94e0f612569", "experiment_runner"),
    "task14_upstream_writers_v2_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2_result.json",
        "5c021cad2f73663f2176a813fc1f4ceffef555b48d7d00c050d0f60d0a2434fa", "screen_result"),
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
            "task14_current_cached_v2_result",
            "task14_upstream_writers_v2_capability_license",
            "task14_upstream_writers_v2_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": supersedes, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _ranges(cells: list[dict], arm: str) -> list[dict[str, float]]:
    return [c[arm] for c in cells]


def _range(items: list[dict[str, Any]], key: str) -> list[float]:
    values = [x[key] for x in items]
    return [min(values), max(values)]


def _positive_fractions(cells: list[dict], arm: str, key: str) -> list[float]:
    return [sum(v > 0 for v in c[arm][key]) / len(c[arm][key]) for c in cells]


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
    # Keep this historical publisher auditable after later revisions land.
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM} \
            and not any(c["claim_id"] == NEW_CLAIM for c in record["claims"]):
        raise PublicationError("canonical Task14 contains neither a v13 migration state nor v14")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v13 is not the exact audited base")

    v1 = docs["task14_upstream_writers_v1_invalid_result"]
    correction = docs["task14_upstream_writers_v2_correction"]
    license_doc = docs["task14_upstream_writers_v2_capability_license"]
    result = docs["task14_upstream_writers_v2_result"]
    if v1.get("terminal") != "invalid" or v1["score"].get("state_sum_max_absolute_error") != 0.00146484375:
        raise PublicationError("v1 invalid-instrument provenance changed")
    if correction["v1_provenance"].get("invalid_result_sha256") != ARTIFACT_SPECS["task14_upstream_writers_v1_invalid_result"][1]:
        raise PublicationError("correction does not bind v1")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("v2 license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("v2 terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("v2 split scope changed")
    expected = {
        "pred_a_instrument_live": True,
        "pred_b_embedding_carries_task": False,
        "pred_c_attention_carries_task": False,
        "pred_d_MLP_carries_task": True,
        "pred_e_distributed_across_writer_families": False,
        "pred_f_interaction_is_needed": False,
        "pred_g_number_specific": True,
        "pred_h_lexical_collateral": False,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    cells = list(score["cells"].values())
    arms = {name: _ranges(cells, f"opposite_{name}") for name in ("E", "A", "M")}
    all_donor = _ranges(cells, "opposite_EAM")
    recovery = {
        name: [c["derived"]["family_margin_recovery"][name] for c in cells]
        for name in ("E", "A", "M")
    }

    invalid_id = "task14_head11_3.fresh_upstream_writers.v1_instrument.invalid.v1"
    invalid = _event(
        invalid_id, "null_control", "invalid", "implementation_failure",
        "task14_upstream_writers_v1_invalid_result", "task14_upstream_writers_scientific_prior_art",
        [metric("state_sum_max_absolute_error", v1["score"]["state_sum_max_absolute_error"], "<=0.00005")],
        "Engineering-only invalid result: regrouping the sequential float32 residual additions as E+A+M exceeded the frozen exactness bar. V1 scientific arms are not evidence.")
    common = {
        "result_id": "task14_upstream_writers_v2_result",
        "prereg_id": "task14_upstream_writers_v2_correction",
    }
    instrument = _event(
        "task14_head11_3.fresh_upstream_writers.v2_instrument.complete.v1",
        "null_control", "held", None, common["result_id"], common["prereg_id"],
        [
            metric("native_replay_max_absolute_logit_error", score["native_replay_max_absolute_logit_error"], "<=0.00007"),
            metric("state_sum_max_absolute_error", score["state_sum_max_absolute_error"], "<=0.00005"),
            metric("normalized_state_max_absolute_error", score["normalized_state_max_absolute_error"], "<=0.00005"),
            metric("source_term_sum_max_absolute_error", score["source_term_sum_max_absolute_error"], "<=0.00005"),
            metric("all_donor_current_head_max_absolute_error", score["all_donor_current_head_max_absolute_error"], "<=0.00005"),
            metric("same_batch_native_noop_endpoint_max_absolute_error", score["same_batch_native_noop_endpoint_max_absolute_error"], "<=0.00007"),
            metric("installed_head_max_absolute_error", score["installed_head_max_absolute_error"], "<=0.00005"),
            metric("minimum_all_donor_mean_margin", min(x["mean_margin"] for x in all_donor), ">=0.05 in every cell"),
            metric("minimum_all_donor_mean_CE", min(x["mean_CE"] for x in all_donor), ">=0 in every cell"),
            metric("minimum_all_donor_positive_row_fraction", min(
                _positive_fractions(cells, "opposite_EAM", "margin_values")
                + _positive_fractions(cells, "opposite_EAM", "CE_values")), ">=0.75 in every cell"),
            metric("uncorrected_state_max_absolute_error_diagnostic", score["uncorrected_state_max_absolute_error"], "diagnostic only"),
        ],
        "The fixed-remainder v2 decomposition passed exact replay, recombination, installation, no-op, and live all-donor task controls. The numerical remainder is assigned to E by the frozen convention.",
        supersedes=invalid_id)

    family_events = []
    family_specs = {
        "E": ("embedding_skip_family", "null", "scientific_null",
              "The embedding/skip writer family did not reach the registered 70% recovery criterion in any licensed cell."),
        "A": ("earlier_attention_family", "null", "scientific_null",
              "The cumulative attention-write family from blocks 0--10 did not reach the registered 70% recovery criterion in any licensed cell."),
        "M": ("earlier_MLP_family", "held", None,
              "The cumulative MLP-write family from blocks 0--10 carried at least 70% of the all-donor margin effect in every licensed cell, with positive margin and CE on at least 3/4 rows per cell."),
    }
    for name, (label, verdict, failure, notes) in family_specs.items():
        family_events.append(_event(
            f"task14_head11_3.fresh_upstream_writers.{label}.complete.v1",
            "composition", verdict, failure, common["result_id"], common["prereg_id"],
            [
                metric(f"{name}_mean_margin_range", _range(arms[name], "mean_margin"), ">=0.05 in every cell"),
                metric(f"{name}_mean_CE_range", _range(arms[name], "mean_CE"), ">=0 in every cell"),
                metric(f"minimum_{name}_positive_row_fraction", min(
                    _positive_fractions(cells, f"opposite_{name}", "margin_values")
                    + _positive_fractions(cells, f"opposite_{name}", "CE_values")), ">=0.75 in every cell"),
                metric(f"{name}_over_all_donor_margin_range", [min(recovery[name]), max(recovery[name])], ">=0.70 in every cell"),
            ], notes))

    distributed = _event(
        "task14_head11_3.fresh_upstream_writers.distributed_families.complete.v1",
        "composition", "null", "scientific_null", common["result_id"], common["prereg_id"],
        [
            metric("E_over_all_donor_margin_range", [min(recovery["E"]), max(recovery["E"])], ">=0.25 in every cell"),
            metric("A_over_all_donor_margin_range", [min(recovery["A"]), max(recovery["A"])], ">=0.25 in every cell"),
            metric("M_over_all_donor_margin_range", [min(recovery["M"]), max(recovery["M"])], "<0.70 in every cell"),
        ],
        "The registered distributed-family account was null: M alone exceeded 70% recovery in every cell, while E and A did not provide two families above 25% in every cell.")
    total_interaction_recovery = [c["derived"]["total_interaction_margin_recovery"] for c in cells]
    total_interaction_margin = [c["derived"]["total_interaction_mean_margin"] for c in cells]
    total_interaction_ce = [c["derived"]["total_interaction_mean_CE"] for c in cells]
    interaction = _event(
        "task14_head11_3.fresh_upstream_writers.interaction_needed.complete.v1",
        "composition", "null", "scientific_null", common["result_id"], common["prereg_id"],
        [
            metric("maximum_M_over_all_donor_margin", max(recovery["M"]), "<=0.50 in every cell"),
            metric("total_interaction_over_all_donor_margin_range", [min(total_interaction_recovery), max(total_interaction_recovery)], ">=0.50 in every cell"),
            metric("total_interaction_mean_margin_range", [min(total_interaction_margin), max(total_interaction_margin)], ">0 with >=0.75 helpful rows in every cell"),
            metric("total_interaction_mean_CE_range", [min(total_interaction_ce), max(total_interaction_ce)], ">0 with >=0.75 helpful rows in every cell"),
        ],
        "The registered interaction-needed account was null: M alone exceeded 50%, while total interactions were small and changed sign across cells.")
    lexical_margin = [c["derived"]["maximum_lexical_margin_ratio"] for c in cells]
    lexical_ce = [c["derived"]["maximum_lexical_CE_ratio"] for c in cells]
    specificity = _event(
        "task14_head11_3.fresh_upstream_writers.number_specificity.complete.v1",
        "invariance", "held", None, common["result_id"], common["prereg_id"],
        [
            metric("maximum_lexical_margin_ratio", max(lexical_margin), "<=0.25 in every cell"),
            metric("maximum_lexical_CE_ratio", max(lexical_ce), "<=0.25 in every cell"),
        ],
        "The registered number-specificity alternative held: all same-number different-lemma writer-family effects were small relative to the corresponding opposite-number all-donor effect.")
    collateral = _event(
        "task14_head11_3.fresh_upstream_writers.lexical_collateral.complete.v1",
        "invariance", "null", "scientific_null", common["result_id"], common["prereg_id"],
        [
            metric("maximum_lexical_margin_ratio", max(lexical_margin), ">=0.50 in any cell"),
            metric("maximum_lexical_CE_ratio", max(lexical_ce), ">=0.50 in any cell"),
        ],
        "The registered lexical-collateral prediction was null: no same-number different-lemma effect reached half of its opposite-number reference effect.")
    events = [invalid, instrument, *family_events, distributed, interaction, specificity, collateral]

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 14, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed upstream-writer factorial is complete and must not be repeated. With recipient p_8, cached value, and native non-subject complement fixed, cumulative MLP writes from blocks 0--10 carried the current-state subject-value transfer; embedding/skip, cumulative earlier-attention, distributed-family, interaction-needed, and lexical-collateral predictions were null, while number-specificity held. This localizes a broad writer family only. It does not establish necessity, new syntax, downstream readers, or any individual MLP layer. Next causally split the cumulative MLP family without treating native MLP boundaries as final semantic units, and identify the downstream readers of the resulting computation."
        ),
    })
    if not any(s["site_id"] == SITE_ID for s in revision["candidate_sites"]):
        revision["candidate_sites"].append({
            "site_id": SITE_ID,
            "tensor_path": "the E/A/M decomposition of the pre-attention-11 subject residual that feeds the L11H3 current-state value branch",
            "shape": ["batch", 3, 1152],
            "intervention": "take recipient or matched donor E, A, and M families factorially at the subject position, add the frozen E-owned numerical remainder, apply native RMSNorm and V11H3, then install only the subject source term",
            "ceiling_event_ids": [],
        })
    family_id = "fresh_matched_subject_current_upstream_writer_families_v1"
    if not any(f["family_id"] == family_id for f in revision["counterfactual_families"]):
        revision["counterfactual_families"].append({
            "family_id": family_id, "role": "interchange",
            "changes": ["the embedding/skip, cumulative attention-write, and cumulative MLP-write contributions to the subject pre-attention-11 state, singly and factorially"],
            "holds_fixed": ["licensed HOLDOUT text", "recipient subject score p_8", "recipient cached layer-0 value branch", "native non-subject source-term complement"],
            "builder_artifact_id": "task14_fresh_matched_capability_authority",
            "control_ids": ["fixed-remainder exact state closure", "same-batch no-op", "all-donor current-head reproduction", "same-number different-lemma controls"],
            "split_plan_id": SPLIT_ID, "status": "validated",
        })
    return {
        "schema": "task14_fresh_upstream_writers_publication_v14",
        "canonical_tag": TAG, "artifacts": artifacts, "events": events,
        "claim_revision": revision,
    }


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
