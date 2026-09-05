#!/usr/bin/env python3
"""Publish the Task14 grouped MLP6--7 split and quadratic program as v22."""

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
BASE_CLAIM = "grammatical_subject_number.v21"
NEW_CLAIM = "grammatical_subject_number.v22"
BASE_CLAIM_SHA256 = "345c231a4e7f1de0cd64edbf74ddfafff90a3d3139e19ffedbe8d3d09a59347f"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"

ARTIFACT_SPECS = {
    "task14_mlp6_7_split_prior_art": ("circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1.json", "94a41203cd8b9176cad2323d353cba4c8b26d369a83598298ae178ffdd211c96", "preregistration"),
    "task14_mlp6_7_split_oom_amendment": ("circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1_oom_retry_amendment.json", "d1b559439472d939acedad38eaebc21d196fdaa4a23b628017ed1dd3f8081008", "preregistration_amendment"),
    "task14_mlp6_7_split_runner": ("ops/run_task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial.py", "5f06982a615209deef7c5608dd26afb10cb7a4a6799646ad42772705d84dbf88", "experiment_runner"),
    "task14_mlp6_7_split_result": ("circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1_result.json", "eff2b9e7ab76b4335733e8bde6708435a18e6b0da14073373e211d9588994efa", "screen_result"),
    "task14_mlp6_7_tangent_prior_art": ("circuits/prior_art/task14_mlp6_7_contextual_midpoint_tangent_readout_v1.json", "2a51f5d3acaf07ebcb115eecc0a9636cd81b7dafaf421bd2df28f3a18e1453a8", "preregistration"),
    "task14_mlp6_7_tangent_runner": ("ops/run_task14_mlp6_7_contextual_midpoint_tangent_readout.py", "43232401ebd7f0ee03d8d5cdb57b0a2452ec8a4abb253665448198913d906aac", "experiment_runner"),
    "task14_mlp6_7_tangent_result": ("circuits/fast_screens/task14_mlp6_7_contextual_midpoint_tangent_readout_v1_result.json", "48c72ea08c2573d520e639bbd34805ce6b60f4ec10d420fbbebed8e6112a65aa", "screen_result"),
    "task14_mlp6_7_gain_prior_art": ("circuits/prior_art/task14_mlp6_7_quadratic_gain_manipulation_v1.json", "5a96aa03a9b412c63a437376b6d9bf055cb80196710f98c5a1781a25c67b518b", "preregistration"),
    "task14_mlp6_7_gain_runner": ("ops/run_task14_mlp6_7_quadratic_gain_manipulation.py", "bc5087cab603a8dae4194f97dae6bf10096561f8a6c6d98a38451e4262102bd5", "experiment_runner"),
    "task14_mlp6_7_gain_result": ("circuits/fast_screens/task14_mlp6_7_quadratic_gain_manipulation_v1_result.json", "5285b484a33814fce50a90e905adba4550b46c12eda45ac20ad0b42a8bd84ffc", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, verdict: str, failure_kind: str | None, test_type: str,
           site_id: str, result_id: str, prereg_id: str,
           metrics: list[dict], notes: str) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": verdict, "failure_kind": failure_kind,
        "family_ids": [], "site_id": site_id, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": prereg_id, "result_artifact_id": result_id,
        "input_artifact_ids": [prereg_id, result_id], "seed": None,
        "checkpoint_sha256": CHECKPOINT, "supersedes_event_id": None,
        "replicates_event_id": None, "sections": [], "notes": notes,
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
        if kind == "screen_result":
            docs[artifact_id] = json.loads(path.read_text())

    record = json.loads(registry.circuit_path(TAG).read_text())
    base = next((c for c in record["claims"] if c["claim_id"] == BASE_CLAIM), None)
    if base is None or registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v21 is not the exact audited base")
    split = docs["task14_mlp6_7_split_result"]
    tangent = docs["task14_mlp6_7_tangent_result"]
    gain = docs["task14_mlp6_7_gain_result"]
    for result, expected_split in ((split, "LICENSED_HOLDOUT"),
                                   (tangent, "LICENSED_HOLDOUT_REUSED_TEXT"),
                                   (gain, "LICENSED_HOLDOUT_REUSED_TEXT")):
        if result.get("terminal") != "valid_causal_screen" \
                or result.get("checkpoint_weights_sha256") != CHECKPOINT \
                or result.get("evaluated_splits") != [expected_split] \
                or result.get("forbidden_splits_opened") != []:
            raise PublicationError("terminal, checkpoint, or split scope changed")
    expected_split_predictions = {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_Z_mlp7_dominant": False, "pred_c_Y_mlp6_dominant": False,
        "pred_d_distributed_mlp6_7": True, "pred_e_YZ_composition": True,
        "pred_f_direction_switch": True, "pred_g_number_specific": False}
    expected_tangent_predictions = {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_midpoint_quadratic_readout": True,
        "pred_c_endpoint_local_readout": False,
        "pred_d_material_nonquadratic_transport": False,
        "pred_e_context_changes_readout": False,
        "pred_f_parent_level_lexical_specificity": False}
    expected_gain_predictions = {
        "pred_a_instrument_closure": True, "pred_b_quadratic_head_prediction": True,
        "pred_c_quadratic_task_manipulation": True,
        "pred_d_extrapolation_stable": True,
        "pred_e_lexical_effect_bounded": False}
    if split["score"]["predictions"] != expected_split_predictions \
            or tangent["score"]["predictions"] != expected_tangent_predictions \
            or gain["score"]["predictions"] != expected_gain_predictions:
        raise PublicationError("registered outcomes changed")

    split_site = "MLP.block8.subject_input.E_A_U_W_Y_Z_mlp6_7_layers.response_to_attention11_head3.final_position"
    tangent_site = "MLP.block8.subject_input.MLP6_7.contextual_tangent_to_attention11_head3.final_position"
    gain_site = "MLP.block8.subject_input.MLP6_7.quadratic_gain_program_to_attention11_head3.final_position"
    s, t, g = split["score"], tangent["score"], gain["score"]
    events = [
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.instrument_parent_closure.complete.v1", "held", None, "null_control", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("parent_lattice_mobius_max_absolute_error", s["parent_lattice_mobius_max_absolute_error"], "<=5e-5"), metric("parent_downstream_outcome_max_absolute_error", s["parent_downstream_outcome_max_absolute_error"], "<=5e-5")], "The memory-bounded complete E/A/U/W/Y/Z factorial exactly closes and regroups to the frozen parent lattice."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.MLP7_dominant.complete.v1", "null", "scientific_null", "composition", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("MLP7_dominance_prediction", False, "registered all-cell dominance")], "MLP7 is not a stable dominant child of the grouped MLP6--7 source."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.MLP6_dominant.complete.v1", "null", "scientific_null", "composition", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("MLP6_dominance_prediction", False, "registered all-cell dominance")], "MLP6 is not a stable dominant child of the grouped MLP6--7 source."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.distributed.complete.v1", "held", None, "composition", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("prediction", True, "registered distributed criterion")], "Both MLP6 and MLP7 contribute materially; preserve them as one operational source unit."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.composition.complete.v1", "held", None, "composition", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("prediction", True, "registered interaction criterion")], "MLP6-by-MLP7 composition is intervention-live."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.direction_switch.complete.v1", "held", None, "composition", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("prediction", True, "registered direction criterion")], "Child contributions change by grammatical direction without producing a stable child winner."),
        _event("task14_head11_3.fresh_MLP8_mlp6_7_split.number_specificity.complete.v1", "null", "scientific_null", "invariance", split_site, "task14_mlp6_7_split_result", "task14_mlp6_7_split_prior_art", [metric("maximum_lexical_ratio", s["maximum_lexical_ratio"], "<=0.25")], "One registered child corner exceeds the lexical-specificity bar, so neither child receives a semantic promotion."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.instrument_parent_closure.complete.v1", "held", None, "null_control", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("parent_finite_margin_max_absolute_error", t["parent_finite_margin_max_absolute_error"], "<=5e-5"), metric("parent_finite_CE_max_absolute_error", t["parent_finite_CE_max_absolute_error"], "<=5e-5")], "Every finite endpoint reproduces the frozen parent margin and CE effects."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.midpoint_quadratic.complete.v1", "held", None, "composition", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("minimum_midpoint_cosine", min(e["midpoint"]["cosine"] for c in t["cells"].values() for k,e in c.items() if k.startswith("opposite:")), ">=0.95"), metric("maximum_midpoint_relative_error", max(e["midpoint"]["relative_error"] for c in t["cells"].values() for k,e in c.items() if k.startswith("opposite:")), "<=0.25")], "The midpoint derivative accurately predicts the complete finite grouped-source head effect."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.endpoint_local.complete.v1", "null", "scientific_null", "composition", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("prediction", False, "registered endpoint-local gates")], "Recipient-endpoint linearization fails, including task-effect sign reversals."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.material_nonquadratic.complete.v1", "null", "scientific_null", "composition", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("prediction", False, "registered opposing account")], "Residual midpoint error is not materially nonquadratic under the registered bar."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.context_change.complete.v1", "null", "scientific_null", "invariance", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("maximum_background_midpoint_error_gap", t["maximum_background_midpoint_error_gap"], ">=0.15")], "Recipient and donor E/A/U/W backgrounds do not materially change midpoint readout accuracy."),
        _event("task14_head11_3.fresh_MLP6_7_tangent.lexical_specificity.complete.v1", "null", "scientific_null", "invariance", tangent_site, "task14_mlp6_7_tangent_result", "task14_mlp6_7_tangent_prior_art", [metric("maximum_lexical_ratio", t["maximum_lexical_ratio"], "<=0.25")], "The grouped operational source remains lexically entangled."),
        _event("task14_head11_3.fresh_MLP6_7_gain.instrument.complete.v1", "held", None, "null_control", gain_site, "task14_mlp6_7_gain_result", "task14_mlp6_7_gain_prior_art", [metric("prediction", True, "all exactness gates <=5e-5")], "The off-grid native-tail intervention instrument closes."),
        _event("task14_head11_3.fresh_MLP6_7_gain.quadratic_head_prediction.complete.v1", "held", None, "composition", gain_site, "task14_mlp6_7_gain_result", "task14_mlp6_7_gain_prior_art", [metric("minimum_predicted_cosine", g["minimum_opposite_predicted_cosine"], ">=0.98"), metric("maximum_predicted_relative_error", g["maximum_opposite_predicted_relative_error"], "<=0.10")], "Frozen endpoint/midpoint coefficients predict exact head vectors at all unseen gains."),
        _event("task14_head11_3.fresh_MLP6_7_gain.task_manipulation.complete.v1", "held", None, "composition", gain_site, "task14_mlp6_7_gain_result", "task14_mlp6_7_gain_prior_art", [metric("task_recovery_range", [g["minimum_opposite_task_recovery"], g["maximum_opposite_task_recovery"]], "within [0.80,1.20]")], "Predicted heads reproduce the exact gain interventions after installation into the native tail."),
        _event("task14_head11_3.fresh_MLP6_7_gain.extrapolation.complete.v1", "held", None, "cross_family_transfer", gain_site, "task14_mlp6_7_gain_result", "task14_mlp6_7_gain_prior_art", [metric("registered_gains", [-0.5, 1.5], "both pass head and task gates")], "The quadratic law remains predictive at both registered extrapolation gains."),
        _event("task14_head11_3.fresh_MLP6_7_gain.lexical_specificity.complete.v1", "null", "scientific_null", "invariance", gain_site, "task14_mlp6_7_gain_result", "task14_mlp6_7_gain_prior_art", [metric("maximum_lexical_ratio", g["maximum_lexical_ratio"], "<=0.25")], "Predictive manipulation does not repair the grouped source's lexical collateral."),
    ]

    revision = copy.deepcopy(base)
    revision.update({"claim_id": NEW_CLAIM, "revision": 22, "supersedes": BASE_CLAIM,
                     "status": "site_live",
                     "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
                     "next_missing": "The MLP6--7 child split is complete and must not be repeated: neither layer is a stable semantic unit, while their grouped source has a validated context-preserving quadratic readout through MLP8 and L11H3. Frozen coefficients predict unseen interpolation and extrapolation interventions, but same-number lexical collateral remains above bar. OOD/independent-text transfer, coefficient sharing or generation, necessity outside the fixed L11H3 interface, and literal program pricing remain untested."})
    revision["candidate_sites"].extend([
        {"site_id": split_site, "tensor_path": "MLP8 subject input split into E, A, U=MLP0--3, W=MLP4--5, Y=MLP6, and Z=MLP7", "shape": ["batch", 6, 1152], "intervention": "complete E/A/U/W/Y/Z source factorial through exact MLP8 and fixed L11H3", "ceiling_event_ids": []},
        {"site_id": tangent_site, "tensor_path": "contextual directional derivative of grouped MLP6--7 transport through exact MLP8 and L11H3", "shape": ["batch", 1152], "intervention": "install exact, endpoint-JVP, or midpoint-JVP head vectors", "ceiling_event_ids": []},
        {"site_id": gain_site, "tensor_path": "quadratic gain law for grouped MLP6--7 transport to the L11H3 head vector", "shape": ["batch", 1152], "intervention": "install exact or frozen-coefficient predicted heads at t=-0.5,0.5,1.5", "ceiling_event_ids": []},
    ])
    revision["counterfactual_families"].extend([
        {"family_id": "fresh_matched_subject_MLP8_E_A_U_W_Y_Z_mlp6_7_split_v1", "role": "interchange", "changes": ["MLP6 and MLP7 source writes separately and jointly"], "holds_fixed": ["licensed text", "all E/A/U/W source choices", "recipient L11H3 score, cached value, and non-subject complement"], "builder_artifact_id": "task14_fresh_matched_capability_authority", "control_ids": ["complete parent regrouping", "same-number lexical lattice"], "split_plan_id": SPLIT_ID, "status": "validated"},
        {"family_id": "fresh_matched_subject_MLP6_7_contextual_tangent_v1", "role": "interchange", "changes": ["exact versus endpoint and midpoint differential transport"], "holds_fixed": ["grouped MLP6--7 unit", "recipient or donor E/A/U/W background", "native tail"], "builder_artifact_id": "task14_fresh_matched_capability_authority", "control_ids": ["parent margin and CE closure", "same-number lexical source"], "split_plan_id": SPLIT_ID, "status": "validated"},
        {"family_id": "fresh_matched_subject_MLP6_7_quadratic_gain_v1", "role": "interchange", "changes": ["grouped source gain t=-0.5,0.5,1.5"], "holds_fixed": ["quadratic coefficients frozen before gain outcomes", "recipient or donor E/A/U/W background", "native tail"], "builder_artifact_id": "task14_fresh_matched_capability_authority", "control_ids": ["exact gain heads", "same-number lexical gains"], "split_plan_id": SPLIT_ID, "status": "validated"},
    ])
    return {"schema": "task14_mlp6_7_quadratic_program_publication_v22",
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
    args = parser.parse_args(); plan = build_plan()
    if args.apply:
        apply_plan(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
