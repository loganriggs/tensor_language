#!/usr/bin/env python3
"""Publish the licensed Task14 MLP8 E/A/M input-writer factorial as v19."""

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
BASE_CLAIM = "grammatical_subject_number.v18"
NEW_CLAIM = "grammatical_subject_number.v19"
BASE_CLAIM_SHA256 = "7b5dd0790c6623e254a4ff8c996962b14c9fe10b57e485a2b4c49173e08857c5"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "MLP.block8.subject_input.E_A_M_writer_families.response_to_attention11_head3.final_position"

ARTIFACT_SPECS = {
    "task14_mlp8_input_writers_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_mlp8_input_writer_response_factorial_v1_capability_license.json",
        "693580f63d4e40ee9f36a0b32a733d7768aa1a28c3d13ad689080241f70adba2", "capability_license"),
    "task14_mlp8_input_writers_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1.json",
        "6ceb69fa0860890534e89151c0b4a20290a271f400c13844c229008e56849b8b", "preregistration"),
    "task14_mlp8_input_writers_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial.py",
        "68e642aa68b30cbdffa616adca70900d1bfadd727f70241478f6ae5351bb1bdf", "experiment_runner"),
    "task14_mlp8_input_writers_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1_result.json",
        "da639bb23aef25b78da20170b52e31e4e2d5f64a95fd5586b2bf07a12cb1a7ed", "screen_result"),
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
        "prereg_artifact_id": "task14_mlp8_input_writers_prior_art",
        "result_artifact_id": "task14_mlp8_input_writers_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_mlp8_polarized_v2_result",
            "task14_upstream_writers_v2_result",
            "task14_mlp8_input_writers_capability_license",
            "task14_mlp8_input_writers_runner",
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
        artifacts[artifact_id] = {"path": str(path.relative_to(REPO)), "sha256": actual,
                                  "kind": kind, "status": "frozen"}
        if path.suffix == ".json":
            docs[artifact_id] = json.loads(path.read_text())

    record = json.loads(registry.circuit_path(TAG).read_text())
    if record["claims"][-1]["claim_id"] not in {BASE_CLAIM, NEW_CLAIM} \
            and not any(c["claim_id"] == NEW_CLAIM for c in record["claims"]):
        raise PublicationError("canonical Task14 contains neither a v18 migration state nor v19")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v18 is not the exact audited base")

    license_doc = docs["task14_mlp8_input_writers_capability_license"]
    result = docs["task14_mlp8_input_writers_result"]
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_live": True, "pred_b_M_source_dominant": False,
        "pred_c_E_source_dominant": False, "pred_d_A_source_dominant": False,
        "pred_e_distributed_additive": False, "pred_f_source_interaction_needed": True,
        "pred_g_direction_stable": False, "pred_h_direction_switch": False,
        "pred_i_number_specific": True, "pred_j_lexical_collateral": False,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    if score.get("direction_component_winners") != {
            "plural_to_singular": {"cross": None, "full": None, "quadratic": None},
            "singular_to_plural": {"cross": None, "full": None, "quadratic": None}}:
        raise PublicationError("direction winner nulls changed")
    cells = list(score["cells"].values())

    exact_names = [
        "native_replay_max_absolute_logit_error", "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error", "hybrid_endpoint_max_absolute_error",
        "source_term_sum_max_absolute_error", "product_closure_max_absolute_error",
        "output_closure_max_absolute_error", "propagated_endpoint_max_absolute_error",
        "gauge_invariance_max_absolute_error", "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
        "downstream_state_closure_max_absolute_error", "downstream_normalized_closure_max_absolute_error",
    ]
    full_margin = [c["opposite"]["full"]["margin"]["effects"]["EAM"] for c in cells]
    full_ce = [c["opposite"]["full"]["CE"]["effects"]["EAM"] for c in cells]
    events = [_event(
        "task14_head11_3.fresh_MLP8_input_writers.instrument.complete.v1", "held", None,
        [metric(n, score[n], "within frozen exactness bar") for n in exact_names] + [
            metric("full_EAM_margin_effect_range", _range(full_margin), ">=0.03 with >=0.75 helpful rows in every cell"),
            metric("full_EAM_CE_effect_range", _range(full_ce), ">=0 with >=0.75 helpful rows in every cell")],
        "The exact E/A/M input-writer factorial passed every state, normalized-input, hybrid-endpoint, response-closure, gauge, replay, no-op, installation, and full-effect control.",
        test_type="null_control")]

    recovery = {family: [
        c["opposite"][piece][outcome]["signed_recovery"][family]
        for c in cells for piece in ("cross", "quadratic", "full")
        for outcome in ("margin", "CE")]
        for family in ("E", "A", "M")}
    family_notes = {
        "M": "The prior-MLP input family did not satisfy the registered dominant-and-minor all-cell criterion across cross, quadratic, and full responses.",
        "E": "The embedding/skip input family did not satisfy the registered dominant-and-minor all-cell criterion.",
        "A": "The prior-attention input family did not satisfy the registered dominant-and-minor all-cell criterion.",
    }
    for family in ("M", "E", "A"):
        events.append(_event(
            f"task14_head11_3.fresh_MLP8_input_writers.{family}_dominant.complete.v1",
            "null", "scientific_null",
            [metric(f"{family}_signed_recovery_range", _range(recovery[family]),
                    ">=0.70 while both other families are <=0.25 in absolute value, across every cell/component/outcome")],
            family_notes[family]))

    interaction = {piece: [
        c["opposite"][piece][outcome]["interaction_residual_fraction"]
        for c in cells for outcome in ("margin", "CE")]
        for piece in ("cross", "quadratic", "full")}
    events.extend([
        _event("task14_head11_3.fresh_MLP8_input_writers.distributed_additive.complete.v1", "null", "scientific_null",
               [metric("M_signed_recovery_range", _range(recovery["M"]), "at least two families >=0.25 in every cell/component/outcome"),
                metric("E_signed_recovery_range", _range(recovery["E"]), "at least two families >=0.25 in every cell/component/outcome"),
                metric("A_signed_recovery_range", _range(recovery["A"]), "at least two families >=0.25 in every cell/component/outcome"),
                metric("interaction_fraction_overall_range", _range([v for values in interaction.values() for v in values]), "<=0.25 everywhere")],
               "The registered distributed-additive account was null: the required two-family support and small interaction residual did not hold across every component and cell."),
        _event("task14_head11_3.fresh_MLP8_input_writers.source_interaction_needed.complete.v1", "held", None,
               [metric("cross_interaction_fraction_range", _range(interaction["cross"]), ">=0.25 for one component across every cell/outcome"),
                metric("quadratic_interaction_fraction_range", _range(interaction["quadratic"]), ">=0.25 for one component across every cell/outcome"),
                metric("full_interaction_fraction_range", _range(interaction["full"]), ">=0.25 for one component across every cell/outcome")],
               "Input-family interactions are required under the registered causal set-function criterion: quadratic and full response residuals exceed 0.25 across every cell and task outcome. These are normalization-mediated causal interactions, not tensor identities."),
        _event("task14_head11_3.fresh_MLP8_input_writers.direction_stable.complete.v1", "null", "scientific_null",
               [metric("direction_component_winners", score["direction_component_winners"], "a unique source winner exists and agrees across directions")],
               "Direction stability was null because no cross, quadratic, or full component had a unique dominant E/A/M family in either direction."),
        _event("task14_head11_3.fresh_MLP8_input_writers.direction_switch.complete.v1", "null", "scientific_null",
               [metric("direction_component_winners", score["direction_component_winners"], "a component has different unique winners across directions")],
               "Direction switching was also null because no component had unique winners to compare; this is distinct from evidence for stable identity."),
    ])

    events.extend([
        _event("task14_head11_3.fresh_MLP8_input_writers.number_specificity.complete.v1", "held", None,
               [metric("maximum_lexical_ratio", score["maximum_lexical_ratio"], "<=0.25 across every source subset/component/outcome")],
               "Every same-number different-lemma source-subset effect remained below one quarter of its opposite-number full-source reference.", test_type="invariance"),
        _event("task14_head11_3.fresh_MLP8_input_writers.lexical_collateral.complete.v1", "null", "scientific_null",
               [metric("maximum_lexical_ratio", score["maximum_lexical_ratio"], ">=0.50 for any source subset/component/outcome")],
               "The preregistered lexical-collateral alternative was null.", test_type="invariance"),
    ])

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 19, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The licensed MLP8 E/A/M input-writer factorial is complete and must not be repeated. M-, E-, and A-dominance, distributed-additive, direction-stable, direction-switch, and lexical-collateral predictions were null. Source interaction was required for the quadratic and full MLP8 responses; number specificity held. E/A/M are operational native writer families and the interactions are causal set-function effects after normalization, not unique semantic units or tensor identities. Individual writers, a gauge-stable within-input basis, new independent data, OOD replication of this input split, downstream readers, and necessity outside the fixed L11H3 interface remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "MLP8's subject-position normalized input, decomposed operationally into embedding/skip E, attention writes A0--8, and MLP writes M0--7 before exact cross/quadratic response computation",
        "shape": ["batch", 3, 1152],
        "intervention": "factorially swap every nonempty E/A/M donor subset, compute exact invariant cross/quadratic/full MLP8 responses, and install the propagated response only through the fixed L11H3 subject-value interface",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_MLP8_input_writer_E_A_M_factorial_v1",
        "role": "interchange",
        "changes": ["embedding/skip input E", "accumulated attention-write input A", "accumulated prior-MLP input M", "all pair and triple combinations"],
        "holds_fixed": ["licensed HOLDOUT text and subject position 8", "recipient other MLP4--10 downstream background", "recipient L11H3 p_8 and cached value", "native non-subject L11H3 source complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["exact E+A+M normalized-input closure", "hybrid response and propagated endpoint closure", "gauge invariance", "same-batch no-op and full EAM response", "same-number lexical subsets"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_mlp8_input_writers_publication_v19",
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
