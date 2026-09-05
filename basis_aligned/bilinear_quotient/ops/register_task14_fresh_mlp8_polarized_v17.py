#!/usr/bin/env python3
"""Publish the corrected licensed Task14 MLP8 polarized-response screen as v17."""

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
BASE_CLAIM = "grammatical_subject_number.v16"
NEW_CLAIM = "grammatical_subject_number.v17"
BASE_CLAIM_SHA256 = "2ad72c5e522738d13ccbcc85e6ba9f3b0b03596ea3e4f558868f279ecc7400f2"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "MLP.block8.subject_response.cross_quadratic_to_attention11_head3.final_position"

ARTIFACT_SPECS = {
    "task14_mlp8_polarized_v1_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_mlp8_polarized_response_factorial_v1_capability_license.json",
        "31c395e00f47c4d27ef36da44f6dd8e2b926c81c0a93cee430ea1c88f22e3950", "capability_license"),
    "task14_mlp8_polarized_v2_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_mlp8_polarized_response_factorial_v2_capability_license.json",
        "ad1792c2e5b211cb46f1f372d23eaba6a328141c7177eb09d035f8f41be8e919", "capability_license"),
    "task14_mlp8_polarized_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1.json",
        "fea58038705bb665bde416a2ac3e02451226226571f6815eab2d60d3a7dd00a6", "preregistration"),
    "task14_mlp8_polarized_v1_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial.py",
        "22100ae8bbe1fec5fe55117321a0acf0380873169460ba123255668665ca75f6", "experiment_runner"),
    "task14_mlp8_polarized_v1_invalid_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1_result.json",
        "27c0a70502765b15dc6b2f4118e176d60b4cd63ec47bae1a19e366799f42dc39", "screen_result"),
    "task14_mlp8_polarized_v2_correction": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_numerical_repair_v2.json",
        "6dd99090a2ac12bb9ff0ea4f3e77d4ada9ab7a4c2c1cc3f5937479a51016f763", "preregistration"),
    "task14_mlp8_polarized_v2_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2.py",
        "749bccf880e179c92c565c9eb43543faae0fcb6b04fd3813d2448581b0138cba", "experiment_runner"),
    "task14_mlp8_polarized_v2_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2_result.json",
        "55d5413306f4471b0c9b8345732d317d0c1c4b82395153a119af3d56514f5ad6", "screen_result"),
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
        "prereg_artifact_id": "task14_mlp8_polarized_v2_correction",
        "result_artifact_id": "task14_mlp8_polarized_v2_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_mlp4_10_conditional_result",
            "task14_mlp8_polarized_v2_capability_license",
            "task14_mlp8_polarized_prior_art",
            "task14_mlp8_polarized_v2_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def _recovery(cells: list[dict], component: str, metric_name: str) -> list[float]:
    return [c["derived"][background]["recovery"][component][metric_name]
            for c in cells for background in ("standalone", "conditional")]


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
        raise PublicationError("canonical Task14 contains neither a v16 migration state nor v17")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v16 is not the exact audited base")

    v1 = docs["task14_mlp8_polarized_v1_invalid_result"]
    correction = docs["task14_mlp8_polarized_v2_correction"]
    license_doc = docs["task14_mlp8_polarized_v2_capability_license"]
    result = docs["task14_mlp8_polarized_v2_result"]
    if v1.get("terminal") != "invalid":
        raise PublicationError("v1 invalid provenance changed")
    if correction.get("corrects", {}).get("result_sha256") != ARTIFACT_SPECS["task14_mlp8_polarized_v1_invalid_result"][1]:
        raise PublicationError("correction does not bind v1 invalid result")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("v2 license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("split scope changed")
    expected = {
        "pred_a_instrument_live": True, "pred_b_cross_dominant": False,
        "pred_c_quadratic_dominant": False, "pred_d_distributed": False,
        "pred_e_downstream_interaction_needed": False, "pred_f_background_stable": True,
        "pred_g_number_specific": True, "pred_h_lexical_collateral": False,
    }
    score = result["score"]
    if score.get("predictions") != expected:
        raise PublicationError("registered outcomes changed")
    cells_by_name = score["cells"]
    cells = list(cells_by_name.values())

    exact_names = [
        "native_replay_max_absolute_logit_error", "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error", "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error", "output_closure_max_absolute_error",
        "propagated_recipient_MLP8_max_absolute_error", "propagated_source_MLP8_max_absolute_error",
        "gauge_invariance_max_absolute_error", "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    events = [_event(
        "task14_head11_3.fresh_MLP8_polarized.instrument.complete.v1", "held", None,
        [metric(n, score[n], "within frozen exactness bar") for n in exact_names] + [
            metric(f"{background}_full_margin_range",
                   _range([c["derived"][background]["full"]["mean_margin"] for c in cells]),
                   ">=0.03 with >=0.75 helpful rows in every cell")
            for background in ("standalone", "conditional")
        ] + [
            metric(f"{background}_full_CE_range",
                   _range([c["derived"][background]["full"]["mean_CE"] for c in cells]),
                   ">=0 with >=0.75 helpful rows in every cell")
            for background in ("standalone", "conditional")
        ],
        "The repaired invariant cross/quadratic split passed all frozen closure, gauge-invariance, replay, no-op, installation, and full-MLP8 task controls. V1 remains invalid numerical provenance, not scientific evidence.",
        test_type="null_control")]

    cross_margin, cross_ce = _recovery(cells, "cross", "margin"), _recovery(cells, "cross", "CE")
    quad_margin, quad_ce = _recovery(cells, "quadratic", "margin"), _recovery(cells, "quadratic", "CE")
    events.extend([
        _event("task14_head11_3.fresh_MLP8_polarized.cross_dominant.complete.v1", "null", "scientific_null",
               [metric("cross_margin_recovery_range", _range(cross_margin), ">=0.70 in every cell/background"),
                metric("cross_CE_recovery_range", _range(cross_ce), ">=0.70 in every cell/background"),
                metric("quadratic_margin_recovery_range", _range(quad_margin), "absolute value <=0.25 in every cell/background"),
                metric("quadratic_CE_recovery_range", _range(quad_ce), "absolute value <=0.25 in every cell/background")],
               "The registered cross-dominant account was null because the cross term changes sign with transfer direction and the quadratic term is not uniformly minor."),
        _event("task14_head11_3.fresh_MLP8_polarized.quadratic_dominant.complete.v1", "null", "scientific_null",
               [metric("quadratic_margin_recovery_range", _range(quad_margin), ">=0.70 in every cell/background"),
                metric("quadratic_CE_recovery_range", _range(quad_ce), ">=0.70 in every cell/background"),
                metric("cross_margin_recovery_range", _range(cross_margin), "absolute value <=0.25 in every cell/background"),
                metric("cross_CE_recovery_range", _range(cross_ce), "absolute value <=0.25 in every cell/background")],
               "The registered quadratic-dominant account was null because the quadratic term also changes sign with direction and the cross term is not uniformly minor."),
        _event("task14_head11_3.fresh_MLP8_polarized.distributed.complete.v1", "null", "scientific_null",
               [metric("cross_margin_recovery_range", _range(cross_margin), ">=0.25 in every cell/background"),
                metric("cross_CE_recovery_range", _range(cross_ce), ">=0.25 in every cell/background"),
                metric("quadratic_margin_recovery_range", _range(quad_margin), ">=0.25 in every cell/background"),
                metric("quadratic_CE_recovery_range", _range(quad_ce), ">=0.25 in every cell/background")],
               "The all-cell distributed definition was null: signed recovery reverses across plural-to-singular versus singular-to-plural transfer."),
    ])

    interaction_margin = [c["derived"][b]["interaction_recovery"]["margin"] for c in cells for b in ("standalone", "conditional")]
    interaction_ce = [c["derived"][b]["interaction_recovery"]["CE"] for c in cells for b in ("standalone", "conditional")]
    events.append(_event(
        "task14_head11_3.fresh_MLP8_polarized.downstream_interaction_needed.complete.v1", "null", "scientific_null",
        [metric("interaction_margin_recovery_range", _range(interaction_margin), ">=0.25 in every cell/background"),
         metric("interaction_CE_recovery_range", _range(interaction_ce), ">=0.25 in every cell/background")],
        "The registered all-cell downstream-interaction criterion was null; interaction recovery was substantial only in some plural-to-singular cells."))

    background_margin = [abs(c["derived"]["standalone"]["recovery"][component]["margin"] -
                             c["derived"]["conditional"]["recovery"][component]["margin"])
                         for c in cells for component in ("cross", "quadratic")]
    background_ce = [abs(c["derived"]["standalone"]["recovery"][component]["CE"] -
                         c["derived"]["conditional"]["recovery"][component]["CE"])
                     for c in cells for component in ("cross", "quadratic")]
    events.append(_event(
        "task14_head11_3.fresh_MLP8_polarized.background_stable.complete.v1", "held", None,
        [metric("background_recovery_difference_margin_range", _range(background_margin), "<=0.25 for every cell/component"),
         metric("background_recovery_difference_CE_range", _range(background_ce), "<=0.25 for every cell/component")],
        "Cross and quadratic recovery changed by at most 0.058 between recipient and opposite-number MLP4--10 backgrounds.", test_type="invariance"))

    lexical_margin = [c["derived"]["lexical_ratio"]["margin"] for c in cells]
    lexical_ce = [c["derived"]["lexical_ratio"]["CE"] for c in cells]
    events.extend([
        _event("task14_head11_3.fresh_MLP8_polarized.number_specificity.complete.v1", "held", None,
               [metric("lexical_margin_ratio_range", _range(lexical_margin), "<=0.25 in every cell"),
                metric("lexical_CE_ratio_range", _range(lexical_ce), "<=0.25 in every cell")],
               "Same-number different-lemma full-MLP8 effects were small relative to opposite-number effects.", test_type="invariance"),
        _event("task14_head11_3.fresh_MLP8_polarized.lexical_collateral.complete.v1", "null", "scientific_null",
               [metric("maximum_lexical_margin_ratio", max(lexical_margin), ">=0.50 in any cell"),
                metric("maximum_lexical_CE_ratio", max(lexical_ce), ">=0.50 in any cell")],
               "The preregistered lexical-collateral alternative was null.", test_type="invariance"),
    ])

    def direction_values(prefix: str, component: str) -> list[float]:
        subset = [c for name, c in cells_by_name.items() if name.startswith(prefix)]
        return [c["derived"][background]["recovery"][component][out]
                for c in subset for background in ("standalone", "conditional")
                for out in ("margin", "CE")]

    events.append(_event(
        "task14_head11_3.fresh_MLP8_polarized.direction_polarization_exploratory.complete.v1", "inconclusive", None,
        [metric("plural_to_singular_cross_recovery_range", _range(direction_values("plural_to_singular", "cross")), "exploratory; no preregistered direction-specific bar"),
         metric("plural_to_singular_quadratic_recovery_range", _range(direction_values("plural_to_singular", "quadratic")), "exploratory; no preregistered direction-specific bar"),
         metric("singular_to_plural_cross_recovery_range", _range(direction_values("singular_to_plural", "cross")), "exploratory; no preregistered direction-specific bar"),
         metric("singular_to_plural_quadratic_recovery_range", _range(direction_values("singular_to_plural", "quadratic")), "exploratory; no preregistered direction-specific bar")],
        "Exploratory, not a preregistered held claim: plural-to-singular transfer is cross-positive (2.62--2.93) and quadratic-negative (-1.55---1.47), whereas singular-to-plural is cross-negative (-0.64---0.33) and quadratic-positive (1.33--1.65). The polarization is stable across standalone and conditional backgrounds."))

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 17, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The repaired licensed MLP8 invariant-response screen is complete and must not be repeated. Registered cross-dominant, quadratic-dominant, distributed, and downstream-interaction-needed all-cell claims were null; background stability and number specificity held, and lexical collateral was null. Exploratorily, the two exact invariant response components polarize by number-transfer direction and remain stable across recipient versus opposite-number MLP4--10 backgrounds. This is an exact within-MLP8 response split at subject position 8 through L11H3, not native product identities or a final semantic basis. OOD syntax, necessity, native product identities, and downstream readers remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "the exact invariant cross and quadratic components of MLP8's subject-position response, propagated through blocks 9--11 and read by L11H3 current value",
        "shape": ["batch", 2, 1152],
        "intervention": "swap exact cross, quadratic, or both MLP8 response components under recipient and opposite-number MLP4--10 backgrounds while holding L11H3 score, cached value, and non-subject source terms fixed",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "fresh_matched_subject_MLP8_invariant_polarized_response_v2",
        "role": "interchange",
        "changes": ["the exact invariant cross response", "the exact invariant quadratic response", "both response components together"],
        "holds_fixed": ["licensed HOLDOUT text and subject position 8", "recipient E+A+R and MLP0--3+MR", "chosen recipient or opposite-number background for other MLP4--10 writes", "recipient L11H3 p_8, cached value, and native non-subject complement"],
        "builder_artifact_id": "task14_fresh_matched_capability_authority",
        "control_ids": ["product and output closure", "gauge invariance", "same-batch no-op", "full-MLP8 reproduction", "same-number different-lemma controls"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_fresh_mlp8_polarized_publication_v17",
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
