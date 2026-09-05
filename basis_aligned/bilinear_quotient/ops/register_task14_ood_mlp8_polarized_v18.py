#!/usr/bin/env python3
"""Publish the prospective MLP8 intervention on reused OOD Task14 text as v18."""

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
BASE_CLAIM = "grammatical_subject_number.v17"
NEW_CLAIM = "grammatical_subject_number.v18"
BASE_CLAIM_SHA256 = "3f3c86c55a391576c9394327242d217ebc4602160f4de1931ffcc8033af44895"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "MLP.block8.OOD_fronted_subject_response.cross_quadratic_to_attention11_head3.final_position"
EVAL_ROLE = "OOD_TEXT_REUSE_NEW_MLP8_INTERVENTION"

ARTIFACT_SPECS = {
    "task14_ood_mlp8_authority": (
        "ops/circuit_fast_screen_candidate_task14_ood_fronted_mlp8_polarized_response.py",
        "9945ef76cb65fe3717f54be478dab2d7444f92738c4075c3ea47b54ab252cccb", "dataset_builder"),
    "task14_ood_mlp8_capability_result": (
        "circuits/fast_screens/task14_ood_fronted_mlp8_native_capability_v1_result.json",
        "5db771b8910bca085d201893552b409e96f66d0f01921f0c1cb5e4ba905d8615", "capability_result"),
    "task14_ood_mlp8_capability_license": (
        "circuits/fast_screens/task14_ood_fronted_mlp8_polarized_response_v1_capability_license.json",
        "43090b258a75e257b8bd186dd970eb40c14ff166003a8d1a7fb160e2de3303d6", "capability_license"),
    "task14_ood_mlp8_prior_art": (
        "circuits/prior_art/task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1.json",
        "394c1e3c0233cefa3ba2bb07f7a83e2f9aaa2bb991e5625595e8b0c5f17c360f", "preregistration"),
    "task14_ood_mlp8_runner": (
        "ops/run_task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial.py",
        "b9858214eba49b1a8f64d69f42790765472c15e76391d44a52ba6d94afe0ad8d", "experiment_runner"),
    "task14_ood_mlp8_result": (
        "circuits/fast_screens/task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1_result.json",
        "31e379a376f29a6b71cb33fc77078edcaee9a64783793984ed0ca6df1a9cfd0b", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, test_type: str, metrics: list[dict], notes: str) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": "held", "failure_kind": None,
        "family_ids": [], "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": EVAL_ROLE, "metrics": metrics,
        "prereg_artifact_id": "task14_ood_mlp8_prior_art",
        "result_artifact_id": "task14_ood_mlp8_result" if test_type != "capability" else "task14_ood_mlp8_capability_result",
        "input_artifact_ids": [
            "task14_ood_mlp8_authority", "task14_ood_mlp8_capability_result",
            "task14_ood_mlp8_capability_license", "task14_ood_mlp8_prior_art",
            "task14_ood_mlp8_runner", "task14_mlp8_polarized_v2_result",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def _direction(cells: dict, prefix: str, component: str) -> list[float]:
    cell = next(c for name, c in cells.items() if name.startswith(prefix))
    return [cell["derived"][background]["recovery"][component][outcome]
            for background in ("standalone", "conditional")
            for outcome in ("margin", "CE")]


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
        raise PublicationError("canonical Task14 contains neither a v17 migration state nor v18")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v17 is not the exact audited base")

    capability = docs["task14_ood_mlp8_capability_result"]
    license_doc = docs["task14_ood_mlp8_capability_license"]
    result = docs["task14_ood_mlp8_result"]
    if len(capability.get("cells", {})) != 6 or any(c.get("accuracy") != 1.0 for c in capability["cells"].values()):
        raise PublicationError("scoped capability cells changed")
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("capability license does not bind result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("terminal or checkpoint changed")
    if result.get("evaluated_splits") != [EVAL_ROLE] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("evaluation scope changed")
    expected = {
        "pred_a_instrument_live": True,
        "pred_b_plural_to_singular_cross_positive_quadratic_negative": True,
        "pred_c_singular_to_plural_cross_negative_quadratic_positive": True,
        "pred_d_signed_direction_pattern": True, "pred_e_background_stable": True,
        "pred_f_number_specific": True, "pred_g_selective_removal_direction_pattern": True,
    }
    score = result["score"]
    if score.get("predictions") != expected or score.get("selective_removal_independence") != "algebraically dependent on the same four corners":
        raise PublicationError("registered outcomes or removal dependence changed")
    cells = score["cells"]

    events = [_event(
        "task14_head11_3.OOD_MLP8_polarized.scoped_capability.complete.v1", "capability",
        [metric("cell_accuracies", {name: cell["accuracy"] for name, cell in capability["cells"].items()}, ">=0.875 in each of six cells"),
         metric("minimum_cell_accuracy", min(cell["accuracy"] for cell in capability["cells"].values()), ">=0.875")],
        "All six direction-by-role native capability cells scored 1.0. The OOD text and whole-head outcomes were previously open; only the MLP8 intervention was prospective, so this is not pristine held-out OOD.")]

    exact_names = [
        "native_replay_max_absolute_logit_error", "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error", "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error", "output_closure_max_absolute_error",
        "propagated_recipient_MLP8_max_absolute_error", "propagated_source_MLP8_max_absolute_error",
        "gauge_invariance_max_absolute_error", "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error", "installed_head_max_absolute_error",
    ]
    events.append(_event(
        "task14_head11_3.OOD_MLP8_polarized.instrument.complete.v1", "null_control",
        [metric(n, score[n], "within frozen exactness bar") for n in exact_names],
        "The prospective MLP8 intervention passed all exactness, gauge-invariance, replay, no-op, installation, and full-effect controls on reused OOD text."))

    ps_cross = _direction(cells, "plural_to_singular", "cross")
    ps_quad = _direction(cells, "plural_to_singular", "quadratic")
    sp_cross = _direction(cells, "singular_to_plural", "cross")
    sp_quad = _direction(cells, "singular_to_plural", "quadratic")
    events.extend([
        _event("task14_head11_3.OOD_MLP8_polarized.plural_to_singular_signed_split.complete.v1", "ood",
               [metric("cross_recovery_range", _range(ps_cross), ">=1.5 for margin and CE in both backgrounds"),
                metric("quadratic_recovery_range", _range(ps_quad), "<=-0.5 for margin and CE in both backgrounds")],
               "On fronted two-attractor plural-to-singular transfer, the cross response was strongly positive and the quadratic response negative in both backgrounds."),
        _event("task14_head11_3.OOD_MLP8_polarized.singular_to_plural_signed_split.complete.v1", "ood",
               [metric("cross_recovery_range", _range(sp_cross), "<=-0.1 for margin and CE in both backgrounds"),
                metric("quadratic_recovery_range", _range(sp_quad), ">=1.1 for margin and CE in both backgrounds")],
               "On fronted two-attractor singular-to-plural transfer, the cross response was negative and the quadratic response strongly positive in both backgrounds."),
        _event("task14_head11_3.OOD_MLP8_polarized.joint_direction_pattern.complete.v1", "ood",
               [metric("plural_to_singular_cross_recovery_range", _range(ps_cross), ">=1.5"),
                metric("plural_to_singular_quadratic_recovery_range", _range(ps_quad), "<=-0.5"),
                metric("singular_to_plural_cross_recovery_range", _range(sp_cross), "<=-0.1"),
                metric("singular_to_plural_quadratic_recovery_range", _range(sp_quad), ">=1.1")],
               "Both preregistered opposing signed direction patterns held on the reused OOD text under the prospective MLP8 intervention."),
    ])

    bg_diffs = {outcome: [
        abs(c["derived"]["standalone"]["recovery"][component][outcome] -
            c["derived"]["conditional"]["recovery"][component][outcome])
        for c in cells.values() for component in ("cross", "quadratic")]
        for outcome in ("margin", "CE")}
    lexical = {outcome: [c["derived"]["lexical_ratio"][outcome] for c in cells.values()]
               for outcome in ("margin", "CE")}
    events.extend([
        _event("task14_head11_3.OOD_MLP8_polarized.background_stability.complete.v1", "invariance",
               [metric("background_difference_margin_range", _range(bg_diffs["margin"]), "<=0.25 for every component/direction"),
                metric("background_difference_CE_range", _range(bg_diffs["CE"]), "<=0.25 for every component/direction")],
               "The signed response recoveries were stable between recipient and opposite-number other-MLP backgrounds."),
        _event("task14_head11_3.OOD_MLP8_polarized.number_specificity.complete.v1", "invariance",
               [metric("lexical_margin_ratio_range", _range(lexical["margin"]), "<=0.25 in every direction"),
                metric("lexical_CE_ratio_range", _range(lexical["CE"]), "<=0.25 in every direction")],
               "Same-number different-lemma full-MLP8 effects remained small relative to opposite-number effects."),
    ])

    removal = {name: [c["derived"]["selective_removal"][name][outcome]
                      for c in cells.values() for outcome in ("margin", "CE")]
               for name in ("remove_cross", "remove_quadratic")}
    events.append(_event(
        "task14_head11_3.OOD_MLP8_polarized.selective_removal_direction_pattern.complete.v1", "removal",
        [metric("remove_cross_signed_values", removal["remove_cross"], "positive plural-to-singular and negative singular-to-plural"),
         metric("remove_quadratic_signed_values", removal["remove_quadratic"], "negative plural-to-singular and positive singular-to-plural"),
         metric("independent_replication", False, "explicitly not independent")],
        "The direction-reversed selective-removal signs held, but they are deterministic algebraic contrasts of the same factorial corners. This event is dependent evidence, not an independent removal replication."))

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 18, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The prospective MLP8 intervention on reused fronted two-attractor OOD text is complete and must not be repeated. Scoped capability, exactness, both signed direction splits, their joint pattern, background stability, number specificity, and the algebraically dependent selective-removal pattern held. The OOD text and whole-head outcomes were previously open; only the MLP8 intervention was prospective, so this is not pristine held-out OOD. Independent selective removal, unrelated-behavior selectivity, necessity, native product identities, and downstream readers remain untested."
        ),
    })
    revision["candidate_sites"].append({
        "site_id": SITE_ID,
        "tensor_path": "the invariant cross and quadratic components of MLP8's subject-position response on fronted two-attractor text, propagated to L11H3",
        "shape": ["batch", 2, 1152],
        "intervention": "prospectively swap cross, quadratic, or full MLP8 response on previously opened OOD text under recipient and opposite-number other-MLP backgrounds",
        "ceiling_event_ids": [],
    })
    revision["counterfactual_families"].append({
        "family_id": "OOD_fronted_subject_MLP8_invariant_polarized_response_v1",
        "role": "interchange",
        "changes": ["the exact MLP8 cross response", "the exact MLP8 quadratic response", "both responses together"],
        "holds_fixed": ["reused OOD fronted two-attractor text and subject position 8", "recipient E+A+R and MLP0--3+MR", "specified recipient or opposite-number background for other MLP4--10 writes", "recipient L11H3 p_8, cached value, and non-subject complement"],
        "builder_artifact_id": "task14_ood_mlp8_authority",
        "control_ids": ["six-cell scoped native capability", "exact product/output closure and gauge invariance", "same-batch no-op and full-effect reproduction", "same-number different-lemma controls"],
        "split_plan_id": SPLIT_ID, "status": "validated",
    })
    return {"schema": "task14_ood_mlp8_polarized_publication_v18",
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
