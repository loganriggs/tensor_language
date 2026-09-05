#!/usr/bin/env python3
"""Publish the licensed Task14 subject-term/complement factorial as v12."""

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
BASE_CLAIM = "grammatical_subject_number.v11"
NEW_CLAIM = "grammatical_subject_number.v12"
BASE_CLAIM_SHA256 = "d306a0a9a2ad462b925d5928b02d61db7430b2c2df55a24b1fdd111adcb4c047"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT_ID = "task14_fresh_matched_natural_split_v1"
SITE_ID = "attention.block11.head3.subject_term_and_complement.final_position"

ARTIFACT_SPECS = {
    "task14_fresh_subject_term_capability_license": (
        "circuits/fast_screens/task14_fresh_matched_subject_term_vs_complement_factorial_v1_capability_license.json",
        "9ac6596d92a0b7e75a65a31edebcb480676fa81505cd211028babc56ca1ecf18", "capability_license"),
    "task14_fresh_subject_term_prior_art": (
        "circuits/prior_art/task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_v1.json",
        "ea9385643db35efc29e006b79b661f24ea9acf6912190c29d0fad480d857c84c", "preregistration"),
    "task14_fresh_subject_term_runner": (
        "ops/run_task14_head11_3_fresh_matched_subject_term_vs_complement_factorial.py",
        "2709accd9c6e62e8930ef4d2ca119e6155e2b768e532a44319132aada3880afa", "experiment_runner"),
    "task14_fresh_subject_term_result": (
        "circuits/fast_screens/task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_v1_result.json",
        "f398ec274c68aeabe956aa2b957de9ccfb275efe70eaf592809e3475dc9bbbc6", "screen_result"),
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(event_id: str, test_type: str, verdict: str, failure_kind: str | None,
           metrics: list[dict], notes: str) -> dict:
    return {
        "event_id": event_id, "claim_id": BASE_CLAIM, "test_type": test_type,
        "stage": "complete", "verdict": verdict, "failure_kind": failure_kind,
        "family_ids": [], "site_id": SITE_ID, "split_plan_id": SPLIT_ID,
        "evaluation_role": "FRESH_LICENSED_HOLDOUT", "metrics": metrics,
        "prereg_artifact_id": "task14_fresh_subject_term_prior_art",
        "result_artifact_id": "task14_fresh_subject_term_result",
        "input_artifact_ids": [
            "task14_fresh_matched_capability_authority",
            "task14_fresh_matched_capability_license_result",
            "task14_fresh_subject_term_capability_license",
            "task14_fresh_subject_term_runner",
        ],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
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
        raise PublicationError("canonical Task14 latest claim is not v11/v12")
    base = next(c for c in record["claims"] if c["claim_id"] == BASE_CLAIM)
    if registry._canonical_hash(base) != BASE_CLAIM_SHA256:
        raise PublicationError("canonical Task14 v11 is not the exact audited base")

    license_doc = docs["task14_fresh_subject_term_capability_license"]
    prereg = docs["task14_fresh_subject_term_prior_art"]
    result = docs["task14_fresh_subject_term_result"]
    if license_doc.get("decision") != "pass" or license_doc.get("causal_candidate_id") != result.get("candidate_id"):
        raise PublicationError("derivative capability license does not bind the result")
    if prereg.get("candidate_id") != result.get("candidate_id"):
        raise PublicationError("preregistration does not bind the result")
    if result.get("terminal") != "valid_causal_screen" or result.get("checkpoint_weights_sha256") != CHECKPOINT:
        raise PublicationError("result terminal or checkpoint changed")
    if result.get("evaluated_splits") != ["LICENSED_HOLDOUT"] or result.get("forbidden_splits_opened") != []:
        raise PublicationError("result split scope changed")
    expected_predictions = {
        "pred_a_instrument_live": True,
        "pred_b_interaction_repairs_p2s": False,
        "pred_c_complement_independently_carries_task": False,
        "pred_d_complement_asymmetry_persists": False,
    }
    score = result["score"]
    if score.get("predictions") != expected_predictions:
        raise PublicationError("registered outcomes changed")
    cells = score["cells"]
    complete = [c["complete_head"] for c in cells.values()]
    subject = [c["subject_only"] for c in cells.values()]
    complement = [c["complement_only"] for c in cells.values()]
    interaction = [c["interaction"] for c in cells.values()]
    margin_recovery = [s["mean_margin"] / h["mean_margin"] for s, h in zip(subject, complete)]
    ce_recovery = [s["mean_CE"] / h["mean_CE"] for s, h in zip(subject, complete)]
    p2s = [c for k, c in cells.items() if k.startswith("plural_to_singular")]

    instrument = _event(
        "task14_head11_3.fresh_subject_term.instrument.complete.v1", "null_control", "held", None,
        [metric("native_replay_max_absolute_logit_error", score["native_replay_max_absolute_logit_error"], "<=0.00007"),
         metric("source_term_sum_max_absolute_error", score["source_term_sum_max_absolute_error"], "<=0.00005"),
         metric("same_batch_native_noop_endpoint_max_absolute_error", score["same_batch_native_noop_endpoint_max_absolute_error"], "<=0.00007"),
         metric("installed_head_max_absolute_error", score["installed_head_max_absolute_error"], "<=0.00005"),
         metric("complete_head_vector_max_absolute_error", score["complete_head_vector_max_absolute_error"], "<=0.00005"),
         metric("minimum_complete_head_mean_margin", min(x["mean_margin"] for x in complete), ">=0.05 in every cell"),
         metric("minimum_complete_head_mean_CE", min(x["mean_CE"] for x in complete), ">=0 in every cell")],
        "Exact algebra, no-op, and complete-donor-head controls passed; this validates the instrument only.")
    repair_null = _event(
        "task14_head11_3.fresh_subject_term.interaction_repair.complete.v1", "composition", "null", "scientific_null",
        [metric("plural_to_singular_subject_mean_margin_range", [min(c["subject_only"]["mean_margin"] for c in p2s), max(c["subject_only"]["mean_margin"] for c in p2s)], "subject-only must be harmful in both cells"),
         metric("plural_to_singular_interaction_mean_margin_range", [min(c["interaction"]["mean_margin"] for c in p2s), max(c["interaction"]["mean_margin"] for c in p2s)], "positive and at least the absolute subject-only harm"),
         metric("plural_to_singular_interaction_mean_CE_range", [min(c["interaction"]["mean_CE"] for c in p2s), max(c["interaction"]["mean_CE"] for c in p2s)], "positive in both cells")],
        "Registered interaction-repair prediction was null: subject-only was helpful, not harmful, and the interaction was near zero.")
    complement_null = _event(
        "task14_head11_3.fresh_subject_term.complement_independent_use.complete.v1", "composition", "null", "scientific_null",
        [metric("complement_mean_margin_range", [min(x["mean_margin"] for x in complement), max(x["mean_margin"] for x in complement)], ">=0.05 in every cell"),
         metric("complement_mean_CE_range", [min(x["mean_CE"] for x in complement), max(x["mean_CE"] for x in complement)], ">=0 in every cell"),
         metric("maximum_complement_over_complete_margin", max(x["mean_margin"] / h["mean_margin"] for x, h in zip(complement, complete)), ">=0.70 in every cell")],
        "Registered independent-complement prediction was null; the exact all-other-source complement did not carry the task on its own.")
    asymmetry_null = _event(
        "task14_head11_3.fresh_subject_term.complement_asymmetry.complete.v1", "composition", "null", "scientific_null",
        [metric("plural_to_singular_complement_mean_margin_range", [min(c["complement_only"]["mean_margin"] for c in p2s), max(c["complement_only"]["mean_margin"] for c in p2s)], "<=-0.05 in both cells"),
         metric("singular_to_plural_complement_mean_margin_range", [min(c["complement_only"]["mean_margin"] for k, c in cells.items() if k.startswith("singular_to_plural")), max(c["complement_only"]["mean_margin"] for k, c in cells.items() if k.startswith("singular_to_plural"))], ">=0.05 in both cells")],
        "Registered complement-asymmetry prediction was null; complement effects did not meet the directional magnitude and sign bars.")
    localization = _event(
        "task14_head11_3.fresh_subject_term.exact_localization.complete.v1", "composition", "held", None,
        [metric("subject_over_complete_mean_margin_range", [min(margin_recovery), max(margin_recovery)], ">=0.90 in every cell"),
         metric("subject_over_complete_mean_CE_range", [min(ce_recovery), max(ce_recovery)], ">=0.90 in every cell"),
         metric("maximum_absolute_complement_mean_margin", max(abs(x["mean_margin"]) for x in complement), "descriptive: <0.05"),
         metric("maximum_absolute_complement_mean_CE", max(abs(x["mean_CE"]) for x in complement), "descriptive: <0.05"),
         metric("maximum_absolute_interaction_mean_margin", max(abs(x["mean_margin"]) for x in interaction), "descriptive: <0.005"),
         metric("maximum_absolute_interaction_mean_CE", max(abs(x["mean_CE"]) for x in interaction), "descriptive: <0.005")],
        "Directly supported by the same registered factorial: replacing the exact full subject source term p_8 u_8 reproduced 0.956–1.098 of complete-head margin and 0.954–1.103 of complete-head CE, while the exact complement and interaction were small. This localizes the effect to the full subject term; it does not establish value-vector semantics, necessity/removal, new OOD generalization, or downstream reuse.")
    events = [instrument, repair_null, complement_null, asymmetry_null, localization]

    revision = copy.deepcopy(base)
    revision.update({
        "claim_id": NEW_CLAIM, "revision": 12, "supersedes": BASE_CLAIM,
        "status": "site_live",
        "evidence_event_ids": base["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": (
            "The fresh licensed subject-term/complement factorial is complete and must not be repeated. It localizes the complete-head task effect to the exact full subject term p_8 u_8, while all three registered complement/interaction predictions were null. Next split the value side of that exact subject term to test what value-vector component carries the causal effect, then test which downstream readers reuse it. Necessity/selective removal and genuinely new OOD constructions remain open."
        ),
    })
    if not any(s["site_id"] == SITE_ID for s in revision["candidate_sites"]):
        revision["candidate_sites"].append({
            "site_id": SITE_ID,
            "tensor_path": "exact subject source term p_8 u_8 and exact all-other-source complement sum_{j != 8} p_j u_j",
            "shape": ["batch", 2, 1152],
            "intervention": "replace the subject term, complement, or both with the matched donor while retaining the other exact recipient component",
            "ceiling_event_ids": [],
        })
    family_id = "fresh_matched_subject_term_complement_v1"
    if not any(f["family_id"] == family_id for f in revision["counterfactual_families"]):
        revision["counterfactual_families"].append({
            "family_id": family_id, "role": "interchange",
            "changes": ["the exact subject source term p_8 u_8, exact all-other-source complement, or both between opposite subject numbers"],
            "holds_fixed": ["licensed HOLDOUT text and answer position", "the exact component not selected for replacement"],
            "builder_artifact_id": "task14_fresh_matched_capability_authority",
            "control_ids": ["same-batch native no-op and source-term sum", "complete opposite-head positive control"],
            "split_plan_id": SPLIT_ID, "status": "validated",
        })
    return {"schema": "task14_fresh_subject_term_publication_v12", "canonical_tag": TAG,
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
