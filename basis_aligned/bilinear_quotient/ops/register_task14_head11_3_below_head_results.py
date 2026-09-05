#!/usr/bin/env python3
"""Dry-run-first publication of the audited Task14 below-head evidence bundle."""

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


SPEC = BQ / "circuits/fast_screens/task14_head11_3_below_head_v1_publication.json"
RESULT_DIR = BQ / "circuits/fast_screens"
PRIOR_DIR = BQ / "circuits/prior_art"
RESULT_PATHS = {
    "subject_factor_select": "task14_head11_3_subject_attractor_score_payload_factorial_v1_result.json",
    "subject_value_specificity_select": "task14_head11_3_subject_payload_number_specificity_v1_result.json",
    "subject_value_test": "task14_head11_3_subject_payload_test_transfer_v1_result.json",
    "subject_value_lemma_direction_test": "task14_head11_3_subject_payload_lemma_direction_factorial_v1_result.json",
    "score_context_test_invalid": "task14_head11_3_subject_score_context_gate_factorial_v1_result.json",
    "score_context_test": "task14_head11_3_subject_score_context_gate_factorial_v2_result.json",
    "same_syntax_value_atlas_test": "task14_head11_3_same_syntax_source_value_atlas_v1_result.json",
    "value_group_test_invalid": "task14_head11_3_source_value_role_group_factorial_v1_result.json",
    "value_group_test": "task14_head11_3_source_value_role_group_factorial_v2_result.json",
    "ood_value_atlas_invalid": "task14_head11_3_ood_same_syntax_source_value_atlas_v1_result.json",
    "ood_value_atlas": "task14_head11_3_ood_same_syntax_source_value_atlas_v2_result.json",
    "ood_score_role_invalid": "task14_head11_3_ood_fronted_score_role_factorial_v1_result.json",
    "ood_score_role": "task14_head11_3_ood_fronted_score_role_factorial_v2_result.json",
    "ood_self_qk": "task14_head11_3_ood_fronted_self_qk_factorial_v1_result.json",
    "ood_natural_qk_specificity": "task14_head11_3_ood_fronted_natural_qk_number_specificity_v1_result.json",
    "fresh_natural_qk_specificity_invalid": "task14_head11_3_fresh_fronted_natural_qk_number_specificity_v1_result.json",
}
REPAIRS = {
    "score_context_test": "score_context_test_invalid",
    "value_group_test": "value_group_test_invalid",
    "ood_value_atlas": "ood_value_atlas_invalid",
    "ood_score_role": "ood_score_role_invalid",
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _cell_values(result: dict, path: tuple[str, ...]) -> list[float]:
    values = []
    for cell in result["score"]["cells"].values():
        value: Any = cell
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def decisive_metrics(key: str, result: dict) -> list[dict[str, Any]]:
    """Recompute the small searchable contract from per-condition/cell scores."""
    score = result["score"]
    if key == "subject_factor_select":
        conditions = score["conditions"]
        return [
            metric("subject_value_margin_recovery", conditions["subject_payload"]["margin_recovery_of_complete_head"], ">=0.25"),
            metric("subject_score_margin_recovery", conditions["subject_score"]["margin_recovery_of_complete_head"], "diagnostic"),
            metric("maximum_absolute_attractor_margin_recovery", max(abs(conditions[x]["margin_recovery_of_complete_head"]) for x in ("attractor_score", "attractor_payload", "attractor_joint")), "diagnostic"),
        ]
    if key == "subject_value_specificity_select":
        return [
            metric("opposite_number_mean_directed_margin_effect", score["opposite_number"]["mean_directed_margin_effect"], ">0"),
            metric("same_number_absolute_margin_leakage_fraction", score["same_number"]["absolute_margin_leakage_over_live_effect"], "<=0.10"),
            metric("same_number_absolute_ce_leakage_fraction", score["same_number"]["absolute_ce_leakage_over_live_effect"], "<=0.10"),
        ]
    if key == "subject_value_test":
        recoveries = _cell_values(result, ("subject_payload", "margin_recovery_of_complete_head"))
        passed = sum(bool(c["subject_payload"]["passed"]) for c in score["cells"].values())
        return [metric("passing_subject_value_cells", passed, "4 of 4"), metric("minimum_subject_value_margin_recovery", min(recoveries), ">=0.25 each cell")]
    if key == "subject_value_lemma_direction_test":
        weak = score["cells"]["pp_plural_to_relative_singular"]
        gaps = [abs(c["same_lemma_payload"]["margin_recovery_of_complete_head"] - c["cross_noun_payload"]["margin_recovery_of_complete_head"]) for c in score["cells"].values()]
        return [metric("weak_cell_same_lemma_margin_recovery", weak["same_lemma_payload"]["margin_recovery_of_complete_head"], ">=0.25 for rescue"), metric("maximum_same_lemma_cross_noun_recovery_gap", max(gaps), "diagnostic")]
    if key.startswith("score_context_test"):
        if result["terminal"] == "invalid":
            return [metric("instrument_live", float(score["predictions"]["pred_a_instrument_live"]), "1 required"), metric("minimum_weak_cell_relative_score_contrast_fraction", min(v["fraction_above_relative_threshold"] for v in score["score_contrast"].values()), ">=0.75")]
        interactions = _cell_values(result, ("score_by_value_interaction", "margin_interaction_recovery_of_complete_head"))
        return [metric("maximum_absolute_score_value_interaction_recovery", max(abs(v) for v in interactions), "<=0.05 for context-independent null"), metric("rescued_weak_cell", float(score["predictions"]["pred_c_opposite_syntax_score_rescues_weak_cell"]), "1 for rescue")]
    if key == "same_syntax_value_atlas_test":
        joint = _cell_values(result, ("joint_all_values", "margin_recovery_of_complete_head"))
        strongest_non_subject = []
        for cell in score["cells"].values():
            strongest_non_subject.append(max(abs(s["margin_recovery_of_complete_head"]) for s in cell["sources"] if s["semantic_role"] != "subject"))
        return [metric("joint_value_margin_recovery_range", [min(joint), max(joint)], ">=0.80 each cell"), metric("minimum_strongest_non_subject_source_recovery", min(strongest_non_subject), ">=0.10")]
    if key.startswith("value_group_test"):
        if result["terminal"] == "invalid":
            return [metric("empty_subset_max_absolute_logit_error", score["empty_subset_max_absolute_logit_error"], "<=0.00005"), metric("instrument_live", float(score["predictions"]["pred_a_instrument_live"]), "1 required")]
        si = _cell_values(result, ("subsets", "SI", "margin_recovery_of_joint_values"))
        sib = _cell_values(result, ("subsets", "SIB", "margin_recovery_of_joint_values"))
        return [metric("minimum_SI_margin_recovery_of_joint_values", min(si), ">=0.70 each cell"), metric("minimum_SIB_margin_recovery_of_joint_values", min(sib), ">=0.80 each cell"), metric("bridge_repairs_failed_SI_cells", float(score["predictions"]["pred_c_bridge_repairs_failed_SI_cells"]), "1")]
    if key.startswith("ood_value_atlas"):
        if result["terminal"] == "invalid":
            return [metric("native_noop_max_absolute_logit_error", score["native_noop_max_absolute_logit_error"], "<=0.00005"), metric("instrument_live", float(score["predictions"]["pred_a_instrument_live"]), "1 required")]
        joint = _cell_values(result, ("joint_all_values", "margin_recovery_of_complete_head"))
        fronted = [c for c in score["cells"].values() if c["target_family"] == "A1"]
        subject = [next(s for s in c["sources"] if s["semantic_role"] == "subject")["margin_recovery_of_complete_head"] for c in fronted]
        return [metric("OOD_joint_value_margin_recovery_range", [min(joint), max(joint)], ">=0.80 each cell"), metric("fronted_subject_margin_recovery_range", [min(subject), max(subject)], ">=0.80 each direction"), metric("two_attractor_relative_later_relay", float(score["predictions"]["pred_d_two_attractor_relative_later_relay"]), "1")]
    if key.startswith("ood_score_role"):
        if result["terminal"] == "invalid":
            return [metric("native_corner_max_absolute_logit_error", score["native_corner_max_absolute_logit_error"], "<=0.00005"), metric("instrument_live", float(score["predictions"]["pred_a_instrument_live"]), "1 required")]
        self_routes = _cell_values(result, ("self_S_route", "mean_margin_contribution"))
        earlier = _cell_values(result, ("earlier_E_plus_D_route", "mean_margin_contribution"))
        return [metric("self_score_sufficiency", float(score["predictions"]["pred_c_self_score_sufficiency"]), "1"), metric("self_score_margin_contribution_range", [min(self_routes), max(self_routes)], "direction-conditional"), metric("maximum_absolute_earlier_score_margin_contribution", max(abs(v) for v in earlier), "diagnostic")]
    if key == "ood_self_qk":
        qk1 = _cell_values(result, ("qk1_pair", "signed_recovery"))
        qk2 = _cell_values(result, ("qk2_pair", "signed_recovery"))
        interaction = _cell_values(result, ("branch_interaction", "absolute_fraction_of_total"))
        return [metric("qk1_pair_signed_recovery_range", [min(qk1), max(qk1)], ">=0.70 each direction for sufficiency"), metric("qk2_pair_signed_recovery_range", [min(qk2), max(qk2)], ">=0.70 each direction for sufficiency"), metric("branch_interaction_absolute_fraction_range", [min(interaction), max(interaction)], ">=0.10 each direction")]
    if key == "ood_natural_qk_specificity":
        cells = list(score["cells"].values())
        joint_leakage = [cell["effects"]["opposite_joint"]["same_number_leakage_ratio"] for cell in cells]
        qk1_fraction = [cell["effects"]["opposite_qk1"]["signed_fraction_of_joint"] for cell in cells]
        qk2_fraction = [cell["effects"]["opposite_qk2"]["signed_fraction_of_joint"] for cell in cells]
        return [
            metric("joint_same_number_leakage_ratio_range", [min(joint_leakage), max(joint_leakage)], "<=0.25 each direction"),
            metric("qk1_signed_fraction_of_joint_range", [min(qk1_fraction), max(qk1_fraction)], ">=0.20 with >=0.75 row sign each direction"),
            metric("qk2_signed_fraction_of_joint_range", [min(qk2_fraction), max(qk2_fraction)], ">=0.20 with >=0.75 row sign each direction"),
        ]
    if key == "fresh_natural_qk_specificity_invalid":
        cells = list(score["cells"].values())
        native_counts = [
            count
            for cell in cells
            for count in cell["native_correct_of_8"].values()
        ]
        row_counts = [
            round(cell["effects"]["opposite_joint"]["expected_margin_sign_fraction"] * 8)
            for cell in cells
        ]
        return [
            metric("minimum_opposite_joint_expected_row_sign_count_of_8", min(row_counts), ">=6 of 8 in every direction-by-template cell"),
            metric("minimum_native_correct_count_of_8", min(native_counts), ">=7 of 8 for every recipient/donor role in every cell"),
            metric("native_replay_max_absolute_logit_error", score["native_replay_max_absolute_logit_error"], "<=0.00007"),
            metric("source_term_identity_max_absolute_error", score["source_term_identity_max_absolute_error"], "<=0.00005"),
            metric("direct_score_identity_max_absolute_error", score["direct_score_identity_max_absolute_error"], "<=0.000005"),
            metric("installed_term_max_absolute_error", score["installed_term_max_absolute_error"], "<=0.00005"),
        ]
    raise PublicationError(f"no decisive-metric reducer for {key}")


def _artifact(path: Path, expected_hash: str, kind: str) -> dict[str, str]:
    actual = registry.file_sha256(path)
    if actual != expected_hash:
        raise PublicationError(f"hash mismatch for {path.name}: {actual} != {expected_hash}")
    return {"path": str(path.relative_to(REPO)), "sha256": actual, "kind": kind, "status": "frozen"}


def build_plan(spec: dict | None = None) -> dict[str, Any]:
    spec = copy.deepcopy(spec or json.loads(SPEC.read_text()))
    if spec.get("schema") != "task14_head11_3_below_head_publication_spec_v1":
        raise PublicationError("wrong publication schema")
    record = json.loads(registry.circuit_path(spec["canonical_tag"]).read_text())
    if record["claims"][-1]["claim_id"] not in {spec["base_claim_id"], spec["new_claim_id"]}:
        raise PublicationError("canonical Task14 base claim moved; review instead of rebasing silently")
    artifacts: dict[str, dict] = {}
    events = []
    event_ids: dict[str, str] = {}
    for entry in spec["entries"]:
        (key, result_hash, result_schema, terminal, prior_hash, prior_slug, role,
         test_type, verdict, failure_kind, site_id) = entry
        result_path = RESULT_DIR / RESULT_PATHS[key]
        prior_path = PRIOR_DIR / f"task14_head11_3_{prior_slug}.json"
        result = json.loads(result_path.read_text())
        prior = json.loads(prior_path.read_text())
        if result.get("schema") != result_schema or result.get("terminal") != terminal:
            raise PublicationError(f"schema/terminal mismatch for {key}")
        if result.get("checkpoint_weights_sha256") != spec["checkpoint_weights_sha256"]:
            raise PublicationError(f"checkpoint mismatch for {key}")
        if result.get("evaluated_splits") != [role] or result.get("forbidden_splits_opened") != []:
            raise PublicationError(f"split-scope mismatch for {key}")
        if result.get("prior_art_sha256") != prior_hash:
            raise PublicationError(f"result does not bind expected prior-art receipt for {key}")
        if not prior.get("candidate_id"):
            raise PublicationError(f"malformed prior-art receipt for {key}")
        result_id, prior_id = f"task14_below_head_{key}_result", f"task14_below_head_{key}_prior_art"
        artifacts[result_id] = _artifact(result_path, result_hash, "screen_result")
        artifacts[prior_id] = _artifact(prior_path, prior_hash, "preregistration")
        event_id = f"task14_head11_3.below_head.{key}.{'invalid' if verdict == 'invalid' else 'complete'}.v1"
        event_ids[key] = event_id
        repair = REPAIRS.get(key)
        events.append({
            "event_id": event_id,
            "claim_id": spec["base_claim_id"],
            "test_type": test_type,
            "stage": "invalid" if verdict == "invalid" else "complete",
            "verdict": verdict,
            "failure_kind": failure_kind,
            "family_ids": [],
            "site_id": site_id,
            "split_plan_id": None,
            "evaluation_role": role,
            "metrics": decisive_metrics(key, result),
            "prereg_artifact_id": prior_id,
            "result_artifact_id": result_id,
            "input_artifact_ids": [],
            "seed": None,
            "checkpoint_sha256": result["checkpoint_weights_sha256"],
            "supersedes_event_id": event_ids.get(repair),
            "replicates_event_id": None,
            "sections": [],
            "notes": (
                spec.get("entry_notes", {}).get(
                    key,
                    "Numerically or instrument-invalid attempt retained as provenance; it is not scientific evidence.",
                )
                if verdict == "invalid" else
                f"Below-head Task14 evidence on {role}. The result bytes and prior-art receipt are frozen; reused TEST/OOD text is not a pristine held-out confirmation."
            ),
        })
    previous = copy.deepcopy(next(c for c in record["claims"] if c["claim_id"] == spec["base_claim_id"]))
    previous.update({
        "claim_id": spec["new_claim_id"], "revision": spec.get("new_revision", 9), "supersedes": spec["base_claim_id"],
        "status": "site_live",
        "evidence_event_ids": previous["evidence_event_ids"] + [e["event_id"] for e in events],
        "next_missing": spec.get("next_missing", (
            "Confirm the below-head subject-value, source-value-group, self-score, QK-branch, and natural-number "
            "specificity findings on fresh counterfactual text; test selective removal and upstream reuse. Current "
            "TEST/OOD analyses reuse text, the subject-value effect is direction-dependent, and neither QK branch "
            "alone is sufficient in the exact four-factor interchange."
        )),
    })
    known_sites = {site["site_id"] for site in previous["candidate_sites"]}
    site_specs = {
        "attention.block11.head3.subject_source.score_and_value.final_position": ("exact subject-source score and projected value factors", ["batch", 2, 1152], "replace the subject-source score, projected value, or both while retaining every other exact source term"),
        "attention.block11.head3.subject_source.projected_value.final_position": ("exact subject-source projected value contribution p_subject*u_subject", ["batch", 1152], "replace only the subject source projected value while retaining recipient source scores"),
        "attention.block11.head3.subject_source.score_by_value.final_position": ("factorial interaction of the subject-source score and projected value", ["batch", 2, 2, 1152], "cross recipient or alternate-context scores with native or opposite-number subject projected values"),
        "attention.block11.head3.all_source.projected_values.final_position": ("exact per-source projected value contributions p_k*u_k", ["batch", "source_position", 1152], "replace selected source projected values while retaining recipient source scores"),
        "attention.block11.head3.source_value_role_groups.final_position": ("semantic groups of exact source projected value contributions", ["batch", 4, 1152], "replace any subset of subject, intermediate, bridge, and attractor groups"),
        "attention.block11.head3.fronted_source_score_roles.final_position": ("fronted-construction source score vector grouped as earlier, determiner, and subject-self scores", ["batch", 3], "replace selected source scores with donor scores while holding the installed subject value fixed"),
        "attention.block11.head3.fronted_subject_self_score.qk_factors.final_position": ("the two bilinear QK branches composing the final-query to subject-self score", ["batch", 4, 128], "interchange q1, k1, q2, and k2 factors factorially while holding the donor subject value fixed"),
    }
    for site_id, (tensor_path, shape, intervention) in site_specs.items():
        if site_id not in known_sites:
            previous["candidate_sites"].append({"site_id": site_id, "tensor_path": tensor_path, "shape": shape, "intervention": intervention, "ceiling_event_ids": []})
    return {"schema": spec["schema"], "canonical_tag": spec["canonical_tag"], "artifacts": artifacts, "events": events, "claim_revision": previous}


def _event_with_keys(record: dict, event: dict) -> dict:
    out = dict(event)
    out["design_key"] = registry.design_key(record, out)
    out["execution_key"] = registry.execution_key(record, out)
    return out


def apply_plan(plan: dict, *, regenerate: bool = True) -> Path:
    """Apply one idempotent artifacts/events/claim prefix; callers should dry-run first."""
    tag = plan["canonical_tag"]
    # Validate the whole proposed prefix in memory before the first write.
    preview = json.loads(registry.circuit_path(tag).read_text())
    for artifact_id, artifact in plan["artifacts"].items():
        if artifact_id in preview["artifacts"] and preview["artifacts"][artifact_id] != artifact:
            raise PublicationError(f"artifact id collision: {artifact_id}")
        preview["artifacts"][artifact_id] = artifact
    for event in plan["events"]:
        expected = _event_with_keys(preview, event)
        found = [item for item in preview["evidence_events"] if item["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event id collision: {event['event_id']}")
        if not found:
            preview["evidence_events"].append(expected)
    revision = plan["claim_revision"]
    found = [item for item in preview["claims"] if item["claim_id"] == revision["claim_id"]]
    if found and found != [revision]:
        raise PublicationError("claim revision id collision")
    if not found:
        preview["claims"].append(revision)
    known_sites = {site["site_id"] for site in revision["candidate_sites"]}
    missing_sites = {event["site_id"] for event in plan["events"]} - known_sites
    if missing_sites:
        raise PublicationError(f"event sites absent from claim revision: {sorted(missing_sites)}")
    registry.validate_v2(preview)
    registry.append_artifacts(tag, plan["artifacts"])
    path = registry.circuit_path(tag)
    for event in plan["events"]:
        current = json.loads(path.read_text())
        expected = _event_with_keys(current, event)
        found = [item for item in current["evidence_events"] if item["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event id collision: {event['event_id']}")
        if not found:
            registry.append_evidence_event(tag, event)
    current = json.loads(path.read_text())
    revision = plan["claim_revision"]
    found = [item for item in current["claims"] if item["claim_id"] == revision["claim_id"]]
    if found and found != [revision]:
        raise PublicationError("claim revision id collision")
    if not found:
        registry.append_claim_revision(tag, revision)
    registry.validate_v2(json.loads(path.read_text()))
    if regenerate:
        for script in ("make_circuit_coverage.py", "make_circuit_experiment_index.py", "make_circuit_campaign_queue.py"):
            subprocess.run([sys.executable, str(BQ / script)], cwd=REPO, check=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="?", type=Path, default=SPEC)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_plan(json.loads(args.spec.read_text()))
    if args.apply:
        apply_plan(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
