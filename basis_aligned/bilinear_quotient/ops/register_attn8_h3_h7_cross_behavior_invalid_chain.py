#!/usr/bin/env python3
"""Dry-run-first publication of the invalid L8H3/H7 cross-behavior chain."""

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


SPEC = BQ / "circuits/fast_screens/attn8_h3_h7_cross_behavior_invalid_chain_publication_v1.json"
SITE_ID = "final_label_l0_value_through_l8h3_h7"
EVENT_NAMESPACES = {
    "task.numbered_list.index_successor": "numbered_list",
    "task.numeric_sequence.continuation": "numeric_sequence",
}


class PublicationError(RuntimeError):
    pass


def metric(name: str, estimate: Any, bar: str) -> dict[str, Any]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _read(relative: str) -> dict:
    return json.loads((BQ / relative).read_text())


def _artifact(relative: str, expected_hash: str, kind: str) -> dict[str, str]:
    path = BQ / relative
    actual = registry.file_sha256(path)
    if actual != expected_hash:
        raise PublicationError(f"hash mismatch for {path.name}: {actual} != {expected_hash}")
    return {"path": str(path.relative_to(REPO)), "sha256": actual,
            "kind": kind, "status": "frozen"}


def _minimum_v3_control_capability(result: dict) -> float:
    return min(
        arm["native_preference_accuracy"]
        for split in result["score"]["splits"].values()
        for family in split["controls"].values()
        for arm in family.values()
    )


def _v5_direction_wins(result: dict) -> list[float]:
    rows = [row for row in result["evidence"]
            if row["family_id"] == "list_step_two_conflict"]
    values = []
    for split in ("FINAL_TEST", "OOD"):
        for direction in ("lower_to_higher", "higher_to_lower"):
            cell = [row for row in rows if row["split"] == split
                    and row["direction"] == direction]
            values.append(sum(row["donor_answer_win"] for row in cell) / len(cell))
    return [min(values), max(values)]


def _events(spec: dict, destination: dict, artifacts: dict[str, dict]) -> list[dict]:
    checkpoint = spec["checkpoint_weights_sha256"]
    v2_raw = _read("circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v2_result.json")
    v2_audit = _read("circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v2_invalid_audit.json")
    v3 = _read("circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v3_control_repair_result.json")
    v4_raw = _read("circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls_result.json")
    v4_audit = _read("circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v4_semantic_role_invalid_audit.json")
    v5 = _read("circuits/fast_screens/attn8_h3_h7_cached_successor_final_ood_v1_result.json")
    expected = (
        (v2_raw, "attn8_h3_h7_cross_behavior_factor_interchange_result_v2", "generic_numeral_or_copy_bus", "attn8_h3_h7_cross_v2_prior"),
        (v3, "attn8_h3_h7_cross_behavior_factor_interchange_result_v3_control_repair", "invalid", "attn8_h3_h7_cross_v3_prior"),
        (v4_raw, "attn8_h3_h7_cross_behavior_factor_interchange_v4_canonical_controls_result", "broad_numeral_or_copy_service", "attn8_h3_h7_cross_v4_prior"),
        (v5, "attn8_h3_h7_cached_successor_final_ood_v1_result", "invalid", "attn8_h3_h7_cross_v5_prior"),
    )
    for result, schema, terminal, prior_id in expected:
        if result.get("schema") != schema or result.get("terminal") != terminal:
            raise PublicationError(f"schema/terminal mismatch for {schema}")
        if result.get("checkpoint_weights_sha256") != checkpoint:
            raise PublicationError(f"checkpoint mismatch for {schema}")
        if result.get("prior_art_sha256") != artifacts[prior_id]["sha256"]:
            raise PublicationError(f"prior-art binding mismatch for {schema}")
    if v2_audit.get("outcome") != "invalid" or v2_audit.get("result_sha256") != artifacts["attn8_h3_h7_cross_v2_raw"]["sha256"]:
        raise PublicationError("v2 invalid audit does not bind the raw result")
    if v4_audit.get("terminal") != "invalid" or v4_audit.get("result", {}).get("sha256") != artifacts["attn8_h3_h7_cross_v4_raw"]["sha256"]:
        raise PublicationError("v4 invalid audit does not bind the raw result")

    common = {
        "claim_id": destination["base_claim_id"], "site_id": SITE_ID,
        "split_plan_id": destination["split_plan_id"], "seed": None,
        "checkpoint_sha256": checkpoint, "replicates_event_id": None,
        "sections": [], "stage": "invalid", "verdict": "invalid",
        "failure_kind": "invalid_instrument",
        "family_ids": destination["family_ids"],
    }
    definitions = [
        ("attn8_h3_h7_cross_behavior.v2.invalid_implementation.v1", "composition", "FIT_and_SELECT_synthetic_controls",
         "attn8_h3_h7_cross_v2_prior", "attn8_h3_h7_cross_v2_invalid_audit", ["attn8_h3_h7_cross_v2_raw"], None,
         [metric("minimum_step_two_native_capability", min(v2_audit["registered_gate_that_failed"]["fit_step_two"], v2_audit["registered_gate_that_failed"]["select_step_two"]), ">=0.85"),
          metric("minimum_target_within_joint_recovery", v2_audit["controls_that_passed"]["target_within_joint_recovery_minimum"], ">=0.50")],
         "Implementation-invalid: control native capability was omitted from the instrument gate. The raw emitted label is void."),
        ("attn8_h3_h7_cross_behavior.v3.invalid_capability.v1", "composition", "FIT_and_SELECT_control_capability_repair",
         "attn8_h3_h7_cross_v3_prior", "attn8_h3_h7_cross_v3_result", [], "attn8_h3_h7_cross_behavior.v2.invalid_implementation.v1",
         [metric("minimum_control_native_capability", _minimum_v3_control_capability(v3), ">=0.85 in every control cell"),
          metric("exact_factor_algebra", float(v3["score"]["head_source_sum_relative_squared_error"] <= 1e-10 and v3["score"]["value_split_relative_squared_error"] <= 1e-10), "1 required")],
         "Instrument-invalid: the repaired scorer correctly stopped on incapable controls. No threshold or row repair is licensed."),
        ("attn8_h3_h7_cross_behavior.v4.invalid_semantic_role.v1", "composition", "FIT_and_SELECT_canonical_controls",
         "attn8_h3_h7_cross_v4_prior", "attn8_h3_h7_cross_v4_invalid_audit", ["attn8_h3_h7_cross_v4_raw"], "attn8_h3_h7_cross_behavior.v3.invalid_capability.v1",
         [metric("semantic_role_registration_valid", 0.0, "1 required"),
          metric("exact_factor_algebra", float(max(v4_audit["exact_checks"].values()) <= 1e-10), "1 required"),
          metric("step_two_cached_donorward_fraction_range", [v4_audit["descriptive_only"]["list_step_two_cached_donorward_fraction_fit"], v4_audit["descriptive_only"]["list_step_two_cached_donorward_fraction_select"]], "descriptive only")],
         "Semantic-role invalid: step-two was incorrectly registered as a negative control although it is a positive instance of the same successor rule. The raw label is void."),
        ("attn8_h3_h7_cross_behavior.v5.invalid_ood_instrument.v1", "ood", "FINAL_TEST_and_OOD_cached_value_only",
         "attn8_h3_h7_cross_v5_prior", "attn8_h3_h7_cross_v5_result", [], "attn8_h3_h7_cross_behavior.v4.invalid_semantic_role.v1",
         [metric("minimum_OOD_word_copy_native_capability", min(v5["score"]["family_reports"]["sequence_word_copy_control"]["OOD"][key] for key in ("recipient_native_accuracy", "donor_native_accuracy")), ">=0.85"),
          metric("step_two_donor_answer_win_fraction", min(v5["score"]["family_reports"]["list_step_two_conflict"][split]["donor_answer_win_fraction"] for split in ("FINAL_TEST", "OOD")), ">=0.60 in both directions"),
          metric("step_two_direction_specific_donor_answer_win_range", _v5_direction_wins(v5), "descriptive; require bidirectional support"),
          metric("minimum_step_two_donorward_fraction", min(v5["score"]["family_reports"]["list_step_two_conflict"][split]["donorward_fraction"] for split in ("FINAL_TEST", "OOD")), ">=0.75"),
          metric("minimum_intervention_norm_fraction_of_target", min(report["median_intervention_norm_fraction_of_target"] for family in v5["score"]["family_reports"].values() for report in family.values()), ">=0.10 in every family and split"),
          metric("exact_and_live", float(max(v5["score"][key] for key in ("native_replay_relative_squared_error", "head_source_sum_relative_squared_error", "value_split_relative_squared_error", "installed_term_max_absolute_error")) <= 1e-10), "1 required")],
         "Invalid held-out instrument: exact/live intervention, but OOD word-copy was incapable and step-two donor-answer transfer was directionally asymmetric. No scientific verdict and no threshold/row repair."),
    ]
    namespace = EVENT_NAMESPACES[destination["canonical_tag"]]
    events = []
    for event_id, test_type, role, prior, result, inputs, supersedes, metrics, notes in definitions:
        events.append({**common, "event_id": f"{namespace}.{event_id}", "test_type": test_type,
                       "evaluation_role": role, "metrics": metrics,
                       "prereg_artifact_id": prior, "result_artifact_id": result,
                       "input_artifact_ids": inputs,
                       "supersedes_event_id": (
                           f"{namespace}.{supersedes}" if supersedes else None
                       ), "notes": notes})
    events[0]["failure_kind"] = "implementation_failure"
    return events


def _event_with_keys(record: dict, event: dict) -> dict:
    output = dict(event)
    output["design_key"] = registry.design_key(record, output)
    output["execution_key"] = registry.execution_key(record, output)
    return output


def _migrate_interrupted_dual_apply(record: dict, destination: dict) -> dict:
    """Repair only the exact uncommitted state made by the first dual apply.

    The first implementation used identical event IDs in both dossiers.  Both
    dossier writes completed, then the global experiment-index rebuild rejected
    them.  This migration namespaces those four already-written invalid events;
    it does not remove evidence or change any metric, verdict, or claim status.
    """
    namespace = EVENT_NAMESPACES[destination["canonical_tag"]]
    expected = {event["event_id"].removeprefix(f"{namespace}."): event["event_id"]
                for event in destination["events"]}
    present = {event["event_id"] for event in record["evidence_events"]}
    legacy_present = set(expected) & present
    if not legacy_present:
        return record
    if legacy_present != set(expected):
        raise PublicationError("partial legacy event set is not safe to migrate")
    migrated = copy.deepcopy(record)
    for event in migrated["evidence_events"]:
        if event["event_id"] in expected:
            event["event_id"] = expected[event["event_id"]]
            old_parent = event.get("supersedes_event_id")
            if old_parent in expected:
                event["supersedes_event_id"] = expected[old_parent]
            event.pop("design_key", None)
            event.pop("execution_key", None)
            event.update(_event_with_keys(migrated, event))
    for claim in migrated["claims"]:
        claim["evidence_event_ids"] = [expected.get(item, item)
                                       for item in claim["evidence_event_ids"]]
    registry.validate_v2(migrated)
    return migrated


def build_plan(spec: dict | None = None) -> dict[str, Any]:
    spec = copy.deepcopy(spec or json.loads(SPEC.read_text()))
    if spec.get("schema") != "attn8_h3_h7_cross_behavior_invalid_chain_publication_spec_v1":
        raise PublicationError("wrong publication schema")
    artifacts = {artifact_id: _artifact(path, digest, kind)
                 for artifact_id, path, digest, kind in spec["artifacts"]}
    destinations = []
    for destination in spec["destinations"]:
        record = json.loads(registry.circuit_path(destination["canonical_tag"]).read_text())
        if record["claims"][-1]["claim_id"] not in {destination["base_claim_id"], destination["new_claim_id"]}:
            raise PublicationError(f"canonical base moved for {destination['canonical_tag']}")
        base = copy.deepcopy(next(claim for claim in record["claims"]
                                  if claim["claim_id"] == destination["base_claim_id"]))
        events = _events(spec, destination, artifacts)
        base.update({"claim_id": destination["new_claim_id"],
                     "revision": destination["new_revision"],
                     "supersedes": destination["base_claim_id"],
                     "evidence_event_ids": base["evidence_event_ids"] + [event["event_id"] for event in events],
                     "next_missing": spec["next_missing"]})
        destinations.append({"canonical_tag": destination["canonical_tag"],
                             "artifacts": artifacts, "events": events,
                             "claim_revision": base})
    return {"schema": spec["schema"], "destinations": destinations}


def _preview(destination: dict) -> dict:
    record = json.loads(registry.circuit_path(destination["canonical_tag"]).read_text())
    record = _migrate_interrupted_dual_apply(record, destination)
    for artifact_id, artifact in destination["artifacts"].items():
        if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != artifact:
            raise PublicationError(f"artifact collision: {artifact_id}")
        record["artifacts"][artifact_id] = artifact
    for event in destination["events"]:
        expected = _event_with_keys(record, event)
        found = [old for old in record["evidence_events"] if old["event_id"] == event["event_id"]]
        if found and found != [expected]:
            raise PublicationError(f"event collision: {event['event_id']}")
        if not found:
            record["evidence_events"].append(expected)
    revision = destination["claim_revision"]
    found = [claim for claim in record["claims"] if claim["claim_id"] == revision["claim_id"]]
    if found and found != [revision]:
        raise PublicationError(f"claim collision: {revision['claim_id']}")
    if not found:
        record["claims"].append(revision)
    registry.validate_v2(record)
    return record


def apply_plan(plan: dict, *, regenerate: bool = True) -> list[Path]:
    for destination in plan["destinations"]:
        path = registry.circuit_path(destination["canonical_tag"])
        current = json.loads(path.read_text())
        migrated = _migrate_interrupted_dual_apply(current, destination)
        if migrated != current:
            registry._atomic_json(path, migrated)
    for destination in plan["destinations"]:
        _preview(destination)
    paths = []
    for destination in plan["destinations"]:
        tag = destination["canonical_tag"]
        registry.append_artifacts(tag, destination["artifacts"])
        for event in destination["events"]:
            current = json.loads(registry.circuit_path(tag).read_text())
            if not any(old["event_id"] == event["event_id"] for old in current["evidence_events"]):
                registry.append_evidence_event(tag, event)
        current = json.loads(registry.circuit_path(tag).read_text())
        revision = destination["claim_revision"]
        if not any(claim["claim_id"] == revision["claim_id"] for claim in current["claims"]):
            registry.append_claim_revision(tag, revision)
        path = registry.circuit_path(tag)
        registry.validate_v2(json.loads(path.read_text()))
        paths.append(path)
    registry.rebuild_registry_v2()
    if regenerate:
        for script in ("make_circuit_coverage.py", "make_circuit_experiment_index.py", "make_circuit_campaign_queue.py"):
            subprocess.run([sys.executable, str(BQ / script)], cwd=REPO, check=True)
    return paths


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
