#!/usr/bin/env python3
# BQLANE: cpu
"""Publish a reviewed fast screen into an existing canonical v2 circuit record.

Fast screens remain screen-tier evidence.  This adapter never invents a behavior
record or guesses semantic mappings: a small literal JSON spec must name the
existing claim, counterfactual families, split, and canonical site.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))

import circuit_registry_v2 as registry  # noqa: E402
import circuit_fast_screen_ledger as screen_ledger  # noqa: E402


SCHEMA = "circuit_fast_screen_publication_spec_v1"
CROSS_SYNTAX_SCHEMA = "circuit_cross_syntax_publication_spec_v1"
SPEC_FIELDS = {
    "schema", "result_path", "result_artifact_id", "canonical_tag",
    "source_claim_id", "event_id", "test_type", "transform_to_family_id",
    "split_plan_id", "result_site_id", "canonical_site", "claim_revision",
    "input_artifact_ids", "sections", "notes", "seed", "claim_ledger_policy",
}
SITE_FIELDS = {"site_id", "tensor_path", "shape", "intervention"}
REVISION_FIELDS = {"claim_id", "revision", "status", "next_missing"}
CROSS_SYNTAX_SPEC_FIELDS = (
    SPEC_FIELDS - {"transform_to_family_id"}
) | {"family_ids"}


class FastScreenPublishError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if type(spec) is not dict or set(spec) != SPEC_FIELDS or spec.get("schema") != SCHEMA:
        raise FastScreenPublishError("publication spec fields or schema changed")
    value = copy.deepcopy(spec)
    for field in (
        "result_path", "result_artifact_id", "canonical_tag", "source_claim_id",
        "event_id", "test_type", "split_plan_id", "result_site_id", "notes",
    ):
        if type(value[field]) is not str or not value[field].strip():
            raise FastScreenPublishError(f"{field} must be nonempty text")
    relative = Path(value["result_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise FastScreenPublishError("result_path must remain inside the bilinear-quotient root")
    mapping = value["transform_to_family_id"]
    if type(mapping) is not dict or set(mapping) != {"A1", "A2", "P", "C"}:
        raise FastScreenPublishError("transform_to_family_id must map exactly A1/A2/P/C")
    if any(type(item) is not str or not item for item in mapping.values()):
        raise FastScreenPublishError("canonical family ids must be nonempty strings")
    site = value["canonical_site"]
    if type(site) is not dict or set(site) != SITE_FIELDS:
        raise FastScreenPublishError("canonical_site fields changed")
    if any(not site.get(field) for field in ("site_id", "tensor_path", "shape", "intervention")):
        raise FastScreenPublishError("canonical_site is incomplete")
    revision = value["claim_revision"]
    if type(revision) is not dict or set(revision) != REVISION_FIELDS:
        raise FastScreenPublishError("claim_revision fields changed")
    if type(revision["revision"]) is not int or revision["revision"] <= 0:
        raise FastScreenPublishError("claim revision must be a positive integer")
    if revision["status"] not in registry.CLAIM_STATUSES:
        raise FastScreenPublishError("claim status is not registered")
    for field in ("claim_id", "next_missing"):
        if type(revision[field]) is not str or not revision[field].strip():
            raise FastScreenPublishError(f"claim_revision.{field} must be nonempty")
    for field in ("input_artifact_ids", "sections"):
        if type(value[field]) is not list or any(type(item) is not str or not item for item in value[field]):
            raise FastScreenPublishError(f"{field} must contain strings")
    if value["test_type"] not in registry.TEST_TYPES:
        raise FastScreenPublishError("test_type is not registered")
    if value["seed"] is not None and type(value["seed"]) not in {int, str}:
        raise FastScreenPublishError("seed must be null, integer, or string")
    if value["claim_ledger_policy"] not in {"legacy_no_claim_event"}:
        raise FastScreenPublishError("unsupported claim-ledger policy")
    return value


def _matching_ledger_entry(result: Mapping[str, Any], result_path: str, root: Path) -> dict:
    ledger_path = root / "circuits" / "fast_screen_ledger.jsonl"
    entries = screen_ledger.read_ledger(ledger_path, result_root=root)
    matches = [
        entry for entry in entries
        if entry["candidate_id"] == result.get("candidate_id")
        and entry["result_path"] == result_path
    ]
    if len(matches) != 1:
        raise FastScreenPublishError("result must have exactly one matching fast-screen ledger entry")
    entry = matches[0]
    comparisons = {
        "terminal": result.get("terminal"),
        "spec_sha256": result.get("spec_sha256"),
        "authority_sha256": result.get("authority_sha256"),
        "selected_site_id": result.get("selected_site_id"),
        "started_utc": result.get("started_utc"),
        "finished_utc": result.get("finished_utc"),
        "serial_seconds": result.get("serial_seconds"),
    }
    for field, observed in comparisons.items():
        if entry[field] != observed:
            raise FastScreenPublishError(f"result and ledger disagree on {field}")
    return entry


def _matching_cross_syntax_ledger_entry(
    result: Mapping[str, Any], result_path: str, root: Path,
) -> dict:
    entries = screen_ledger.read_ledger(
        root / "circuits" / "fast_screen_ledger.jsonl", result_root=root,
    )
    matches = [
        entry for entry in entries
        if entry["candidate_id"] == result.get("candidate_id")
        and entry["result_path"] == result_path
    ]
    if len(matches) != 1:
        raise FastScreenPublishError(
            "result must have exactly one matching fast-screen ledger entry"
        )
    entry = matches[0]
    comparisons = {
        "terminal": result.get("terminal"),
        "spec_sha256": result.get("plan_sha256"),
        "authority_sha256": result.get("authority_sha256"),
        "started_utc": result.get("started_utc"),
        "finished_utc": result.get("finished_utc"),
        "serial_seconds": result.get("serial_seconds"),
    }
    for field, observed in comparisons.items():
        if entry[field] != observed:
            raise FastScreenPublishError(f"result and ledger disagree on {field}")
    return entry


def _site_result(result: Mapping[str, Any], site_id: str) -> dict:
    run = result.get("run")
    if type(run) is not dict or type(run.get("site_results")) is not list:
        raise FastScreenPublishError("result lacks run.site_results")
    matches = [item for item in run["site_results"] if item.get("site", {}).get("site_id") == site_id]
    if len(matches) != 1 or matches[0].get("terminal") != "screen":
        raise FastScreenPublishError("published mechanistic site did not pass the frozen screen gates")
    if matches[0].get("site", {}).get("evidence_kind") == "residual":
        raise FastScreenPublishError("a residual ceiling cannot be promoted as the mechanistic site")
    return matches[0]


def build_plan(spec: Mapping[str, Any], *, root: Path = BQ) -> dict[str, Any]:
    """Validate all semantic bindings and return the exact append-only mutation plan."""
    value = _validate_spec(spec)
    result_path = root / value["result_path"]
    if not result_path.is_file() or result_path.is_symlink():
        raise FastScreenPublishError("result file is missing or unsafe")
    result = json.loads(result_path.read_text())
    if result.get("schema") != "circuit_fast_screen_result_v1" or result.get("terminal") != "screen":
        raise FastScreenPublishError("only a successful fast-screen v1 result can be promoted")
    ledger_entry = _matching_ledger_entry(result, value["result_path"], root)
    site_result = _site_result(result, value["result_site_id"])

    record_path = registry.circuit_path(value["canonical_tag"])
    if not record_path.is_file():
        raise FastScreenPublishError("canonical behavior record does not exist")
    record = json.loads(record_path.read_text())
    registry.validate_v2(record)
    source_claims = [claim for claim in record["claims"] if claim["claim_id"] == value["source_claim_id"]]
    if len(source_claims) != 1:
        raise FastScreenPublishError("source claim is missing or ambiguous")
    source_claim = source_claims[0]
    known_families = {family["family_id"] for family in source_claim["counterfactual_families"]}
    if not set(value["transform_to_family_id"].values()) <= known_families:
        raise FastScreenPublishError("publication maps a transform to an unknown canonical family")
    known_splits = {item["split_plan_id"] for item in record.get("split_plans", [])}
    if value["split_plan_id"] not in known_splits:
        raise FastScreenPublishError("publication names an unknown split plan")
    if not set(value["input_artifact_ids"]) <= set(record["artifacts"]):
        raise FastScreenPublishError("publication names an unknown input artifact")
    if value["claim_revision"]["revision"] <= max(claim["revision"] for claim in record["claims"]):
        existing = [
            claim for claim in record["claims"]
            if claim["claim_id"] == value["claim_revision"]["claim_id"]
        ]
        if not existing:
            raise FastScreenPublishError("new claim revision is not later than current history")

    capability = site_result.get("capability")
    if type(capability) is not list or not capability:
        raise FastScreenPublishError("published site lacks native capability evidence")
    eligible = [
        item for item in result["run"]["site_results"]
        if item.get("terminal") == "screen" and item.get("site", {}).get("evidence_kind") != "residual"
    ]
    metrics = [
        {"name": "minimum_native_cell_accuracy", "estimate": min(item["accuracy"] for item in capability),
         "ci95": None, "bar": "meet every frozen family-specific native capability bar"},
        {"name": "A1_mean_donor_recovery", "estimate": site_result["a1"]["mean_effect"],
         "ci95": None, "bar": ">=0.5 with direction fraction >=0.8"},
        {"name": "A2_mean_donor_recovery", "estimate": site_result["a2"]["mean_effect"],
         "ci95": None, "bar": ">=0.5 with direction fraction >=0.8"},
        {"name": "P_normalized_margin_movement", "estimate": site_result["p_invariance_effect"],
         "ci95": None, "bar": "<=0.2"},
        {"name": "C_absolute_recovery", "estimate": site_result["c_absolute_recovery"],
         "ci95": None, "bar": "<=0.35"},
        {"name": "passing_nonresidual_site_count", "estimate": len(eligible),
         "ci95": None, "bar": ">=1; screen-tier localization only"},
    ]
    artifact = {
        "path": str(result_path.relative_to(REPO)),
        "sha256": _sha256(result_path),
        "kind": "screen_result",
        "status": "frozen",
    }
    event = {
        "event_id": value["event_id"],
        "claim_id": value["source_claim_id"],
        "test_type": value["test_type"],
        "stage": "complete",
        "verdict": "held",
        "failure_kind": None,
        "family_ids": list(value["transform_to_family_id"].values()),
        "site_id": value["canonical_site"]["site_id"],
        "split_plan_id": value["split_plan_id"],
        "evaluation_role": "FIT_screen_only",
        "metrics": metrics,
        "prereg_artifact_id": None,
        "result_artifact_id": value["result_artifact_id"],
        "input_artifact_ids": value["input_artifact_ids"],
        "seed": value["seed"],
        "checkpoint_sha256": None,
        "supersedes_event_id": None,
        "replicates_event_id": None,
        "sections": value["sections"],
        "notes": value["notes"] + " Historical generic fast-screen output did not bind a checkpoint digest.",
    }
    revision = copy.deepcopy(source_claim)
    revision.update(value["claim_revision"])
    revision["supersedes"] = value["source_claim_id"]
    revision["evidence_event_ids"] = list(dict.fromkeys(
        source_claim.get("evidence_event_ids", []) + [value["event_id"]]
    ))
    sites = [site for site in revision["candidate_sites"] if site["site_id"] != value["canonical_site"]["site_id"]]
    sites.append({**value["canonical_site"], "ceiling_event_ids": [value["event_id"]]})
    revision["candidate_sites"] = sites
    return {
        "schema": "circuit_fast_screen_publication_plan_v1",
        "canonical_tag": value["canonical_tag"],
        "ledger_request_id": ledger_entry["request_id"],
        "result_artifact_id": value["result_artifact_id"],
        "artifact": artifact,
        "event": event,
        "claim_revision": revision,
        "claim_ledger_policy": value["claim_ledger_policy"],
    }


def build_cross_syntax_plan(
    spec: Mapping[str, Any], *, root: Path = BQ,
) -> dict[str, Any]:
    """Publish the one legacy cross-syntax screen without pretending it is OOD."""
    if type(spec) is not dict or set(spec) != CROSS_SYNTAX_SPEC_FIELDS \
            or spec.get("schema") != CROSS_SYNTAX_SCHEMA:
        raise FastScreenPublishError("cross-syntax publication spec fields or schema changed")
    value = copy.deepcopy(spec)
    # Reuse the v1 validator for every shared field and the append-only claim rules.
    validation_copy = copy.deepcopy(value)
    validation_copy["schema"] = SCHEMA
    validation_copy["transform_to_family_id"] = {
        "A1": value["family_ids"][0],
        "A2": value["family_ids"][1],
        "P": value["family_ids"][2],
        "C": value["family_ids"][2],
    } if type(value.get("family_ids")) is list and len(value["family_ids"]) == 3 else None
    validation_copy.pop("family_ids")
    _validate_spec(validation_copy)
    if len(set(value["family_ids"])) != 3:
        raise FastScreenPublishError("cross-syntax family_ids must contain three distinct families")

    result_path = root / value["result_path"]
    if not result_path.is_file() or result_path.is_symlink():
        raise FastScreenPublishError("result file is missing or unsafe")
    result = json.loads(result_path.read_text())
    scopes = {
        "task14_cross_syntax_interchange_result_v1": (
            "new_cross_syntax_relations_not_unseen_text",
            "FIT_VALIDATION_new_relations_not_unseen_text",
        ),
        "task14_targeted_cross_syntax_result_v1": (
            "unseen_nouns_and_prompt_templates_after_fit_site_selection",
            "SELECT_HELD_OUT_unseen_nouns_and_prompt_templates",
        ),
    }
    if result.get("schema") not in scopes or result.get("terminal") != "screen":
        raise FastScreenPublishError("only a successful targeted cross-syntax result can be promoted")
    expected_scope, evaluation_role = scopes[result["schema"]]
    if result.get("validation_scope") != expected_scope:
        raise FastScreenPublishError("cross-syntax validation scope changed")
    ledger_entry = _matching_cross_syntax_ledger_entry(result, value["result_path"], root)
    matches = [item for item in result.get("site_results", [])
               if item.get("site_id") == value["result_site_id"]]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise FastScreenPublishError("published cross-syntax site did not pass its frozen gates")
    site_result = matches[0]

    record_path = registry.circuit_path(value["canonical_tag"])
    if not record_path.is_file():
        raise FastScreenPublishError("canonical behavior record does not exist")
    record = json.loads(record_path.read_text())
    registry.validate_v2(record)
    source_claims = [claim for claim in record["claims"]
                     if claim["claim_id"] == value["source_claim_id"]]
    if len(source_claims) != 1:
        raise FastScreenPublishError("source claim is missing or ambiguous")
    source_claim = source_claims[0]
    known_families = {family["family_id"] for family in source_claim["counterfactual_families"]}
    if not set(value["family_ids"]) <= known_families:
        raise FastScreenPublishError("publication names an unknown canonical family")
    if value["split_plan_id"] not in {
        item["split_plan_id"] for item in record.get("split_plans", [])
    }:
        raise FastScreenPublishError("publication names an unknown split plan")
    if not set(value["input_artifact_ids"]) <= set(record["artifacts"]):
        raise FastScreenPublishError("publication names an unknown input artifact")

    cells = site_result.get("cells")
    capability = result.get("capability_cells")
    if type(cells) is not list or len(cells) != 4 or not all(c.get("passed") for c in cells):
        raise FastScreenPublishError("cross-syntax site lacks four passing direction cells")
    if type(capability) is not list or not capability or not all(c.get("passed") for c in capability):
        raise FastScreenPublishError("cross-syntax result lacks native capability evidence")
    passing = [item for item in result["site_results"] if item.get("passed") is True]
    metrics = [
        {"name": "minimum_native_cell_accuracy", "estimate": min(
            min(item["target_accuracy"], item["donor_accuracy"]) for item in capability
        ), "ci95": None, "bar": ">=0.85 in every syntax/number cell"},
        {"name": "cross_syntax_mean_donor_recovery",
         "estimate": site_result["overall_mean_recovery"], "ci95": None, "bar": ">=0.4"},
        {"name": "minimum_cross_syntax_cell_recovery",
         "estimate": min(item["mean_recovery"] for item in cells),
         "ci95": None, "bar": ">=0.4 in every direction cell"},
        {"name": "minimum_cross_syntax_direction_fraction",
         "estimate": min(item["direction_fraction"] for item in cells),
         "ci95": None, "bar": ">=0.75 in every direction cell"},
        {"name": "passing_preselected_site_count", "estimate": len(passing),
         "ci95": None, "bar": ">=1; two preselected sites only"},
    ]
    artifact = {
        "path": str(result_path.relative_to(REPO)),
        "sha256": _sha256(result_path), "kind": "screen_result", "status": "frozen",
    }
    event = {
        "event_id": value["event_id"], "claim_id": value["source_claim_id"],
        "test_type": value["test_type"], "stage": "complete", "verdict": "held",
        "failure_kind": None, "family_ids": value["family_ids"],
        "site_id": value["canonical_site"]["site_id"],
        "split_plan_id": value["split_plan_id"],
        "evaluation_role": evaluation_role,
        "metrics": metrics, "prereg_artifact_id": None,
        "result_artifact_id": value["result_artifact_id"],
        "input_artifact_ids": value["input_artifact_ids"], "seed": value["seed"],
        "checkpoint_sha256": result.get("checkpoint", {}).get("weights_sha256"),
        "supersedes_event_id": None,
        "replicates_event_id": None, "sections": value["sections"],
        "notes": value["notes"] + (
            " Historical result did not bind a checkpoint digest."
            if result.get("checkpoint", {}).get("weights_sha256") is None else ""
        ),
    }
    revision = copy.deepcopy(source_claim)
    revision.update(value["claim_revision"])
    revision["supersedes"] = value["source_claim_id"]
    revision["evidence_event_ids"] = list(dict.fromkeys(
        source_claim.get("evidence_event_ids", []) + [value["event_id"]]
    ))
    sites = [site for site in revision["candidate_sites"]
             if site["site_id"] != value["canonical_site"]["site_id"]]
    sites.append({**value["canonical_site"], "ceiling_event_ids": [value["event_id"]]})
    revision["candidate_sites"] = sites
    return {
        "schema": "circuit_fast_screen_publication_plan_v1",
        "canonical_tag": value["canonical_tag"],
        "ledger_request_id": ledger_entry["request_id"],
        "result_artifact_id": value["result_artifact_id"], "artifact": artifact,
        "event": event, "claim_revision": revision,
        "claim_ledger_policy": value["claim_ledger_policy"],
    }


def _event_with_keys(record: dict, event: dict) -> dict:
    value = copy.deepcopy(event)
    value["design_key"] = registry.design_key(record, value)
    value["execution_key"] = registry.execution_key(record, value)
    return value


def apply_plan(plan: Mapping[str, Any], *, regenerate_commands: Sequence[Sequence[str]] = ()) -> Path:
    """Apply an idempotent artifact -> event -> claim prefix; reruns finish a partial prefix."""
    tag = plan["canonical_tag"]
    registry.append_artifacts(tag, {plan["result_artifact_id"]: plan["artifact"]})
    path = registry.circuit_path(tag)
    record = json.loads(path.read_text())
    expected_event = _event_with_keys(record, plan["event"])
    existing_events = [item for item in record["evidence_events"] if item["event_id"] == expected_event["event_id"]]
    if existing_events and existing_events != [expected_event]:
        raise FastScreenPublishError("event id collision")
    if not existing_events:
        registry.append_evidence_event(tag, plan["event"])
    record = json.loads(path.read_text())
    revision = plan["claim_revision"]
    existing_claims = [item for item in record["claims"] if item["claim_id"] == revision["claim_id"]]
    if existing_claims and existing_claims != [revision]:
        raise FastScreenPublishError("claim revision id collision")
    if not existing_claims:
        registry.append_claim_revision(tag, revision)
    final = json.loads(path.read_text())
    registry.validate_v2(final)
    for command in regenerate_commands:
        subprocess.run(list(command), cwd=REPO, check=True, capture_output=True, text=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    plan = (build_cross_syntax_plan(spec) if spec.get("schema") == CROSS_SYNTAX_SCHEMA
            else build_plan(spec))
    if args.apply:
        apply_plan(plan, regenerate_commands=(
            (sys.executable, "basis_aligned/bilinear_quotient/make_circuit_coverage.py"),
            (sys.executable, "basis_aligned/bilinear_quotient/make_circuit_experiment_index.py"),
            (sys.executable, "basis_aligned/bilinear_quotient/make_circuit_campaign_queue.py"),
        ))
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
