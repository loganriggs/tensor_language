#!/usr/bin/env python3
"""Create the four initial version-2 behavior-circuit records, CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
POLY = REPO / "basis_aligned" / "polynomial_causal"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    design_key,
    execution_key,
    file_sha256,
    write_behavior_circuit,
)


def load_contract_module():
    path = POLY / "circuit_counterfactual_contract_rung536.py"
    spec = importlib.util.spec_from_file_location("r536_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACT_MODULE = load_contract_module()
CONTRACT_JSON = BQ / "circuit_counterfactual_contract_rung536.json"
READINESS_JSON = BQ / "circuit_counterfactual_readiness_rung536.json"

SPECS = {
    "induction_selector_and_payload": {
        "tag": "task.induction.selector_payload",
        "variable": {
            "id": "induction_selector_payload",
            "domain": "valid earlier source positions and the token/value following each source",
            "read": "token equality and source position",
            "operation": "select a matching earlier source, then transport its following payload",
            "write": "selector-dependent payload contribution to the target-token logits",
            "endpoint": "signed donor-answer minus base-answer logit margin at the query continuation",
        },
        "alternatives": ["generic token identity", "generic copy service", "lag/corpus shortcut"],
        "next_missing": "freeze two-valid-source and payload-swap rows; measure selector and value site ceilings",
        "artifacts": {
            "campaign_report": ("basis_aligned/bilinear_quotient/circuits/campaign_2026_08_30/02_induction_copy.md", "report"),
            "rows_receipt": ("basis_aligned/bilinear_quotient/terminal_copy_induction_v2_rows_receipt.json", "split"),
            "ood_result": ("basis_aligned/polynomial_causal/induction_equality_tensor_final_ood_v2_retry1_result.json", "result"),
        },
        "legacy_events": [
            {
                "event_id": "induction_terminal_collateral_failure.legacy.v1",
                "test_type": "removal",
                "stage": "complete",
                "verdict": "failed",
                "failure_kind": "scientific_null",
                "family_ids": ["natural_pair_interchange"],
                "site_id": "terminal_copy_service",
                "evaluation_role": "legacy_final",
                "metrics": [{"name": "unrelated_behavior_preservation", "estimate": None, "ci95": None, "bar": "terminal collateral certificate"}],
                "result_artifact_id": "campaign_report",
                "prereg_artifact_id": None,
                "split_plan_id": None,
                "seed": None,
                "checkpoint_sha256": None,
                "supersedes_event_id": None,
                "replicates_event_id": None,
                "sections": ["campaign_2026_08_30/02_induction_copy.md"],
            }
        ],
    },
    "pending_opener_state": {
        "tag": "task.bracket.pending_opener",
        "variable": {
            "id": "pending_opener_state",
            "domain": "no pending opener or one pending parenthesis/quote-like opener type",
            "read": "recent opener/closer evidence in context",
            "operation": "maintain a recency-weighted pending-opener state",
            "write": "closer-type evidence at the final prediction position",
            "endpoint": "signed required-closer minus alternative-closer logit margin",
        },
        "alternatives": ["punctuation identity", "position shift", "recency steering"],
        "next_missing": "freeze opener-type and closer-reset families; require leave-one-family-out transfer",
        "artifacts": {
            "task_report": ("basis_aligned/qk_mdl/algo_tasks/bracket/report.md", "report"),
            "das_result": ("basis_aligned/qk_mdl/algo_tasks/bracket/das.json", "result"),
            "semantics_report": ("basis_aligned/qk_mdl/algo_tasks/semantics_opener/report.md", "report"),
        },
        "legacy_events": [
            {
                "event_id": "pending_opener_rank4_das.legacy.v1",
                "test_type": "das_interchange",
                "stage": "complete",
                "verdict": "held",
                "failure_kind": None,
                "family_ids": ["opener_presence_edit"],
                "site_id": "layer13.entry",
                "evaluation_role": "legacy_heldout",
                "metrics": [{"name": "heldout_interchange_recovery", "estimate": 0.872, "ci95": None, "bar": "legacy instrument; not CF3 promotion"}],
                "result_artifact_id": "das_result",
                "prereg_artifact_id": None,
                "split_plan_id": None,
                "seed": None,
                "checkpoint_sha256": None,
                "supersedes_event_id": None,
                "replicates_event_id": None,
                "sections": ["qk_mdl/algo_tasks/bracket/report.md"],
            }
        ],
    },
    "successor_pointer_state": {
        "tag": "task.successor.pointer",
        "variable": {
            "id": "successor_pointer_state",
            "domain": "elements of weekday, month, alphabet, digit, and list vocabularies",
            "read": "the final sequence element and coherence/family context",
            "operation": "use an identity pointer to retrieve the next element",
            "write": "evidence for the successor token",
            "endpoint": "signed donor-successor minus base-successor logit margin",
        },
        "alternatives": ["token identity", "sequence-family identity", "prefix coherence"],
        "next_missing": "expand families and test shared-plus-private projectors against failed cross-family transfer",
        "artifacts": {
            "stimuli": ("basis_aligned/qk_mdl/algo_tasks/successor/stimuli.json", "dataset"),
            "task_report": ("basis_aligned/qk_mdl/algo_tasks/successor/report.md", "report"),
            "semantics_report": ("basis_aligned/qk_mdl/algo_tasks/semantics_successor/report.md", "report"),
        },
        "legacy_events": [
            {
                "event_id": "successor_cross_family_transfer.legacy.v1",
                "test_type": "cross_family_transfer",
                "stage": "complete",
                "verdict": "failed",
                "failure_kind": "scientific_null",
                "family_ids": ["same_family_last_element_swap", "coherent_whole_sequence_shift"],
                "site_id": "post_attention8",
                "evaluation_role": "legacy_heldout",
                "metrics": [{"name": "cross_family_transfer", "estimate": 0.0, "ci95": None, "bar": "positive transfer without refitting"}],
                "result_artifact_id": "task_report",
                "prereg_artifact_id": None,
                "split_plan_id": None,
                "seed": None,
                "checkpoint_sha256": None,
                "supersedes_event_id": None,
                "replicates_event_id": None,
                "sections": ["qk_mdl/algo_tasks/successor/report.md"],
            },
            {
                "event_id": "successor_layer8_input_ceiling.legacy.v1",
                "test_type": "full_swap_ceiling",
                "stage": "complete",
                "verdict": "null",
                "failure_kind": "scientific_null",
                "family_ids": ["same_family_last_element_swap"],
                "site_id": "layer8.input",
                "evaluation_role": "legacy_heldout",
                "metrics": [{"name": "full_swap_payload_ceiling", "estimate": 0.0, "ci95": None, "bar": ">0"}],
                "result_artifact_id": "task_report",
                "prereg_artifact_id": None,
                "split_plan_id": None,
                "seed": None,
                "checkpoint_sha256": None,
                "supersedes_event_id": None,
                "replicates_event_id": None,
                "sections": ["qk_mdl/algo_tasks/successor/report.md"],
            },
        ],
    },
    "increment_state": {
        "tag": "task.increment.state",
        "variable": {
            "id": "increment_state",
            "domain": "numeric values represented as digits, words, or list elements",
            "read": "recent numeric state and list relation",
            "operation": "apply an increment relation to the numeric state",
            "write": "evidence for the next numeric token",
            "endpoint": "signed shifted-next-number minus base-next-number logit margin",
        },
        "alternatives": ["generic digit identity", "numeric value without increment", "list-format continuation"],
        "next_missing": "freeze cross-format rows; require number-word transfer and nonincrement numeric controls",
        "artifacts": {
            "task_report": ("basis_aligned/qk_mdl/algo_tasks/increment/report.md", "report"),
            "das_result": ("basis_aligned/qk_mdl/algo_tasks/increment/s3_das.json", "result"),
            "postattn_result": ("basis_aligned/qk_mdl/algo_tasks/increment/s3b_das_postattn.json", "result"),
        },
        "legacy_events": [
            {
                "event_id": "increment_postattn_rank4_das.legacy.v1",
                "test_type": "das_interchange",
                "stage": "complete",
                "verdict": "held",
                "failure_kind": None,
                "family_ids": ["coherent_constant_shift"],
                "site_id": "post_attention8",
                "evaluation_role": "legacy_heldout",
                "metrics": [
                    {"name": "heldout_flip_rate", "estimate": 0.8, "ci95": None, "bar": "legacy instrument; not CF3 promotion"},
                    {"name": "heldout_recovery", "estimate": 0.939, "ci95": None, "bar": "legacy instrument; not CF3 promotion"}
                ],
                "result_artifact_id": "postattn_result",
                "prereg_artifact_id": None,
                "split_plan_id": None,
                "seed": None,
                "checkpoint_sha256": None,
                "supersedes_event_id": None,
                "replicates_event_id": None,
                "sections": ["qk_mdl/algo_tasks/increment/report.md"],
            }
        ],
    },
}


def artifact(path_string: str, kind: str) -> dict:
    path = REPO / path_string
    if not path.is_file():
        return {"path": path_string, "sha256": None, "kind": kind, "status": "missing"}
    return {"path": path_string, "sha256": file_sha256(path), "kind": kind, "status": "frozen"}


def candidate_sites(pilot: dict) -> list[dict]:
    return [
        {
            "site_id": site.replace(" ", "_").replace("/", "_").lower(),
            "tensor_path": site,
            "shape": ["batch", "position", "site-specific feature dimension"],
            "intervention": "replace the complete registered site state before fitting a subspace",
            "ceiling_event_ids": [],
        }
        for site in pilot["candidate_sites"]
    ]


def build_record(pilot: dict) -> dict:
    spec = SPECS[pilot["circuit_id"]]
    artifacts = {
        "contract_snapshot": artifact(
            str(CONTRACT_JSON.relative_to(REPO)), "preregistration"
        ),
        "readiness_audit": artifact(
            str(READINESS_JSON.relative_to(REPO)), "audit"
        ),
    }
    artifacts.update({key: artifact(path, kind) for key, (path, kind) in spec["artifacts"].items()})
    families = []
    for item in pilot["counterfactual_families"]:
        families.append({
            "family_id": item["family_id"],
            "role": item["intervention_role"],
            "changes": [item["changed_variable"], item["intervention"]],
            "holds_fixed": item["held_fixed"],
            "builder_artifact_id": None,
            "control_ids": item["controls"],
            "split_plan_id": "joint_split_v1",
            "status": "proposed",
        })
    claim_id = f"{pilot['circuit_id']}.v1"
    claim = {
        "claim_id": claim_id,
        "revision": 1,
        "status": "proposed",
        "supersedes": None,
        "causal_variable": spec["variable"],
        "alternative_explanations": spec["alternatives"],
        "counterfactual_families": families,
        "candidate_sites": candidate_sites(pilot),
        "split_plan_ids": ["joint_split_v1"],
        "evidence_event_ids": [event["event_id"] for event in spec["legacy_events"]],
        "translation_ids": [],
        "next_missing": spec["next_missing"],
    }
    record = {
        "schema_version": 2,
        "tag": spec["tag"],
        "identity": {
            "kind": "behavior_circuit",
            "instance": None,
            "identity_artifact_id": "contract_snapshot",
            "aliases": [pilot["circuit_id"]],
        },
        "claims": [claim],
        "split_plans": [
            {
                "split_plan_id": "joint_split_v1",
                "unit": "document/template/entity as appropriate; shared across all families",
                "partition_artifact_id": None,
                "builder_artifact_id": None,
                "seed": None,
                "groups": {"FIT": 0, "SELECT": 0, "FINAL_TEST": 0, "OOD": 0},
                "leakage_group_keys": pilot["split_axes"],
                "sealed_before_outcomes": False,
                "sealed_at": None,
            }
        ],
        "evidence_events": [],
        "translations": [],
        "artifacts": artifacts,
        "provenance": {
            "rung": 536,
            "contract_sha256": artifacts["contract_snapshot"]["sha256"],
            "readiness_sha256": artifacts["readiness_audit"]["sha256"],
        },
    }
    for raw in spec["legacy_events"]:
        event = dict(raw, claim_id=claim_id)
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        record["evidence_events"].append(event)
    return record


def main() -> None:
    written = []
    for pilot in CONTRACT_MODULE.PILOTS:
        record = build_record(pilot)
        path = write_behavior_circuit(record)
        written.append(str(path.relative_to(REPO)))
    print(json.dumps({"written": written, "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
