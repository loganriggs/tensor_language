#!/usr/bin/env python3
"""Register the verified R538 site result and promote the site-live claim."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_artifacts, append_claim_revision, append_evidence_event,
    circuit_path, file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM = "pending_opener_state.v6"
NEW_CLAIM = "pending_opener_state.v7"
INVALID_EVENT = "pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1"
COMPLETE_EVENT = "pending_opener_common_site_ceiling.r538.complete.v2"
RESULT = "basis_aligned/bilinear_quotient/pending_opener_common_site_rung538_results.json"
AUDIT = "basis_aligned/bilinear_quotient/pending_opener_common_site_rung538_terminal_audit.json"
AUDIT_IMPL = "basis_aligned/bilinear_quotient/ops/pending_opener_common_site_rung538_terminal_audit.py"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def main() -> None:
    result = json.loads((REPO / RESULT).read_text())
    audit = json.loads((REPO / AUDIT).read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["pred_b_common_live_site"] is True
    assert result["pred_c_frozen_causal_order_selection"] is True
    assert result["selected_site"] == "resid8"
    assert audit["all_checks_pass"] is True
    artifacts = {
        "r538_site_result_v2": frozen(RESULT, "result"),
        "r538_site_terminal_audit": frozen(AUDIT, "audit"),
        "r538_site_terminal_audit_implementation": frozen(AUDIT_IMPL, "audit_implementation"),
    }
    append_artifacts(TAG, artifacts)
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    if not any(item["event_id"] == COMPLETE_EVENT for item in record["evidence_events"]):
        selected = result["reports"]["resid8"]
        cells = [
            selected[split][family][direction]
            for split in ("FIT", "SELECT")
            for family in ("opener_type_substitution", "closed_then_reopened_type")
            for direction in ("base_to_donor", "donor_to_base")
        ]
        append_evidence_event(TAG, {
            "event_id": COMPLETE_EVENT,
            "claim_id": "pending_opener_state.v3",
            "test_type": "full_swap_ceiling",
            "stage": "complete",
            "verdict": "held",
            "failure_kind": None,
            "family_ids": ["opener_type_substitution", "closed_then_reopened_type"],
            "site_id": None,
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "evaluation_role": "FIT_SELECT_only",
            "metrics": [
                {
                    "name": "signed_donorward_movement",
                    "estimate": min(item["mean_donorward_movement"] for item in cells),
                    "ci95": [min(item["bootstrap95_lower_mean"] for item in cells), None],
                    "bar": "positive both directions with group-bootstrap lower bound>0",
                },
                {
                    "name": "individual_direction_success",
                    "estimate": min(item["positive_movement_fraction"] for item in cells),
                    "ci95": None,
                    "bar": ">=0.70 at one common site",
                },
            ],
            "prereg_artifact_id": "r538_site_preregistration",
            "result_artifact_id": "r538_site_result_v2",
            "input_artifact_ids": [
                "r537_rows", "r537_rows_receipt", "r537_capability_result",
                "r538_site_preregistration", "r538_site_implementation_v2",
            ],
            "seed": 538,
            "checkpoint_sha256": result["checkpoint"]["weights_sha256"],
            "supersedes_event_id": INVALID_EVENT,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_COMMON_SITE_RUNG538_PREREGISTRATION.md"],
            "notes": (
                "The terminal audit independently recomputes every cell from saved row-level movements. "
                "The corrected loader reproduces the invalid predecessor's scientific matrix exactly."
            ),
        })

    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 7, "status": "site_live",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": [
                "pending_opener_capability.r537.complete.v1", INVALID_EVENT, COMPLETE_EVENT,
            ],
            "next_missing": (
                "measure full resid8 swaps for both invariance/control families, then freeze shared-versus-"
                "family-specific projector fitting; FINAL_TEST and OOD remain unopened"
            ),
        })
        claim["candidate_sites"] = [{
            "site_id": "residual.block8.entry.final_position",
            "tensor_path": "residual stream entering block 8 at the final prompt position",
            "shape": ["batch", 1152],
            "intervention": "complete donor-state interchange; projector fitting not yet run",
            "ceiling_event_ids": [COMPLETE_EVENT],
        }] + [site for site in previous["candidate_sites"]
             if site["site_id"] != "residual.layer8_to14.final_position"]
        append_claim_revision(TAG, claim)

    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "site_live",
        "event_id": COMPLETE_EVENT, "selected_site": "resid8",
        "result_sha256": artifacts["r538_site_result_v2"]["sha256"],
        "audit_sha256": artifacts["r538_site_terminal_audit"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
