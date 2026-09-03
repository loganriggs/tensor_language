#!/usr/bin/env python3
"""Attach the frozen rung-537 row authority to the canonical behavior record."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_claim_revision,
    circuit_path,
    file_sha256,
    rebuild_registry_v2,
    validate_v2,
)

TAG = "task.bracket.pending_opener"


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


ARTIFACTS = {
    "r537_rows": frozen("basis_aligned/bilinear_quotient/pending_opener_multifamily_rows_rung537.json", "dataset"),
    "r537_rows_receipt": frozen("basis_aligned/bilinear_quotient/pending_opener_multifamily_rows_rung537_receipt.json", "split"),
    "r537_rows_builder": frozen("basis_aligned/bilinear_quotient/ops/pending_opener_multifamily_rows_rung537.py", "builder"),
    "r537_preregistration": frozen("basis_aligned/polynomial_causal/PENDING_OPENER_MULTI_COUNTERFACTUAL_RUNG537_PREREGISTRATION.md", "preregistration"),
}

SPLIT = {
    "split_plan_id": "pending_opener_joint_split_r537_v1",
    "unit": "group_id shared across every counterfactual family",
    "partition_artifact_id": "r537_rows_receipt",
    "builder_artifact_id": "r537_rows_builder",
    "seed": None,
    "groups": {"FIT": 48, "SELECT": 16, "FINAL_TEST": 16, "OOD": 16},
    "leakage_group_keys": ["group_id", "lexical pool", "prefix family"],
    "sealed_before_outcomes": True,
    "sealed_at": "2026-09-03T14:25Z",
}

CLAIM = {
    "claim_id": "pending_opener_state.v2",
    "revision": 2,
    "status": "proposed",
    "supersedes": "pending_opener_state.v1",
    "causal_variable": {
        "id": "pending_opener_state",
        "domain": "pending parenthesis or quote state at the final prediction position",
        "read": "opener, closer, ordering, and recency evidence in the preceding context",
        "operation": "maintain which opener type remains pending after completed earlier spans",
        "write": "signed evidence for the matching closer token",
        "endpoint": "symmetric donor-closer versus base-closer final-logit margin",
    },
    "alternative_explanations": [
        "single punctuation-token identity",
        "position shift from opener deletion",
        "template or lexical shortcut",
        "off-manifold steering that exceeds natural full-state interchange",
    ],
    "counterfactual_families": [
        {
            "family_id": "opener_type_substitution",
            "role": "interchange",
            "changes": ["one opener token: parenthesis versus quote", "correct next closer"],
            "holds_fixed": ["token length", "all lexical tokens", "opener position", "body text"],
            "builder_artifact_id": "r537_rows_builder",
            "control_ids": ["non-opener punctuation substitution", "wrong closer", "random subspace"],
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "status": "frozen",
        },
        {
            "family_id": "closed_then_reopened_type",
            "role": "interchange",
            "changes": ["order of completed and newly opened punctuation structures", "correct next closer"],
            "holds_fixed": ["token length", "lexical token multiset", "prefix", "final lexical suffix"],
            "builder_artifact_id": "r537_rows_builder",
            "control_ids": ["same-order paraphrase", "wrong closer", "random subspace"],
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "status": "frozen",
        },
        {
            "family_id": "pending_state_preserved_surface_edit",
            "role": "invariance",
            "changes": ["wording", "lexical identities", "opener-to-final distance"],
            "holds_fixed": ["pending parenthesis state", "correct closer", "task role"],
            "builder_artifact_id": "r537_rows_builder",
            "control_ids": ["held-out prefixes", "held-out lexical pool", "distance strata"],
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "status": "frozen",
        },
        {
            "family_id": "later_matching_closer_reset",
            "role": "necessity",
            "changes": ["whether the earlier opener remains unclosed"],
            "holds_fixed": ["original opener", "scored closer token", "matched suffix"],
            "builder_artifact_id": None,
            "control_ids": ["nonmatching closer", "punctuation insertion", "distance-matched filler"],
            "split_plan_id": "pending_opener_joint_split_r537_v1",
            "status": "proposed",
        },
    ],
    "candidate_sites": [
        {
            "site_id": "residual.layer8_to14.final_position",
            "tensor_path": "final-position residual entering each layer 8 through 14",
            "shape": ["batch", "1152"],
            "intervention": "complete donor-state interchange before any projector fit",
            "ceiling_event_ids": [],
        },
        {
            "site_id": "attention13.head8.output.final_position",
            "tensor_path": "layer 13 head 8 pre-projection/output contribution at final position",
            "shape": ["batch", "128 or 1152 after output projection"],
            "intervention": "complete donor head state/output interchange",
            "ceiling_event_ids": [],
        },
        {
            "site_id": "mlp8_to14.product.final_position",
            "tensor_path": "bilinear product activations for MLP layers 8 through 14 at final position",
            "shape": ["batch", "4608"],
            "intervention": "complete donor product-state interchange before subspace optimization",
            "ceiling_event_ids": [],
        },
    ],
    "split_plan_ids": ["pending_opener_joint_split_r537_v1"],
    "evidence_event_ids": [],
    "translation_ids": [],
    "next_missing": "materialize non-opener/wrong-closer controls, then run FIT/SELECT capability gates with FINAL_TEST and OOD sealed",
}


def main() -> None:
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    existing = next((claim for claim in record["claims"] if claim["claim_id"] == CLAIM["claim_id"]), None)
    if existing is None:
        append_claim_revision(TAG, CLAIM, artifacts=ARTIFACTS, split_plans=[SPLIT])
    else:
        assert existing == CLAIM
        assert all(record["artifacts"].get(key) == value for key, value in ARTIFACTS.items())
        assert SPLIT in record["split_plans"]
        validate_v2(record)
        rebuild_registry_v2()
    print(json.dumps({"tag": TAG, "claim_id": CLAIM["claim_id"], "status": CLAIM["status"], "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
