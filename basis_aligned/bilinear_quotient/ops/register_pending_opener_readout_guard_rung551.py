#!/usr/bin/env python3
"""Register the conditional R551 readout-independence guard before R549 runs."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    append_artifacts, append_claim_revision, append_evidence_event, circuit_path,
    file_sha256, rebuild_registry_v2, validate_v2,
)

TAG = "task.bracket.pending_opener"
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v21", "pending_opener_state.v22"
EVENT = "pending_opener_downstream_readout_guard.r551.preregistered.v1"
PATHS = {
    "r551_downstream_readout_guard_preregistration": (
        "basis_aligned/polynomial_causal/PENDING_OPENER_DOWNSTREAM_READOUT_INDEPENDENCE_RUNG551_PREREGISTRATION.md",
        "preregistration"),
    "r551_downstream_readout_guard_implementation": (
        "basis_aligned/bilinear_quotient/ops/pending_opener_downstream_readout_independence_rung551.py",
        "implementation"),
    "r551_downstream_readout_guard_test": (
        "basis_aligned/bilinear_quotient/ops/test_pending_opener_downstream_readout_independence_rung551.py", "test"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def main() -> None:
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    claim = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
    target_families = [
        family["family_id"] for family in claim["counterfactual_families"]
        if family["family_id"] in {
            "direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
        }
    ]
    if not any(old["event_id"] == EVENT for old in record["evidence_events"]):
        append_evidence_event(TAG, {
            "event_id": EVENT,
            "claim_id": OLD_CLAIM,
            "test_type": "null_control",
            "stage": "preregistered",
            "verdict": "inconclusive",
            "failure_kind": None,
            "family_ids": target_families,
            "site_id": "r549_selected_later_write_vs_direct_closer_readout_span",
            "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
            "evaluation_role": "FIT_only_pre_outcome_interpretation_guard",
            "metrics": [
                {"name": "median_transition_template_norm_fraction_in_closer_readout_span",
                 "estimate": None, "ci95": None, "bar": "<=0.50 after R549 SELECT validation"},
                {"name": "r549_pairwise_readout_cosine_recomputation", "estimate": None, "ci95": None,
                 "bar": "exactly reproduce the saved pairwise diagnostic within 2e-7"},
            ],
            "prereg_artifact_id": "r551_downstream_readout_guard_preregistration",
            "result_artifact_id": None,
            "input_artifact_ids": [
                "r549_downstream_response_atlas_preregistration",
                "r549_downstream_response_atlas_implementation",
                "r551_downstream_readout_guard_implementation",
                "r551_downstream_readout_guard_test",
            ],
            "seed": None,
            "checkpoint_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            "supersedes_event_id": None,
            "replicates_event_id": None,
            "sections": ["PENDING_OPENER_DOWNSTREAM_READOUT_INDEPENDENCE_RUNG551_PREREGISTRATION.md"],
            "notes": (
                "Frozen before R549 outcome. This guard cannot change the R549 winner; it only decides whether a "
                "validated response is nonredundant with the direct closer-token output span."
            ),
        })

    record = json.loads(path.read_text())
    if not any(item["claim_id"] == NEW_CLAIM for item in record["claims"]):
        previous = next(item for item in record["claims"] if item["claim_id"] == OLD_CLAIM)
        updated = copy.deepcopy(previous)
        updated.update({
            "claim_id": NEW_CLAIM,
            "revision": 22,
            "status": "specified",
            "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [EVENT],
            "next_missing": (
                "execute R549, independently audit it with R550, and apply the frozen R551 readout-span guard before "
                "using any later response as a multi-output DAS target; FINAL_TEST/OOD remain unopened"
            ),
        })
        append_claim_revision(TAG, updated)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "status": "specified", "event": EVENT,
        "model_outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
