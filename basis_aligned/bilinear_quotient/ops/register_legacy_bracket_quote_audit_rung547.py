#!/usr/bin/env python3
"""Attach audited pre-registry bracket/quote evidence without promoting it."""

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
OLD_CLAIM, NEW_CLAIM = "pending_opener_state.v18", "pending_opener_state.v19"
PATHS = {
    "r547_legacy_audit": ("basis_aligned/bilinear_quotient/legacy_bracket_quote_evidence_audit_rung547.json", "audit"),
    "r547_legacy_audit_implementation": ("basis_aligned/bilinear_quotient/ops/legacy_bracket_quote_evidence_audit_rung547.py", "audit_implementation"),
    "legacy_bracket_match_implementation": ("basis_aligned/bilinear_quotient/bracket_match.py", "legacy_implementation"),
    "legacy_bracket_match_result": ("basis_aligned/bilinear_quotient/bracket_match_results.json", "legacy_result"),
    "legacy_bracket_pointer_implementation": ("basis_aligned/bilinear_quotient/bracket_pointer_pairs.py", "legacy_implementation"),
    "legacy_bracket_pointer_result": ("basis_aligned/bilinear_quotient/bracket_pointer_pairs_results.json", "legacy_result"),
    "legacy_bracket_query_rank_implementation": ("basis_aligned/bilinear_quotient/bracket_query_rank.py", "legacy_implementation"),
    "legacy_bracket_query_rank_result": ("basis_aligned/bilinear_quotient/bracket_query_rank_results.json", "legacy_result"),
    "legacy_quote_head_implementation": ("basis_aligned/bilinear_quotient/quote_close_heads.py", "legacy_implementation"),
    "legacy_quote_head_result": ("basis_aligned/bilinear_quotient/quote_close_heads_results.json", "legacy_result"),
    "legacy_quote_state_causal_implementation": ("basis_aligned/bilinear_quotient/quote_state_causal.py", "legacy_implementation"),
    "legacy_quote_state_causal_result": ("basis_aligned/bilinear_quotient/quote_state_causal_results.json", "legacy_result"),
}


def frozen(path: str, kind: str) -> dict:
    return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}


def event(event_id: str, test_type: str, site_id: str, metrics: list[dict], inputs: list[str], notes: str) -> dict:
    return {
        "event_id": event_id, "claim_id": OLD_CLAIM, "test_type": test_type, "stage": "invalid",
        "verdict": "invalid", "failure_kind": "invalid_instrument", "family_ids": [], "site_id": site_id,
        "split_plan_id": None, "evaluation_role": "post_hoc_legacy_audit_descriptive_only", "metrics": metrics,
        "prereg_artifact_id": None, "result_artifact_id": "r547_legacy_audit",
        "input_artifact_ids": inputs + ["r547_legacy_audit_implementation"], "seed": None,
        "checkpoint_sha256": None, "supersedes_event_id": None, "replicates_event_id": None,
        "sections": ["circuits/campaign_2026_08_30/04_bracket_closure.md",
                     "circuits/campaign_2026_08_30/09_quote_parity.md"], "notes": notes,
    }


def main() -> None:
    audit = json.loads((REPO / PATHS["r547_legacy_audit"][0]).read_text())
    assert audit["model_forwards"] == 0 and audit["outcomes_opened"] == []
    assert audit["bracket_pointer_pairs"]["sparsity_prediction_held"] is False
    assert audit["bracket_query_rank"]["smallest_rank_reaching_80_percent"] == 64
    assert audit["quote_parity_direction"]["causal_prediction_held"] is False
    append_artifacts(TAG, {key: frozen(*value) for key, value in PATHS.items()})
    events = [
        event(
            "legacy_bracket_match.r547.invalid_unsealed_rows.v1", "full_swap_ceiling",
            "attention13.head8.output.final_position",
            [
                {"name": "matched_score_edge_deletion_CE_nat", "estimate": 0.689, "ci95": None,
                 "bar": "descriptive only; requires frozen independent rows and document bootstrap"},
                {"name": "nested_match_control_n", "estimate": 1, "ci95": None,
                 "bar": "sufficient independent nested examples"},
            ],
            ["legacy_bracket_match_implementation", "legacy_bracket_match_result"],
            "Strong descriptive L13H8/matched-edge effect, but mutable fineweb_rows, no row receipt, no checkpoint "
            "hash, no split authority, and only one nested example make it non-promotive. R545/R546 do not duplicate "
            "a valid confirmation.",
        ),
        event(
            "legacy_bracket_pointer.r547.invalid_dense_decomposition.v1", "compiled_equivalence",
            "attention13.head8.double_qk_score",
            [
                {"name": "exact_score_relative_error", "estimate": audit["bracket_pointer_pairs"]["exact_relative_score_error"],
                 "ci95": None, "bar": "<=1e-4 numerical replay"},
                {"name": "top10_match_mass_fraction", "estimate": 0.1425, "ci95": None, "bar": ">=0.50"},
                {"name": "match_vs_distractor_top10_pair_difference", "estimate": 1, "ci95": None, "bar": ">=4"},
            ],
            ["legacy_bracket_pointer_implementation", "legacy_bracket_pointer_result",
             "legacy_bracket_query_rank_implementation", "legacy_bracket_query_rank_result"],
            "The 625-term writer-pair expansion replays the score exactly, but it is dense: both sparse-pair "
            "predictions failed, and rank 64 rather than rank 8 was needed for 80% of the query effect. Exact anatomy "
            "is not a sparse or causal weight-level circuit.",
        ),
        event(
            "legacy_quote_l13h8_parity.r547.invalid_unsealed_rows.v1", "das_interchange",
            "attention13.head8.output.final_position",
            [
                {"name": "l13h8_quote_target_deletion_CE_nat", "estimate": 0.524, "ci95": None,
                 "bar": "descriptive only; requires independent counterfactual confirmation"},
                {"name": "decoded_parity_rank1_gap_fraction_removed", "estimate": 0.0049, "ci95": None,
                 "bar": ">=0.50 for a causal parity carrier"},
            ],
            ["legacy_quote_head_implementation", "legacy_quote_head_result",
             "legacy_quote_state_causal_implementation", "legacy_quote_state_causal_result"],
            "L13H8 is a strong descriptive quote-closer owner, but the decoded rank-one parity direction was "
            "noncausal and the shared rows were unsealed. Do not equate head ownership with a pending-opener state.",
        ),
    ]
    path = circuit_path(TAG)
    record = json.loads(path.read_text())
    for item in events:
        if not any(old["event_id"] == item["event_id"] for old in record["evidence_events"]):
            append_evidence_event(TAG, item)
            record = json.loads(path.read_text())
    if not any(claim["claim_id"] == NEW_CLAIM for claim in record["claims"]):
        previous = next(claim for claim in record["claims"] if claim["claim_id"] == OLD_CLAIM)
        claim = copy.deepcopy(previous)
        claim.update({
            "claim_id": NEW_CLAIM, "revision": 19, "status": "specified", "supersedes": OLD_CLAIM,
            "evidence_event_ids": list(previous["evidence_event_ids"]) + [item["event_id"] for item in events],
            "next_missing": previous["next_missing"],
        })
        append_claim_revision(TAG, claim)
    final = json.loads(path.read_text())
    validate_v2(final)
    rebuild_registry_v2()
    print(json.dumps({
        "tag": TAG, "claim_id": NEW_CLAIM, "legacy_events": len(events),
        "campaign_tier4_preserved_as_claim": False, "model_outcomes_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
