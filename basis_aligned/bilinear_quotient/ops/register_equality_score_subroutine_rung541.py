#!/usr/bin/env python3
"""Register the cross-head equality-score subroutine and its existing evidence.

This is a CPU-only organization rung.  It makes no new scientific claim and opens
no model outcomes; it binds already-existing preregistrations/results so future
work can query one canonical record before repeating an experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import (  # noqa: E402
    design_key,
    execution_key,
    file_sha256,
    write_behavior_circuit,
)


TAG = "subroutine.induction.equality_score"
CLAIM_ID = "cross_head_equality_score.v1"


ARTIFACT_PATHS = {
    "r536_independent_audit": (
        "basis_aligned/polynomial_causal/R536_MULTI_COUNTERFACTUAL_PILOT_AUDIT.md", "audit"),
    "r459_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_TERM_SCORE_PAYLOAD_RUNG459_PREREGISTRATION.md", "preregistration"),
    "r459_result": (
        "basis_aligned/bilinear_quotient/equality_term_score_payload_rung459_results.json", "result"),
    "r459_implementation": (
        "basis_aligned/bilinear_quotient/ops/equality_term_score_payload_rung459.py", "implementation"),
    "r460_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_SCORE_CODE_OOD_RUNG460_PREREGISTRATION.md", "preregistration"),
    "r460_result": (
        "basis_aligned/bilinear_quotient/equality_score_code_ood_rung460_results.json", "result"),
    "r462_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_SCORE_DOWNSTREAM_GATE_RUNG462_PREREGISTRATION.md", "preregistration"),
    "r462_result": (
        "basis_aligned/bilinear_quotient/equality_score_downstream_gate_rung462_results.json", "result"),
    "r464_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_SCORE_CORRECTION_INTERCHANGE_RUNG464_PREREGISTRATION.md", "preregistration"),
    "r464_result": (
        "basis_aligned/bilinear_quotient/equality_score_correction_interchange_rung464_results.json", "result"),
    "r498_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_MATCHER_CAUSAL_ACTION_QUOTIENT_RUNG498_PREREGISTRATION.md", "preregistration"),
    "r498_result": (
        "basis_aligned/bilinear_quotient/equality_matcher_causal_action_quotient_rung498_results.json", "result"),
    "r498_implementation": (
        "basis_aligned/bilinear_quotient/ops/equality_matcher_causal_action_quotient_rung498.py", "implementation"),
    "r500_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_MATCHER_MLP9_READER_CALIBRATION_RUNG500_PREREGISTRATION.md", "preregistration"),
    "r500_result": (
        "basis_aligned/bilinear_quotient/equality_matcher_mlp9_reader_calibration_rung500_results.json", "result"),
    "r531_preregistration": (
        "basis_aligned/polynomial_causal/EQUALITY_SCORE_FACTOR_BRANCH_SHARING_RUNG531_PREREGISTRATION.md", "preregistration"),
    "r531_result": (
        "basis_aligned/bilinear_quotient/equality_score_factor_branch_sharing_rung531_results.json", "result"),
    "r531_receipt": (
        "basis_aligned/polynomial_causal/EQUALITY_SCORE_FACTOR_BRANCH_SHARING_RUNG531_TERMINAL_RECEIPT.md", "audit"),
    "terminal_copy_rows_receipt": (
        "basis_aligned/bilinear_quotient/terminal_copy_induction_v2_rows_receipt.json", "split"),
    "terminal_copy_negative_receipt": (
        "basis_aligned/polynomial_causal/terminal_copy_selection_v1_attempt2_negative_receipt.json", "result"),
    "synthetic_copy_builder": (
        "basis_aligned/polynomial_causal/terminal_copy_induction_v1.py", "builder"),
}


def frozen(path: str, kind: str) -> dict:
    source = REPO / path
    assert source.is_file(), path
    return {"path": path, "sha256": file_sha256(source), "kind": kind, "status": "frozen"}


def metric(name: str, estimate, bar: str, ci95=None) -> dict:
    return {"name": name, "estimate": estimate, "ci95": ci95, "bar": bar}


def event(
    event_id: str,
    test_type: str,
    verdict: str,
    failure_kind,
    family_ids: list[str],
    site_id: str | None,
    metrics: list[dict],
    prereg: str | None,
    result: str,
    inputs: list[str],
    checkpoint_sha256: str | None,
    sections: list[str],
    notes: str,
) -> dict:
    return {
        "event_id": event_id,
        "claim_id": CLAIM_ID,
        "test_type": test_type,
        "stage": "complete",
        "verdict": verdict,
        "failure_kind": failure_kind,
        "family_ids": family_ids,
        "site_id": site_id,
        "split_plan_id": "legacy_natural_code_roles_v1",
        "evaluation_role": "already_open_preregistered_lineage",
        "metrics": metrics,
        "prereg_artifact_id": prereg,
        "result_artifact_id": result,
        "input_artifact_ids": inputs,
        "seed": None,
        "checkpoint_sha256": checkpoint_sha256,
        "supersedes_event_id": None,
        "replicates_event_id": None,
        "sections": sections,
        "notes": notes,
    }


def main() -> None:
    r459 = json.loads((BQ / "equality_term_score_payload_rung459_results.json").read_text())
    r460 = json.loads((BQ / "equality_score_code_ood_rung460_results.json").read_text())
    r462 = json.loads((BQ / "equality_score_downstream_gate_rung462_results.json").read_text())
    r464 = json.loads((BQ / "equality_score_correction_interchange_rung464_results.json").read_text())
    r498 = json.loads((BQ / "equality_matcher_causal_action_quotient_rung498_results.json").read_text())
    r500 = json.loads((BQ / "equality_matcher_mlp9_reader_calibration_rung500_results.json").read_text())
    r531 = json.loads((BQ / "equality_score_factor_branch_sharing_rung531_results.json").read_text())
    terminal = json.loads((REPO / ARTIFACT_PATHS["terminal_copy_negative_receipt"][0]).read_text())

    artifacts = {key: frozen(*value) for key, value in ARTIFACT_PATHS.items()}
    families = [
        {
            "family_id": "cross_head_score_swap",
            "role": "interchange",
            "changes": [
                "the complete double-query/key equality score supplied to L8H4",
                "replace L8H4's native score with the scaled L5H5 score while retaining L8H4's value/output payload",
            ],
            "holds_fixed": ["input document", "query position", "L8H4 payload", "all non-equality attention terms"],
            "builder_artifact_id": "r459_implementation",
            "control_ids": ["L7H3 score donor", "L5H5 payload donor", "sign reversal", "matched random subspace"],
            "split_plan_id": "legacy_natural_code_roles_v1",
            "status": "validated",
        },
        {
            "family_id": "text_match_pattern_edit_payload_fixed",
            "role": "interchange",
            "changes": [
                "which earlier position matches the current query token",
                "edit the input tokens to change the valid equality match while retaining the payload token",
            ],
            "holds_fixed": ["sequence length", "current query", "payload identity", "token multiset where possible"],
            "builder_artifact_id": "synthetic_copy_builder",
            "control_ids": ["irrelevant-source edit", "distance-matched decoy", "same-token-frequency decoy"],
            "split_plan_id": "equality_multifamily_future_v1",
            "status": "proposed",
        },
        {
            "family_id": "matched_natural_whole_state_swap",
            "role": "interchange",
            "changes": [
                "the naturally occurring equality-match state",
                "swap between documents matched on position, lag, frequency, and copy-task coordinate",
            ],
            "holds_fixed": ["corpus role", "position stratum", "distance stratum", "frequency stratum"],
            "builder_artifact_id": "r498_implementation",
            "control_ids": ["within-stratum wrong donor", "matched negative donor", "random subspace"],
            "split_plan_id": "equality_multifamily_future_v1",
            "status": "proposed",
        },
        {
            "family_id": "payload_swap_match_preserved",
            "role": "invariance",
            "changes": [
                "the value/output payload paired with an equality match",
                "swap the copied content while keeping the score-pattern relation fixed",
            ],
            "holds_fixed": ["source/query equality", "match positions", "score relation", "distance stratum"],
            "builder_artifact_id": "r498_implementation",
            "control_ids": ["complete payload full-swap ceiling", "L5H5 payload donor", "random subspace"],
            "split_plan_id": "equality_multifamily_future_v1",
            "status": "proposed",
        },
        {
            "family_id": "match_break_answer_fixed",
            "role": "necessity",
            "changes": [
                "availability of the earlier equality match",
                "replace the earlier matching token by a decoy while retaining the original answer token",
            ],
            "holds_fixed": ["current query", "payload/answer token", "length", "local syntax"],
            "builder_artifact_id": "synthetic_copy_builder",
            "control_ids": ["irrelevant-source edit", "offset control", "token derangement"],
            "split_plan_id": "equality_multifamily_future_v1",
            "status": "frozen",
        },
    ]
    claim = {
        "claim_id": CLAIM_ID,
        "revision": 1,
        "status": "site_live",
        "supersedes": None,
        "causal_variable": {
            "id": "cross_head_equality_score",
            "domain": "earlier key positions relative to a current query position",
            "read": "whether the current token matches a token at each earlier position",
            "operation": "construct an attention score pattern over matching earlier positions independently of the copied payload",
            "write": "a score pattern that can be combined with L8H4's value/output payload and read by MLP9",
            "endpoint": "signed causal recovery of L8H4's copy-related effect and donor-answer logit movement",
        },
        "alternative_explanations": [
            "whole-head identity rather than a cross-head subroutine",
            "corpus- or lag-specific score shortcut",
            "generic repeated-token detector without copy-task selectivity",
            "score/payload mixture caused by the chosen attention factorization",
        ],
        "counterfactual_families": families,
        "candidate_sites": [
            {
                "site_id": "attention8.head4.double_qk_score",
                "tensor_path": "L8H4 complete double-query/key score matrix before value weighting",
                "shape": ["batch", "query_position", "key_position"],
                "intervention": "replace the complete score pattern using the same semantics as a later learned subspace",
                "ceiling_event_ids": ["equality_score_swap.r459.natural.v1", "equality_score_swap.r460.code.v1"],
            },
            {
                "site_id": "residual.block8.entry.query_position",
                "tensor_path": "residual stream entering block 8 at the query position",
                "shape": ["batch", 1152],
                "intervention": "complete donor-state interchange before fitting a query-side subspace",
                "ceiling_event_ids": [],
            },
            {
                "site_id": "residual.block8.entry.key_positions",
                "tensor_path": "residual stream entering block 8 at matched and control key positions",
                "shape": ["batch", "key_position", 1152],
                "intervention": "complete donor-state interchange before fitting a key-side subspace",
                "ceiling_event_ids": [],
            },
            {
                "site_id": "mlp7.product.query_and_key_positions",
                "tensor_path": "MLP7 bilinear product activations at query and candidate key positions",
                "shape": ["batch", "selected_position", 4608],
                "intervention": "complete donor product-state interchange before any DAS fit",
                "ceiling_event_ids": [],
            },
            {
                "site_id": "mlp9.write.copy_reader",
                "tensor_path": "MLP9 residual write at copy-positive positions",
                "shape": ["batch", 1152],
                "intervention": "measure whether a proposed score intervention reproduces the calibrated downstream reader response",
                "ceiling_event_ids": ["equality_mlp9_reader.r500.v1"],
            },
        ],
        "split_plan_ids": ["legacy_natural_code_roles_v1", "equality_multifamily_future_v1"],
        "evidence_event_ids": [],
        "translation_ids": [],
        "next_missing": (
            "materialize the text-edit and matched-natural answer-changing families plus the payload-preserving "
            "invariance family; then measure complete-state query/key/MLP7 ceilings with identical patch semantics "
            "before fitting a shared subspace"
        ),
    }
    record = {
        "schema_version": 2,
        "tag": TAG,
        "identity": {
            "kind": "shared_subroutine",
            "instance": None,
            "identity_artifact_id": "r536_independent_audit",
            "aliases": ["L5H5_score_to_L8H4_payload", "copy_equality_score", "four_head_equality_score_family"],
        },
        "claims": [claim],
        "split_plans": [
            {
                "split_plan_id": "legacy_natural_code_roles_v1",
                "unit": "document, with disjoint halves within each natural/code corpus role",
                "partition_artifact_id": "terminal_copy_rows_receipt",
                "builder_artifact_id": "r498_implementation",
                "seed": None,
                "groups": {"FIT": 96, "SELECT": 96, "FINAL_TEST": 0, "OOD": 192},
                "leakage_group_keys": ["document", "natural versus code", "near versus far", "one versus multiple predecessor"],
                "sealed_before_outcomes": True,
                "sealed_at": "legacy lineage; see per-rung preregistrations",
            },
            {
                "split_plan_id": "equality_multifamily_future_v1",
                "unit": "document/template/query/payload group shared across every future counterfactual family",
                "partition_artifact_id": None,
                "builder_artifact_id": "synthetic_copy_builder",
                "seed": None,
                "groups": {"FIT": 0, "SELECT": 0, "FINAL_TEST": 0, "OOD": 0},
                "leakage_group_keys": ["document", "template", "query token", "payload token", "lag", "corpus"],
                "sealed_before_outcomes": False,
                "sealed_at": None,
            },
        ],
        "evidence_events": [],
        "translations": [],
        "artifacts": artifacts,
        "provenance": {
            "rung": 541,
            "relationship": "shared subroutine within task.induction.selector_payload; not a duplicate behavior circuit",
            "audit_sha256": artifacts["r536_independent_audit"]["sha256"],
        },
    }

    r459_validation = r459["validation"]
    r460_analysis = r460["analysis"]
    event_specs = [
        event(
            "equality_score_swap.r459.natural.v1", "composition", "held", None,
            ["cross_head_score_swap"], "attention8.head4.double_qk_score",
            [
                metric("heldout_causal_recovery", r459_validation["causal_recovery"]["recovery"],
                       "registered recovery and causal-effect bars pass",
                       [r459_validation["causal_recovery"]["simultaneous_95_lower"], None]),
                metric("off_target_CE_difference_nat", r459_validation["off_target_hybrid_minus_reference_nat"],
                       "below registered off-target bar"),
            ],
            "r459_preregistration", "r459_result", ["r459_implementation"],
            r459["checkpoint_weights_sha256"],
            ["EQUALITY_TERM_SCORE_PAYLOAD_RUNG459_PREREGISTRATION.md"],
            "Natural-text held-out evidence that L5H5's complete score can replace L8H4's score while retaining L8H4's payload.",
        ),
        event(
            "equality_score_swap.r460.code.v1", "ood", "failed", "scientific_null",
            ["cross_head_score_swap"], "attention8.head4.double_qk_score",
            [
                metric("code_causal_recovery", r460_analysis["selected_causal_recovery"]["recovery"],
                       "positive causal recovery",
                       [r460_analysis["selected_causal_recovery"]["simultaneous_95_lower"], None]),
                metric("registered_prediction_fraction", 4 / 5, "all five preregistered predictions pass"),
            ],
            "r460_preregistration", "r460_result", ["r459_result"],
            r460["checkpoint_weights_sha256"],
            ["EQUALITY_SCORE_CODE_OOD_RUNG460_PREREGISTRATION.md"],
            "The causal transplant recovered the code effect, but the registered downstream-response condition failed; preserve as mixed OOD evidence, not a full confirmation.",
        ),
        event(
            "equality_downstream_gate.r462.null.v1", "composition", "null", "scientific_null",
            ["cross_head_score_swap"], "mlp9.write.copy_reader",
            [metric("registered_prediction_fraction", 1 / 5, "all four scientific predictions pass after the instrument gate")],
            "r462_preregistration", "r462_result", ["r459_result", "r460_result"],
            r462["checkpoint_weights_sha256"],
            ["EQUALITY_SCORE_DOWNSTREAM_GATE_RUNG462_PREREGISTRATION.md"],
            "No single tested downstream mediator isolated the transplanted score's full effect.",
        ),
        event(
            "equality_source_correction.r464.v1", "composition", "held", None,
            ["cross_head_score_swap"], "mlp9.write.copy_reader",
            [
                metric("registered_prediction_fraction", 1.0, "all five preregistered predictions pass"),
                metric("matched_correction_cosine", r464["analysis"]["pooled"]["matched_correction_comparison"]["cosine"],
                       ">=0.9 under both score sources"),
            ],
            "r464_preregistration", "r464_result", ["r459_result", "r460_result"],
            r464["checkpoint_weights_sha256"],
            ["EQUALITY_SCORE_CORRECTION_INTERCHANGE_RUNG464_PREREGISTRATION.md"],
            "Native and transplanted scores share a downstream context-dependent correction, but the correction is not an autonomous circuit.",
        ),
        event(
            "equality_action_quotient.r498.null.v1", "cross_family_transfer", "null", "scientific_null",
            ["cross_head_score_swap", "matched_natural_whole_state_swap", "payload_swap_match_preserved"],
            "attention8.head4.double_qk_score",
            [metric("registered_prediction_fraction", 2 / 6, "all six preregistered clauses pass")],
            "r498_preregistration", "r498_result", ["r498_implementation", "r459_result"],
            r498["checkpoint_weights_sha256"],
            ["EQUALITY_MATCHER_CAUSAL_ACTION_QUOTIENT_RUNG498_PREREGISTRATION.md"],
            "A broad finite action quotient did not recover the known positive or reject all controls; do not repeat this observation set.",
        ),
        event(
            "equality_mlp9_reader.r500.v1", "composition", "held", None,
            ["cross_head_score_swap", "payload_swap_match_preserved"], "mlp9.write.copy_reader",
            [
                metric("registered_prediction_fraction", 1.0, "all six preregistered clauses pass"),
                metric("minimum_copy_specificity_cosine_margin",
                       min(x["cosine_margin"] for x in r500["checks"]["copy_specificity"]),
                       "positive in every background and document quarter"),
            ],
            "r500_preregistration", "r500_result", ["r498_result"],
            r500["checkpoint_weights_sha256"],
            ["EQUALITY_MATCHER_MLP9_READER_CALIBRATION_RUNG500_PREREGISTRATION.md"],
            "MLP9 is a calibrated downstream reader of the known score relation and rejects the L7H3-score and L5H5-payload controls.",
        ),
        event(
            "equality_factor_branch_sharing.r531.null.v1", "null_control", "null", "scientific_null",
            ["cross_head_score_swap"], "attention8.head4.double_qk_score",
            [metric("best_heldout_single_factor_cosine", 0.8598, ">=0.90 with product reconstruction and gauge consistency")],
            "r531_preregistration", "r531_result", ["r531_receipt", "r459_result"],
            r531["checkpoint"]["weights_sha256"],
            ["EQUALITY_SCORE_FACTOR_BRANCH_SHARING_RUNG531_TERMINAL_RECEIPT.md"],
            "The shared object is supported at complete-product/score level, not as an identified individual Q or K factor.",
        ),
        event(
            "terminal_copy_four_head_removal.collateral_failure.v1", "removal", "failed", "scientific_null",
            ["match_break_answer_fixed"], "terminal_copy_service",
            [metric("four_head_collateral_margin_simultaneous_lower", terminal["bootstrap_simultaneous_lower_bounds"][20],
                    "positive and above the registered collateral-preservation bar")],
            None, "terminal_copy_negative_receipt", ["terminal_copy_rows_receipt"],
            None, ["terminal_copy_selection_v1_attempt2_negative_receipt.json"],
            "The broad four-head removal had a positive task signal but failed unrelated-behavior preservation; it is not a validated terminal-copy circuit.",
        ),
    ]
    for item in event_specs:
        item["design_key"] = design_key(record, item)
        item["execution_key"] = execution_key(record, item)
        record["evidence_events"].append(item)
        claim["evidence_event_ids"].append(item["event_id"])

    path = write_behavior_circuit(record)
    print(json.dumps({
        "tag": TAG,
        "claim_id": CLAIM_ID,
        "status": claim["status"],
        "events": len(event_specs),
        "families": len(families),
        "path": str(path.relative_to(REPO)),
        "gpu_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
