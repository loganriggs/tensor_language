#!/usr/bin/env python3
"""Audit committed Task14 evidence that postdates registry claim v22."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CIRCUITS = REPO / "basis_aligned/bilinear_quotient/circuits"
OUT = HERE / "SUBJECT_NUMBER_POST_V22_AUTHORITY_AUDIT_V1.json"
FILES = {
    "ood_tangent": CIRCUITS / "fast_screens/task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_v1_result.json",
    "ood_gate": CIRCUITS / "fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json",
    "prompt_loo": CIRCUITS / "fast_screens/task14_ood_mlp6_7_prompt_level_composition_crossvalidation_v1_result.json",
    "fresh_composition": CIRCUITS / "fast_screens/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_result.json",
    "continuous_edit": CIRCUITS / "fast_screens/task14_fresh_fronted_mlp6_7_continuous_background_gain_manipulation_v1_result.json",
    "zero_anchor": CIRCUITS / "fast_screens/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1_result.json",
    "fixed_reader": CIRCUITS / "fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json",
    "guided_edit": CIRCUITS / "fast_screens/task14_fixed_reader_guided_margin_edit_v1_result.json",
    "upstream_program": CIRCUITS / "fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json",
    "narrow_collateral": CIRCUITS / "fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2_result.json",
    "possessive_reuse": CIRCUITS / "fast_screens/task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2_result.json",
    "mediator_split": CIRCUITS / "followups/task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation_v1_result.json",
    "registry": CIRCUITS / "task_subject_verb_number_agreement.json",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(key): return json.loads(FILES[key].read_text())


def predicates(value): return value.get("predictions") or value.get("score", {}).get("predictions", {})


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    values = {key: load(key) for key in FILES}
    registry_claim = values["registry"]["claims"][-1]
    assert registry_claim["claim_id"] == "grammatical_subject_number.v22"
    for key in ("prompt_loo", "fresh_composition", "continuous_edit", "fixed_reader", "guided_edit", "upstream_program"):
        assert all(predicates(values[key]).values()), key
    assert all(predicates(values["narrow_collateral"]).values())
    assert predicates(values["zero_anchor"])["pred_d_zero_anchor_holdout_prediction"] is False
    assert predicates(values["possessive_reuse"])["pred_c_correct_write_moves_possessive_margin_unchanged"] is False
    assert predicates(values["mediator_split"])["pred_b_additive_distributed_pair"] is True
    assert predicates(values["mediator_split"])["pred_c_single_dominant_mediator"] is False
    fixed = values["fixed_reader"]["score"]
    guided = values["guided_edit"]["score"]
    upstream = values["upstream_program"]["score"]
    collateral = values["narrow_collateral"]["score"]
    prompt = values["prompt_loo"]["score"]
    payload = {
        "schema": "subject_number_post_v22_authority_audit_v1",
        "model_loaded": False, "gpu_accessed": False,
        "input_sha256": {key: sha(path) for key, path in FILES.items()},
        "registry_before": {"claim_id": registry_claim["claim_id"], "status": registry_claim["status"], "next_missing": registry_claim["next_missing"]},
        "verified_evidence": {
            "ood_geometry": {"minimum_cosine": values["ood_tangent"]["score"]["minimum_ood_midpoint_cosine"], "maximum_relative_error": values["ood_tangent"]["score"]["maximum_ood_midpoint_relative_error"], "context_gated": True},
            "distributed_background_gate": {"predictions": predicates(values["ood_gate"]), "direction_context_shift": values["ood_gate"]["score"]["direction_context_shift"]},
            "prompt_level_composition": {"median_normalized_mae": prompt["median_live_normalized_mae"], "sse_reduction_over_uniform": prompt["live_aggregate_sse_reduction_over_uniform"]},
            "fresh_construction_composition": {"predictions": predicates(values["fresh_composition"])},
            "continuous_manipulation": {"predictions": predicates(values["continuous_edit"])},
            "zero_anchor_absolute_coefficients": {"predictions": predicates(values["zero_anchor"]), "interpretation": "construction-dependent baseline/amplitude remains necessary"},
            "fixed_two_reader_cross_corpus": {"overall": fixed["overall"], "intermediate_only": fixed["intermediate_only"], "sse_reduction_over_swapped_reader": fixed["sse_reduction_over_swapped_reader"], "stored_scalars": 2304, "target_tail_forwards": 0, "target_tail_backwards": 0},
            "reader_guided_absolute_edit": {"guided": guided["guided"], "mae_reduction_over_half_gain": guided["mae_reduction_over_half_gain"]},
            "ten_vector_upstream_program": {"native_substitution": upstream["native_substitution"], "reader_prediction": upstream["reader_prediction_of_installed_cardinality_effect"], "sse_reduction_over_direction_only": upstream["sse_reduction_over_direction_only"], "stored_vectors": 10},
            "narrow_cross_circuit_collateral": {"predictions": predicates(values["narrow_collateral"]), "maximum_install_error": collateral["repaired_maximum_install_absolute_error"]},
            "cross_task_possessive_reuse": {"terminal": values["possessive_reuse"]["terminal"], "predictions": predicates(values["possessive_reuse"])},
            "downstream_mediation": {"terminal": values["mediator_split"]["terminal"], "predictions": predicates(values["mediator_split"])},
        },
        "audit_conclusion": {
            "registry_is_stale": True,
            "already_tested": ["OOD and independent-text transfer", "two-reader extraction", "full-lattice composition", "prospective guided manipulation", "literal upstream program pricing", "narrow bracket/numbered-list collateral"],
            "supported_program": "ten direction-by-cardinality 1152D upstream writes plus two fixed 1152D downstream readers at the L11H3 projected-write boundary, with native generators and suffix external",
            "preserved_failures": ["global zero-anchor absolute coefficients", "generic possessive-number reuse", "single dominant MLP15 or MLP17 mediator", "universal lexical specificity"],
            "next_action": "register post-v22 receipts before selecting a new subject-number experiment; do not repeat OOD, gain, reader, program, or collateral screens",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["audit_conclusion"], indent=2))


if __name__ == "__main__": main()
