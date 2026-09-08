#!/usr/bin/env python3
"""Conditionally split necessary A11 into fixed singleton/LOO head endpoints."""

# BQGATE: EXPERIMENT pred_a_conditional_authority_capture_hook_coverage_finiteness_and_exact_price pred_b_all_nine_heads_replay_complete_a11_and_parent_joint_program pred_c_prospectively_nominated_h3_is_stably_necessary_and_sufficient_at_the_endpoint pred_d_h3_explains_at_least_seventy_percent_of_complete_a11_contribution pred_e_all_head_arms_preserve_temporal_selectivity
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_p7_residual_two_stream_ood_composition_v1.py"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.json"
OUT = ROOT / "circuits/followups/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1_result.json"
EXPECTED = {
    "prior": "afca834f62f088ad689619d8c58d424344b55bf59eb24f356badb57c7759ed99",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "parent_runner": "0e3f6de1b273d372006652e37c085ddf15d5b120795225ba873de82a402a19ec",
    "necessity_runner": "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT,
         "parent_runner": PARENT_RUNNER, "necessity_runner": NECESSITY_RUNNER}
HEADS = tuple(range(9))
ARMS = (("none", "all_nine")
        + tuple(f"H{head}_singleton" for head in HEADS)
        + tuple(f"without_H{head}" for head in HEADS))
PRICE = {"checkpoint_loads": 1, "model_forwards": 22,
         "sequence_evaluations": 2816, "scored_token_positions": 5632,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_conditional_authority_capture_hook_coverage_finiteness_and_exact_price",
    "pred_b_all_nine_heads_replay_complete_a11_and_parent_joint_program",
    "pred_c_prospectively_nominated_h3_is_stably_necessary_and_sufficient_at_the_endpoint",
    "pred_d_h3_explains_at_least_seventy_percent_of_complete_a11_contribution",
    "pred_e_all_head_arms_preserve_temporal_selectivity",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_head_slices(native, writer, heads, position_rows, *, n_heads=9):
    """Replace selected concatenated pre-c_proj head slices at registered row positions."""
    heads = tuple(int(head) for head in heads)
    if native.shape != writer.shape or native.ndim != 3:
        raise ValueError("head tensors must share [batch, token, residual] shape")
    if len(set(heads)) != len(heads) or any(head < 0 or head >= n_heads for head in heads):
        raise ValueError("heads must be unique valid indices")
    if native.shape[-1] % n_heads:
        raise ValueError("residual width must divide into heads")
    if len(position_rows) != native.shape[0]:
        raise ValueError("position row count mismatch")
    width = native.shape[-1] // n_heads
    changed = native.clone()
    for row, positions in enumerate(position_rows):
        for position in positions:
            if position < 0 or position >= native.shape[1]:
                raise ValueError("head patch position out of range")
            for head in heads:
                sl = slice(head * width, (head + 1) * width)
                changed[row, position, sl] = writer[row, position, sl]
    return changed


def eligibility(binding, result):
    return bool(binding.get("necessity_result_sha256") == sha(NECESSITY_RESULT)
        and binding.get("necessity_runner_sha256") == EXPECTED["necessity_runner"]
        and result.get("predictions", {}).get(
            "pred_a_authority_capture_full_replay_finiteness_and_exact_price") is True
        and result.get("predictions", {}).get(
            "pred_c_attention11_is_stably_necessary_and_licenses_head_splitting") is True
        and result.get("predictions", {}).get(
            "pred_e_every_leave_one_out_arm_is_temporally_selective") is True
        and "A11" in result.get("stable_necessary_modules", ()))


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    base_ok = observed == EXPECTED and prior.get("price") == PRICE
    waiting = {"candidate_id": prior.get("candidate_id"), "status": "awaiting_binding",
        "authority_ok": base_ok, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "arms": list(ARMS), "price": PRICE}
    if not BINDING.exists() or not NECESSITY_RESULT.exists():
        print(json.dumps(waiting, sort_keys=True))
        return
    binding = json.loads(BINDING.read_text())
    result = json.loads(NECESSITY_RESULT.read_text())
    if not base_ok or not eligibility(binding, result):
        raise RuntimeError("A11 endpoint atlas eligibility binding failed")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({**waiting, "status": "eligible_bound"}, sort_keys=True))
        return
    raise RuntimeError("eligible binding landed before the conditional executor was completed")


if __name__ == "__main__":
    main()
