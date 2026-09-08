#!/usr/bin/env python3
"""Conditionally factor necessary M11 into exact activation-conditioned products."""

# BQGATE: EXPERIMENT pred_a_conditional_authority_capture_factor_closure_hook_coverage_finiteness_and_exact_price pred_b_all_three_factors_replay_complete_m11_and_parent_joint_program pred_c_bilinear_interaction_is_stably_necessary pred_d_at_least_one_linear_factor_is_stably_material pred_e_all_factor_arms_preserve_temporal_selectivity
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.json"
EXPECTED = {
    "prior": "1aac98581f8ae575cfad2ca7d7a3daf7de536b2c5aef090566b22b7fd65d75d0",
    "parent_result": "55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7",
    "necessity_runner": "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a",
}
FILES = {"prior": PRIOR, "parent_result": PARENT_RESULT,
         "necessity_runner": NECESSITY_RUNNER}
FACTORS = ("left", "right", "interaction")
ARMS = ("none", "complete_M11", "left", "right", "interaction",
        "left_right", "left_interaction", "right_interaction", "all_three")
PRICE = {"checkpoint_loads": 1, "model_forwards": 11,
         "sequence_evaluations": 1408, "scored_token_positions": 2816,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_conditional_authority_capture_factor_closure_hook_coverage_finiteness_and_exact_price",
    "pred_b_all_three_factors_replay_complete_m11_and_parent_joint_program",
    "pred_c_bilinear_interaction_is_stably_necessary",
    "pred_d_at_least_one_linear_factor_is_stably_material",
    "pred_e_all_factor_arms_preserve_temporal_selectivity",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_product_factors(live_left, live_right, writer_left, writer_right):
    shapes = {tuple(x.shape) for x in (live_left, live_right, writer_left, writer_right)}
    if len(shapes) != 1 or live_left.ndim != 3:
        raise ValueError("factor tensors must share [batch, token, hidden] shape")
    delta_left = writer_left.float() - live_left.float()
    delta_right = writer_right.float() - live_right.float()
    return {
        "left": delta_left * live_right.float(),
        "right": live_left.float() * delta_right,
        "interaction": delta_left * delta_right,
    }


def compose_hidden(live_left, live_right, factors, subset):
    subset = tuple(subset)
    if len(set(subset)) != len(subset) or not set(subset).issubset(FACTORS):
        raise ValueError("invalid factor subset")
    hidden = live_left.float() * live_right.float()
    for name in subset:
        hidden = hidden + factors[name]
    return hidden


def closure_relative_error(live_left, live_right, writer_left, writer_right):
    factors = exact_product_factors(live_left, live_right, writer_left, writer_right)
    predicted = compose_hidden(live_left, live_right, factors, FACTORS)
    exact = writer_left.float() * writer_right.float()
    denominator = max(float(exact.norm()), 1e-30)
    return float((predicted - exact).norm()) / denominator


def eligibility(binding, result):
    return bool(binding.get("necessity_result_sha256") == sha(NECESSITY_RESULT)
        and binding.get("necessity_runner_sha256") == EXPECTED["necessity_runner"]
        and result.get("predictions", {}).get(
            "pred_a_authority_capture_full_replay_finiteness_and_exact_price") is True
        and result.get("predictions", {}).get(
            "pred_d_at_least_one_p7_mlp_is_stably_necessary_and_licenses_weight_factor_splitting") is True
        and result.get("predictions", {}).get(
            "pred_e_every_leave_one_out_arm_is_temporally_selective") is True
        and "M11" in result.get("stable_necessary_modules", ()))


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    base_ok = observed == EXPECTED and prior.get("price") == PRICE
    waiting = {"candidate_id": prior.get("candidate_id"), "status": "awaiting_binding",
        "authority_ok": base_ok, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "factors": list(FACTORS), "arms": list(ARMS),
        "price": PRICE}
    if not BINDING.exists() or not NECESSITY_RESULT.exists():
        print(json.dumps(waiting, sort_keys=True))
        return
    binding = json.loads(BINDING.read_text())
    result = json.loads(NECESSITY_RESULT.read_text())
    if not base_ok or not eligibility(binding, result):
        raise RuntimeError("M11 product-factor eligibility binding failed")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({**waiting, "status": "eligible_bound"}, sort_keys=True))
        return
    raise RuntimeError("eligible binding landed before the conditional executor was completed")


if __name__ == "__main__":
    main()
