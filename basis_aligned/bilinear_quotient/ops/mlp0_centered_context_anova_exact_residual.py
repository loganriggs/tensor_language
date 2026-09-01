"""RUNG 401 -- EXACT VECTOR NORMALIZATION-RESIDUAL REPAIR OF RUNG 400.

Keep every T/C/I/S physical intervention from rung400 unchanged.  Replace the
failed scalar-only identity diagnostic with the exact observed-state expansion

    z = s(e+a) + r,
    T(z,z) = s^2 T(e+a,e+a)
             + T(s(e+a),r) + T(r,s(e+a)) + T(r,r).

The explicit residual R is retained in every arm, as it already was implicitly
in rung400's native-minus-omitted hook.  Rerun the identical FIT/SELECT 16-arm
factorial in a new namespace and compare every outcome with rung400.

Frozen predictions
------------------
pred_a: repaired analytical identity relative MSE<=1e-8 in both roles and live
    call census holds.
pred_b: every pooled arm CE reproduces rung400 within1e-6 and every T/C/I/S
    Shapley value within1e-5.
pred_c: preserve the original frozen outcome honestly: old pred_b remains
    failed, old pred_c holds, I is largest on SELECT, and combined C+I+S>=1.5.
pred_d: T/C/I/S signs and complete rank order match FIT and SELECT.

Strong null: exactness or reproduction fails; repaired float-vs-deployed
residual exceeds1e-5; combined context<=.50; or a context sign reverses.
Full pass validates exact attribution only.  No FINAL, compression, adoption,
or live-context replacement is licensed.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OPS = ROOT / "ops"
OUT = ROOT / "mlp0_centered_context_anova_exact_residual_results.json"
PARENT = ROOT / "mlp0_centered_context_anova_factorial_results.json"


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.name == "mlp0_centered_context_anova_factorial_results.json"
        print("MLP0 EXACT CONTEXT RESIDUAL | dry run: identity, replay, bars valid")
        return

    started = time.time()
    if not PARENT.exists():
        raise RuntimeError("rung400 parent result is missing")
    parent = json.loads(PARENT.read_text())
    sys.path.insert(0, str(OPS))
    import mlp0_centered_context_anova_factorial as base

    original_components = base._components

    @torch.no_grad()
    def exact_components(token_base, attention_write, normalized,
                         reference, left, right, down):
        retained, branches, g, gain, collinearity = original_components(
            token_base, attention_write, normalized, reference, left, right, down)
        z = normalized.float()
        raw = token_base.float() + attention_write.float()
        denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
        scale = (z * raw).sum(-1, keepdim=True) / denominator
        scaled_raw = scale * raw
        residual = z - scaled_raw
        explicit_residual = (
            base._T(scaled_raw, residual, left, right, down)
            + base._T(residual, scaled_raw, left, right, down)
            + base._T(residual, residual, left, right, down))
        return retained + explicit_residual, branches, g, gain, collinearity

    base._components = exact_components
    base.OUT = OUT
    base.main()
    result = json.loads(OUT.read_text())

    max_ce_difference = 0.0
    max_shapley_difference = 0.0
    for role in ("FIT", "SELECT"):
        for arm, value in result["roles"][role]["pooled_ce"].items():
            max_ce_difference = max(
                max_ce_difference,
                abs(value - parent["roles"][role]["pooled_ce"][arm]))
        for branch, value in result["roles"][role]["shapley_ce_benefit"].items():
            max_shapley_difference = max(
                max_shapley_difference,
                abs(value - parent["roles"][role]["shapley_ce_benefit"][branch]))

    fit = result["roles"]["FIT"]["shapley_ce_benefit"]
    select = result["roles"]["SELECT"]["shapley_ce_benefit"]
    identity_hold = all(
        result["roles"][role]["diagnostics"]["analytical_identity_relative_mse"] <= 1e-8
        and result["roles"][role]["diagnostics"]["live_census"]
        for role in ("FIT", "SELECT"))
    replay_hold = max_ce_difference <= 1e-6 and max_shapley_difference <= 1e-5
    combined_context = select["C"] + select["I"] + select["S"]
    old_pred_b_failed = not bool(parent["pred_b_context_main_and_centered_cross_are_large"])
    inherited_c = (
        old_pred_b_failed
        and abs(select["S"]) >= .05
        and combined_context >= select["T"]
        and max(select, key=select.get) == "I"
        and combined_context >= 1.5)
    signs_match = all((fit[name] >= 0) == (select[name] >= 0) for name in base.BRANCHES)
    fit_order = sorted(base.BRANCHES, key=lambda name: fit[name], reverse=True)
    select_order = sorted(base.BRANCHES, key=lambda name: select[name], reverse=True)
    transport_hold = signs_match and fit_order == select_order
    deployed_hold = all(
        result["roles"][role]["diagnostics"]["deployed_bf16_residual_relative_mse"] <= 1e-5
        for role in ("FIT", "SELECT"))
    sign_reversal = any(
        abs(fit[name]) >= .05 and (fit[name] >= 0) != (select[name] >= 0)
        for name in ("C", "I", "S"))
    strong_null = (
        not identity_hold or not replay_hold or not deployed_hold
        or combined_context <= .50 or sign_reversal)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            del result[key]
    result.update({
        "status": "mlp0_centered_context_anova_exact_residual_complete",
        "rung": 401,
        "claim_level": "exact_context_causal_attribution_not_compression",
        "exact_normalization_residual": (
            "R=T(s(e+a),r)+T(r,s(e+a))+T(r,r), r=z-s(e+a); retained in every arm"),
        "parent_rung400_result": str(PARENT),
        "maximum_pooled_arm_ce_difference_vs_rung400": max_ce_difference,
        "maximum_shapley_difference_vs_rung400": max_shapley_difference,
        "original_rung400_pred_b_remains_failed": old_pred_b_failed,
        "select_combined_context_shapley": combined_context,
        'pred_a_exact_residual_identity_and_live_census': bool(identity_hold),
        'pred_b_all_physical_arms_reproduce_rung400': bool(replay_hold),
        'pred_c_inherited_context_outcome_holds_without_bar_change': bool(inherited_c),
        'pred_d_fit_select_sign_and_order_transport': bool(transport_hold),
        "null_exact_repair_or_context_stability_fails": bool(strong_null),
        "next_context_branch": "I" if not strong_null and inherited_c else None,
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "runtime_s": time.time() - started,
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 EXACT CONTEXT RESIDUAL DONE", flush=True)


if __name__ == "__main__":
    main()
