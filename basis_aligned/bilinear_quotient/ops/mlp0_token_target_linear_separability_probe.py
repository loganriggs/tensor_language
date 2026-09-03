#!/usr/bin/env python3
"""Can any fixed linear projector separate MLP0's token-only change from its token-by-context change? (CPU probe, zero forwards, weights only)

# BQGATE: EXPERIMENT
# pred_a_instrument_closed_form_matches_monte_carlo
# pred_b_residual_monotone_and_large_at_rho2
# pred_c_near_separable_at_small_context

R536 token target: observed Dg_T+Dg_I (donor token, base context) -> target Dg_T. Wiener (any-rank) and
reduced-rank (rank-k) linear lower bounds in the W_D output metric, scanned over context scale rho.
DISCLOSURE: rho=1 was observed during library smoke-testing; scored clauses concern only the rho-scan shape.
Preregistration: polynomial_causal/MLP0_TOKEN_TARGET_LINEAR_SEPARABILITY_PROBE_PREREGISTRATION.md
Math: ops/mlp0_hybrid_separability_lib.py (hash-frozen). Lower residual = more separable (1 = nothing recovered).
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import torch
from receipt import dump
import mlp0_hybrid_separability_lib as LIB

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_TOKEN_TARGET_LINEAR_SEPARABILITY_PROBE_PREREGISTRATION.md"
LIBP = ROOT / "ops" / "mlp0_hybrid_separability_lib.py"
OUT = ROOT / "mlp0_token_target_linear_separability_probe_results.json"
HASHES = {
    PREREG: "2110805ffe8245a7dbaa58ac2d97a2ac81a494cadf1664bb64c13a2140d173b9",
    LIBP: "0168d99083c003846b4f14e62dec645e3c801c725201a2bd0c681b0147ae28c8",
}
RHOS = [0.25, 0.5, 1.0, 2.0]
KS = [3, 8, 32, 128, 512, 1152, 4608]
MC_N = 4000
MC_SEED = 1
RUNG = "mlp0_token_target_linear_separability_probe"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_hashes():
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    L, R, D, p = LIB.load_mlp0()
    Mp, Sp, _ = LIB.token_moments(p)
    S_T, _ = LIB.cov_gT(L, R, p)
    ev_min = float(torch.linalg.eigvalsh(S_T).min()); ev_max = float(torch.linalg.eigvalsh(S_T).max())
    per_rho = {}
    for rho in RHOS:
        S_I = LIB.cov_gI_given_token_moment(L, R, Sp, rho)
        per_rho[str(rho)] = LIB.wiener_residuals(S_T, S_I, D, KS)
    pure_ladder, pure_eff = LIB.pure_target_rank_ladder(S_T, D, KS)
    S_I1 = LIB.cov_gI_given_token_moment(L, R, Sp, 1.0)
    mc = LIB.mc_check_token_target(L, R, p, 1.0, MC_N, MC_SEED)
    cf_t = 2.0 * float(S_T.diagonal().sum()); cf_n = 2.0 * float(S_I1.diagonal().sum())
    dev_t = abs(mc["mc_E_sq_target"] / cf_t - 1.0); dev_n = abs(mc["mc_E_sq_nuisance"] / cf_n - 1.0)
    pred_a = bool(dev_t <= 0.05 and dev_n <= 0.05 and abs(mc["mc_cross_normalized"]) <= 0.05
                  and ev_min >= -1e-8 * ev_max)
    res = [per_rho[str(r)]["residual_fraction_output_metric"] for r in RHOS]
    monotone = all(res[i] < res[i + 1] for i in range(len(res) - 1))
    pred_b = bool(pred_a and monotone and res[3] >= 0.50)
    pred_c = bool(pred_a and res[0] <= 0.15)
    strong_null = bool(not (pred_a and pred_b and pred_c))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c:
        verdict = "token_target_separability_is_a_strong_function_of_context_scale"
    elif pred_b and not pred_c:
        verdict = "token_and_context_branches_entangled_at_every_scale"
    elif not pred_b and pred_c:
        verdict = "nuisance_saturates_separable_only_at_small_context"
    else:
        verdict = "neither_shape_prediction_held"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_weight_space_linear_lower_bound_under_stated_isotropic_context_model_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()},
        "target": "token: observed Dg_T+Dg_I -> target Dg_T (R536 hybrid addendum)",
        "input_model": "p=unit-rms wte rows uniform over 50257; q zero-mean isotropic E[qq^T]=rho^2 I",
        "per_rho": per_rho,
        "pure_target_rank_ladder_rho0": pure_ladder, "pure_target_output_effective_rank": pure_eff,
        "mc_check_rho1": {**mc, "closed_form_2trS_T": cf_t, "closed_form_2trS_I": cf_n,
                          "rel_dev_target": dev_t, "rel_dev_nuisance": dev_n},
        "S_T_eig_min_max": [ev_min, ev_max],
        "disclosure": "rho=1 numbers were observed during library smoke-testing before registration; reported not scored",
        "bars": {"rho2_residual_min": 0.50, "rho025_residual_max": 0.15, "mc_rel_tol": 0.05},
        'pred_a_instrument_closed_form_matches_monte_carlo': pred_a,
        'pred_b_residual_monotone_and_large_at_rho2': pred_b,
        'pred_c_near_separable_at_small_context': pred_c,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0, "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "verdict", "strong_null", "per_rho",
                      "pure_target_output_effective_rank", "runtime_s") or k.startswith("pred_")}, indent=2))


if __name__ == "__main__":
    main()
