#!/usr/bin/env python3
"""Can any fixed linear projector separate MLP0's token-by-context change from its context-only change? (CPU probe, zero forwards, weights only)

# BQGATE: EXPERIMENT
# pred_a_instrument_closed_form_matches_monte_carlo
# pred_b_no_exact_linear_separator_at_rho1
# pred_c_rank32_read_insufficient_at_rho1

R536 context target: observed Dg_I+Dg_C (donor context, base token) -> target Dg_I. Wiener (any-rank) and
reduced-rank (rank-k) linear lower bounds in the W_D output metric, scanned over context scale rho. Unseen.
Preregistration: polynomial_causal/MLP0_CONTEXT_TARGET_LINEAR_SEPARABILITY_PROBE_PREREGISTRATION.md
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
PREREG = POLY / "MLP0_CONTEXT_TARGET_LINEAR_SEPARABILITY_PROBE_PREREGISTRATION.md"
LIBP = ROOT / "ops" / "mlp0_hybrid_separability_lib.py"
OUT = ROOT / "mlp0_context_target_linear_separability_probe_results.json"
HASHES = {
    PREREG: "ac3ceb2f3470c1b0cdda3bfe8086ab2f21a405ec9f0b7ff114dc5f39dc634846",
    LIBP: "0168d99083c003846b4f14e62dec645e3c801c725201a2bd0c681b0147ae28c8",
}
RHOS = [0.25, 0.5, 1.0, 2.0]
KS = [3, 8, 32, 128, 512, 1152, 4608]
MC_N = 4000
MC_SEED = 2
RUNG = "mlp0_context_target_linear_separability_probe"


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
    S_I1 = LIB.cov_gI_given_token_moment(L, R, Mp, 1.0)      # Cov(Dg_I) up to the common factor 2
    ev_min = float(torch.linalg.eigvalsh(S_I1).min()); ev_max = float(torch.linalg.eigvalsh(S_I1).max())
    per_rho = {}
    for rho in RHOS:
        S_I = LIB.cov_gI_given_token_moment(L, R, Mp, rho)
        S_C = LIB.cov_gC_gaussian(L, R, rho)
        per_rho[str(rho)] = LIB.wiener_residuals(S_I, S_C, D, KS)
    pure_ladder, pure_eff = LIB.pure_target_rank_ladder(S_I1, D, KS)
    S_C1 = LIB.cov_gC_gaussian(L, R, 1.0)
    mc = LIB.mc_check_context_target(L, R, p, 1.0, MC_N, MC_SEED)
    cf_t = 2.0 * float(S_I1.diagonal().sum()); cf_n = 2.0 * float(S_C1.diagonal().sum())
    dev_t = abs(mc["mc_E_sq_target"] / cf_t - 1.0); dev_n = abs(mc["mc_E_sq_nuisance"] / cf_n - 1.0)
    pred_a = bool(dev_t <= 0.05 and dev_n <= 0.05 and abs(mc["mc_cross_normalized"]) <= 0.05
                  and ev_min >= -1e-8 * ev_max)
    r1 = per_rho["1.0"]
    pred_b = bool(pred_a and r1["residual_fraction_output_metric"] >= 0.20)
    pred_c = bool(pred_a and r1["residual_fraction_output_metric_rank_ladder"][32] >= 0.60)
    strong_null = bool(not (pred_a and pred_b and pred_c))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c:
        verdict = "context_target_not_linearly_separable_and_not_low_dim"
    elif pred_b and not pred_c:
        verdict = "context_target_inseparable_exactly_but_low_dim_read_recovers_majority"
    elif not pred_b:
        verdict = "context_target_near_separable_easier_pilot_than_token_target"
    else:
        verdict = "unreached"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_weight_space_linear_lower_bound_under_stated_isotropic_gaussian_context_model_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()},
        "target": "context: observed Dg_I+Dg_C -> target Dg_I (R536 hybrid addendum)",
        "input_model": "p_b=unit-rms wte rows uniform (uncentered 2nd moment); q_b,q_d iid Gaussian E[qq^T]=rho^2 I",
        "per_rho": per_rho,
        "pure_target_rank_ladder_rho0": pure_ladder, "pure_target_output_effective_rank": pure_eff,
        "mc_check_rho1": {**mc, "closed_form_trCovDgI": cf_t, "closed_form_trCovDgC": cf_n,
                          "rel_dev_target": dev_t, "rel_dev_nuisance": dev_n},
        "S_I_eig_min_max": [ev_min, ev_max],
        "bars": {"rho1_residual_min": 0.20, "rho1_rank32_residual_min": 0.60, "mc_rel_tol": 0.05},
        'pred_a_instrument_closed_form_matches_monte_carlo': pred_a,
        'pred_b_no_exact_linear_separator_at_rho1': pred_b,
        'pred_c_rank32_read_insufficient_at_rho1': pred_c,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0, "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "verdict", "strong_null", "per_rho",
                      "pure_target_output_effective_rank", "runtime_s") or k.startswith("pred_")}, indent=2))


if __name__ == "__main__":
    main()
