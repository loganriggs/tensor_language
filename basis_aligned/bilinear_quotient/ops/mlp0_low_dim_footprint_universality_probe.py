#!/usr/bin/env python3
"""Does MLP10's low-dim reliable-footprint finding generalize to MLP0? (CPU probe, "look elsewhere")

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_r519_whole_effect
# pred_b_mlp0_has_a_reliable_shared_direction
# pred_c_mlp0_reliable_footprint_is_low_dim_majority_coverage

Applies the §2658/§2666 noise-unbiased cross-half instrument to MLP0's 49 interaction-term effects (R519).
Universality check of the low-dim law on a second module. CAVEAT: MLP0 object = 49 terms of one source; MLP10
object = 22 source-stars; comparison is indicative, not apples-to-apples. Zero forwards. Preregistration:
polynomial_causal/MLP0_LOW_DIM_FOOTPRINT_UNIVERSALITY_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_LOW_DIM_FOOTPRINT_UNIVERSALITY_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp0_one_circuit_interaction_atlas_rung519_bundle.pt"
OUT = ROOT / "mlp0_low_dim_footprint_universality_probe_results.json"
HASHES = {
    PREREG: "a46e06b0adbbd4db49098346aa526691e9d130a035567682eaf18bebf53b627f",
    BUNDLE: "54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a",
}
TARGET_IDX = 8
TARGET_W0 = 0.003909140586171755
TARGET_W1 = 0.004190039411971824
K = 3
F_BAR = 0.50
N_RESAMPLE = 400
SEED0 = 13000
# MLP10 reference (§2666) for side-by-side reporting only
MLP10_F = 0.764
MLP10_FLOOR = 0.583


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


def cross_cov(M0, M1):
    return (M0.T @ M1 + M1.T @ M0) / 2.0


def coverage_fraction(S, k):
    w = np.linalg.eigvalsh(S)
    pos = w[w > 0]
    T = float(pos.sum())
    if T <= 0:
        return 0.0, 0.0, float(w[-1])
    topk = float(np.sort(pos)[::-1][:k].sum())
    return topk / T, T, float(w[-1])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp0_low_dim_footprint_universality_probe",
                          "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    de = b["discovery_effects"]
    A = np.asarray(de["circuit"], dtype=np.float64)          # (49,2,32)
    WC = np.asarray(de["whole_circuit"], dtype=np.float64)    # (2,32)
    shape_ok = (A.shape == (49, 2, 32))
    target_ok = (abs(WC[0, TARGET_IDX] - TARGET_W0) < 1e-6
                 and abs(WC[1, TARGET_IDX] - TARGET_W1) < 1e-6)

    M0 = A[:, 0, :].copy(); M1 = A[:, 1, :].copy()
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)
    S = cross_cov(M0, M1)
    f, T, lam1 = coverage_fraction(S, K)

    n = M0.shape[0]
    boot_f, null_f, null_lam1 = [], [], []
    for k in range(N_RESAMPLE):
        rng = np.random.default_rng(SEED0 + k)
        idx = rng.integers(0, n, size=n)
        fb, _, _ = coverage_fraction(cross_cov(M0[idx], M1[idx]), K)
        boot_f.append(fb)
        perm = rng.permutation(n)
        Sp = cross_cov(M0, M1[perm])
        fn, _, ln = coverage_fraction(Sp, K)
        null_f.append(fn); null_lam1.append(ln)
    f_lo, f_med, f_hi = [float(np.quantile(boot_f, q)) for q in (0.025, 0.5, 0.975)]
    f_null_q95 = float(np.quantile(null_f, 0.95))
    lam1_null_q95 = float(np.quantile(null_lam1, 0.95))

    # dimensionality: eigenvalues beating the term-permutation top-eig null
    w = np.linalg.eigvalsh(S)[::-1]
    n_above_null = int(np.sum(w > lam1_null_q95))

    pred_a = bool(shape_ok and target_ok and T > 0 and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(lam1 > lam1_null_q95)
    pred_c = bool(f >= F_BAR and f > f_null_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "mlp0_pooled_terms_no_reliable_shared_direction_law_does_not_extend_here"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "mlp0_reliable_but_higher_dim_than_mlp10"
    elif not strong_null:
        verdict = "low_dim_reliable_footprint_law_generalizes_to_mlp0"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp0_low_dim_footprint_universality_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "universality_of_low_dim_reliable_footprint_indicative_cross_module_not_apples_to_apples",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "module": "MLP0_R519_49_terms_of_one_source",
        "lambda1": lam1,
        "lambda1_null_q95": lam1_null_q95,
        "n_eigs_above_null": n_above_null,
        "total_reliable_variance": T,
        "coverage_fraction_top3": f,
        "coverage_bootstrap_ci95": [f_lo, f_med, f_hi],
        "coverage_pure_noise_baseline_q95": f_null_q95,
        "mlp10_reference_2666": {"coverage": MLP10_F, "noise_floor": MLP10_FLOOR,
                                 "note": "indicative only; different decomposition (22 source-stars vs 49 terms)"},
        "bars": {"k": K, "f_bar": F_BAR, "n_resample": N_RESAMPLE},
        'pred_a_instrument_reproduces_r519_whole_effect': pred_a,
        'pred_b_mlp0_has_a_reliable_shared_direction': pred_b,
        'pred_c_mlp0_reliable_footprint_is_low_dim_majority_coverage': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "lambda1", "lambda1_null_q95",
                               "n_eigs_above_null", "coverage_fraction_top3", "coverage_bootstrap_ci95",
                               "coverage_pure_noise_baseline_q95", "mlp10_reference_2666", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
