#!/usr/bin/env python3
"""What fraction of MLP10's total reliable causal footprint is the shared 3-dim subspace? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_shared_subspace
# pred_b_shared_subspace_captures_majority_of_reliable_variance
# pred_c_coverage_beats_pure_noise_baseline

Coverage-credit input: fraction of total reliable (positive-eigenvalue) circuit-effect variance captured by the
top-3 shared subspace of the noise-unbiased cross-half cross-covariance, with node-bootstrap CI, vs a pure-noise
baseline. Zero forwards. Preregistration:
polynomial_causal/MLP10_SHARED_SUBSPACE_COVERAGE_FRACTION_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_SHARED_SUBSPACE_COVERAGE_FRACTION_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_shared_subspace_coverage_fraction_probe_results.json"
HASHES = {
    PREREG: "ea2a47c1442b1c3da2af48fc7f8c525b28898ec5993be8ca3caa507aa7421d4a",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
K = 3
LAM1_2658 = 0.009330938687130093
LAM1_TOL = 5e-4
F_BAR = 0.50
F_LO_BAR = 0.40
N_RESAMPLE = 400
SEED0 = 12000


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
    w = np.linalg.eigvalsh(S)          # ascending
    pos = w[w > 0]
    T = float(pos.sum())
    if T <= 0:
        return 0.0, 0.0
    topk = float(np.sort(pos)[::-1][:k].sum())
    return topk / T, T


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_shared_subspace_coverage_fraction_probe",
                          "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    disc = b["collections"]["discovery"]
    cs = np.asarray(disc["circuit_sums"], dtype=np.float64)
    cc = np.asarray(disc["circuit_counts"], dtype=np.float64)
    task = np.asarray(disc["task"], dtype=np.float64)
    tcnt = np.asarray(disc["task_counts"], dtype=np.float64)

    memMean = cs[:, :, :, 0, :] / cc[None, None, :, 0, :]
    ctrlMean = cs[:, :, :, 1, :] / cc[None, None, :, 1, :]
    eff_h = ((memMean - memMean[:, 0:1]) - (ctrlMean - ctrlMean[:, 0:1]))[:, 1:, :, :]
    memP = cs[:, :, :, 0, :].sum(2) / cc[:, 0, :].sum(0)[None, None]
    ctrlP = cs[:, :, :, 1, :].sum(2) / cc[:, 1, :].sum(0)[None, None]
    effP = ((memP - memP[:, 0:1]) - (ctrlP - ctrlP[:, 0:1]))[:, 1:, :]
    circ_rms = np.sqrt((effP ** 2).mean(-1))
    tMean = task.sum(2) / tcnt.sum(0)[None, None]
    teffP = (tMean - tMean[:, 0:1])[:, 1:, :]
    task_norm = np.sqrt((teffP[:, :, 1:5] ** 2).sum(-1))
    material = ((circ_rms >= .0005) & (task_norm >= .00025)).reshape(4 * 22)
    n_material = int(material.sum())

    A = eff_h.reshape(4 * 22, 2, 32)[material]
    M0 = A[:, 0, :].copy(); M1 = A[:, 1, :].copy()
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)
    S = cross_cov(M0, M1)
    lam1 = float(np.linalg.eigvalsh(S)[-1])
    f, T = coverage_fraction(S, K)

    n = M0.shape[0]
    boot_f, null_f = [], []
    for k in range(N_RESAMPLE):
        rng = np.random.default_rng(SEED0 + k)
        idx = rng.integers(0, n, size=n)
        fb, _ = coverage_fraction(cross_cov(M0[idx], M1[idx]), K)
        boot_f.append(fb)
        perm = rng.permutation(n)
        fn, _ = coverage_fraction(cross_cov(M0, M1[perm]), K)
        null_f.append(fn)
    f_lo, f_med, f_hi = [float(np.quantile(boot_f, q)) for q in (0.025, 0.5, 0.975)]
    f_null_q95 = float(np.quantile(null_f, 0.95))

    pred_a = bool(n_material == MATERIAL_NODES and abs(lam1 - LAM1_2658) <= LAM1_TOL
                  and T > 0 and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(f >= F_BAR and f_lo >= F_LO_BAR)
    pred_c = bool(f > f_null_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "shared_subspace_captures_minority_footprint_is_higher_dim"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "coverage_not_distinguishable_from_noise_baseline"
    elif not strong_null:
        verdict = "shared_3dim_captures_majority_of_reliable_variance_mlp10_footprint_is_one_low_dim_summary"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_shared_subspace_coverage_fraction_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "effect_variance_coverage_of_the_shared_subspace_coverage_credit_input_not_certificate_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "lambda1": lam1,
        "total_reliable_variance": T,
        "coverage_fraction_top3": f,
        "coverage_bootstrap_ci95": [f_lo, f_med, f_hi],
        "coverage_pure_noise_baseline_q95": f_null_q95,
        "bars": {"material_target": MATERIAL_NODES, "k": K, "f_bar": F_BAR, "f_lo_bar": F_LO_BAR,
                 "n_resample": N_RESAMPLE},
        'pred_a_instrument_reproduces_shared_subspace': pred_a,
        'pred_b_shared_subspace_captures_majority_of_reliable_variance': pred_b,
        'pred_c_coverage_beats_pure_noise_baseline': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "coverage_fraction_top3",
                               "coverage_bootstrap_ci95", "coverage_pure_noise_baseline_q95",
                               "total_reliable_variance", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
