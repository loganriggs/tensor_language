#!/usr/bin/env python3
"""Prequential/MDL cross-validated rank of MLP10's effect matrix (noise-robust coverage credit) (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_effect_matrix
# pred_b_mdl_optimal_rank_is_low
# pred_c_low_rank_saves_bits_and_majority_heldout_coverage

Replaces §2666's soft variance fraction with a prequential/MDL number: fit the top-r circuit subspace on half0,
code half1 with it (held-out captured energy has no noise-floor inflation); BIC selects the effective rank and
gives bits saved. Red-teams §2666. Zero forwards. Preregistration:
polynomial_causal/MLP10_PREQUENTIAL_MDL_RANK_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_PREQUENTIAL_MDL_RANK_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2666 = ROOT / "mlp10_shared_subspace_coverage_fraction_probe_results.json"
OUT = ROOT / "mlp10_prequential_mdl_rank_probe_results.json"
HASHES = {
    PREREG: "3516aecf0e87425493265fc847c47040fc983551dff4be092acfc55e63fa066f",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2666: "9d2cdc37719e3554fc2b375c5f6e27b50d0fb24ca074a0315a35ddc4b4d5172f",
}
MATERIAL_NODES = 83
R_MAX = 12
R_STAR_BAR = 6


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


def right_subspace(M, r):
    if r == 0:
        return np.zeros((M.shape[1], 0))
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:r].T                     # (32, r)


def rss_heldout(train, test, r):
    V = right_subspace(train, r)
    P = V @ V.T if r > 0 else np.zeros((train.shape[1], train.shape[1]))
    resid = test - test @ P
    return float((resid ** 2).sum())


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_prequential_mdl_rank_probe",
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
    normM1 = float((M1 ** 2).sum()); normM0 = float((M0 ** 2).sum())
    n = M0.shape[0] * M0.shape[1]        # 83*32

    curve = []
    for r in range(0, R_MAX + 1):
        rss1 = rss_heldout(M0, M1, r)    # fit half0, test half1
        rss0 = rss_heldout(M1, M0, r)    # symmetric
        rss = 0.5 * (rss1 + rss0)
        g1 = 1 - rss1 / normM1           # cross-validated captured fraction (half1)
        g = 1 - rss / (0.5 * (normM0 + normM1))
        params = r * 32
        dl = (n / 2) * np.log(max(rss / n, 1e-300)) + (params / 2) * np.log(n)
        curve.append({"r": r, "rss_heldout": rss, "cv_captured_fraction": float(g),
                      "cv_captured_fraction_half1": float(g1), "DL_nats": float(dl)})

    dls = np.array([c["DL_nats"] for c in curve])
    r_star = int(np.argmin(dls))
    dl0 = curve[0]["DL_nats"]
    bits_saved = float((dl0 - curve[r_star]["DL_nats"]) / np.log(2))
    g3 = curve[3]["cv_captured_fraction"]

    pred_a = bool(n_material == MATERIAL_NODES and normM1 > 0 and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(1 <= r_star <= R_STAR_BAR)
    pred_c = bool(curve[r_star]["DL_nats"] < dl0 and g3 >= 0.50)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "effect_matrix_not_low_rank_by_prequential_mdl_2658_2666_red_teamed"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "low_rank_but_heldout_coverage_minority_2666_was_noise_floor_inflated"
    elif not strong_null:
        verdict = "effect_matrix_genuinely_low_rank_mdl_optimal_r%d_bits_saved_%.0f" % (r_star, bits_saved)
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_prequential_mdl_rank_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "prequential_mdl_effective_rank_and_bits_saved_coverage_credit_not_certificate",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "mdl_optimal_rank": r_star,
        "bits_saved_vs_mean_only": bits_saved,
        "cv_captured_fraction_at_r3": g3,
        "cv_captured_fraction_at_rstar": curve[r_star]["cv_captured_fraction"],
        "coverage_2666_soft_reference": 0.764,
        "dl_curve": curve,
        "bars": {"material_target": MATERIAL_NODES, "r_max": R_MAX, "r_star_bar": R_STAR_BAR},
        'pred_a_instrument_reproduces_effect_matrix': pred_a,
        'pred_b_mdl_optimal_rank_is_low': pred_b,
        'pred_c_low_rank_saves_bits_and_majority_heldout_coverage': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "mdl_optimal_rank",
                               "bits_saved_vs_mean_only", "cv_captured_fraction_at_r3",
                               "cv_captured_fraction_at_rstar", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
