#!/usr/bin/env python3
"""WHAT does the §2658 shared 3-dim subspace compute — which circuits, reproducibly? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_shared_subspace
# pred_b_top_mode_names_reproducible_circuit_combination
# pred_c_full_3dim_subspace_reproducible_as_basis

First CONSTRUCTIVE characterization of the recent arc's positive object: names the downstream circuits the
source-shared MLP10 summary feeds (eigenvector loadings of the noise-unbiased cross-half cross-covariance) and
node-bootstraps their reproducibility. Hands Codex a labeled target for rung521's shared stage. Zero forwards.
Preregistration: polynomial_causal/MLP10_SHARED_SUBSPACE_CIRCUIT_CONTENT_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_SHARED_SUBSPACE_CIRCUIT_CONTENT_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_shared_subspace_circuit_content_probe_results.json"
HASHES = {
    PREREG: "100304512ad179edcdb26117c1bf1bea9a08e62bb5828e907e083192cf5e8081",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
K = 3
LAM1_2658 = 0.009330938687130093
LAM1_TOL = 5e-4
N_BOOT = 200
N_NULL = 200
STAB_BAR = 0.70
SEED0 = 8000


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


def topk_evecs(S, k):
    w, V = np.linalg.eigh(S)          # ascending
    return w[-1], V[:, -k:]


def subspace_cos(A, B):
    # mean of singular values of A^T B (both orthonormal columns) = mean principal-angle cosine
    return float(np.mean(np.linalg.svd(A.T @ B, compute_uv=False)))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_shared_subspace_circuit_content_probe",
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
    tags = [str(x) for x in disc["circuit_tags"]]

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
    lam1, V3 = topk_evecs(S, K)
    v1 = V3[:, -1]
    ortho = float(np.max(np.abs(V3.T @ V3 - np.eye(K))))

    n = M0.shape[0]
    boot_load, boot_sub = [], []
    for k in range(N_BOOT):
        rng = np.random.default_rng(SEED0 + k)
        idx = rng.integers(0, n, size=n)   # with replacement
        Sb = cross_cov(M0[idx], M1[idx])
        _, V3b = topk_evecs(Sb, K)
        v1b = V3b[:, -1]
        boot_load.append(abs(float(v1b @ v1)))
        boot_sub.append(subspace_cos(V3b, V3))
    load_stab = float(np.median(boot_load))
    sub_stab = float(np.median(boot_sub))

    # nulls
    null_load, null_sub = [], []
    for k in range(N_NULL):
        rng = np.random.default_rng(SEED0 + 5000 + k)
        r = rng.standard_normal(32); r /= np.linalg.norm(r)
        null_load.append(abs(float(r @ v1)))
        G = rng.standard_normal((32, K))
        Q, _ = np.linalg.qr(G)
        null_sub.append(subspace_cos(Q, V3))
    load_null_q95 = float(np.quantile(null_load, 0.95))
    sub_null_q95 = float(np.quantile(null_sub, 0.95))

    # circuit content: per-mode top circuits, and total subspace energy per circuit
    def top_circuits(vec, m=6):
        order = np.argsort(-np.abs(vec))[:m]
        return [[tags[i], float(vec[i])] for i in order]
    modes = {f"mode{j+1}": {"eigenvalue": float(np.linalg.eigvalsh(S)[-(j+1)]),
                            "top_circuits": top_circuits(V3[:, -(j+1)])} for j in range(K)}
    subspace_energy = (V3 ** 2).sum(1)   # per-circuit energy across the 3-dim
    order = np.argsort(-subspace_energy)
    subspace_top = [[tags[i], float(subspace_energy[i])] for i in order[:8]]

    pred_a = bool(n_material == MATERIAL_NODES and abs(lam1 - LAM1_2658) <= LAM1_TOL
                  and ortho < 1e-10 and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(load_stab >= STAB_BAR and load_stab > load_null_q95)
    pred_c = bool(sub_stab >= STAB_BAR and sub_stab > sub_null_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "shared_subspace_stable_but_axes_unnamed_report_subspace_energy_only"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "top_mode_named_but_modes_2_3_rotate_report_top_mode_and_subspace_energy"
    elif not strong_null:
        verdict = "shared_summary_is_reproducible_named_3dim_object_labeled_target_for_rung521_shared_stage"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_shared_subspace_circuit_content_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "circuit_content_and_reproducibility_of_the_shared_subspace_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "lambda1_shared": lam1,
        "top_mode_loading_stability": load_stab,
        "loading_null_q95": load_null_q95,
        "subspace_stability": sub_stab,
        "subspace_null_q95": sub_null_q95,
        "modes": modes,
        "subspace_energy_top_circuits": subspace_top,
        "bars": {"material_target": MATERIAL_NODES, "k": K, "stab_bar": STAB_BAR,
                 "n_boot": N_BOOT, "n_null": N_NULL},
        'pred_a_instrument_reproduces_shared_subspace': pred_a,
        'pred_b_top_mode_names_reproducible_circuit_combination': pred_b,
        'pred_c_full_3dim_subspace_reproducible_as_basis': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_material_nodes",
                               "top_mode_loading_stability", "loading_null_q95",
                               "subspace_stability", "subspace_null_q95",
                               "subspace_energy_top_circuits", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
