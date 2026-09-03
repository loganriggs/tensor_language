#!/usr/bin/env python3
"""Reliable circuit-effect structure BEYOND the §2658 shared 3-dim, and is it private? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_shared_subspace
# pred_b_reliable_residual_structure_beyond_shared
# pred_c_residual_structure_is_source_specific_private

Red-team PREVIEW of Codex rung521's shared-first/private-residual DAS, in circuit-effect space: after removing
the §2658 shared 3-dim, is there reliable residual structure (pred_b) and is it source-specific/private
(pred_c)? Zero forwards. Preregistration:
polynomial_causal/MLP10_EFFECT_SPACE_SHARED_PRIVATE_RESIDUAL_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_EFFECT_SPACE_SHARED_PRIVATE_RESIDUAL_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_effect_space_shared_private_residual_probe_results.json"
HASHES = {
    PREREG: "06721800c2ef300451a5b3d27f9f5b7341aafcf7484f6f4441cecf808d3c3854",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
K_SHARED = 3
LAM1_2658 = 0.009330938687130093
LAM1_TOL = 5e-4
N_PERM = 200
SEED0 = 7000


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


def top_eig(S):
    return float(np.linalg.eigvalsh(S)[-1])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_effect_space_shared_private_residual_probe",
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
    source_names = [str(x) for x in disc["arms"][1:]]   # 22 source names

    memMean = cs[:, :, :, 0, :] / cc[None, None, :, 0, :]
    ctrlMean = cs[:, :, :, 1, :] / cc[None, None, :, 1, :]
    eff_h = ((memMean - memMean[:, 0:1]) - (ctrlMean - ctrlMean[:, 0:1]))[:, 1:, :, :]  # (4,22,2,32)
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
    src_idx = np.tile(np.arange(22), 4)[material]        # source index per material node
    act_idx = np.repeat(np.arange(4), 22)[material]
    M0 = A[:, 0, :].copy(); M1 = A[:, 1, :].copy()
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)

    # shared subspace from noise-unbiased cross-cov
    S = cross_cov(M0, M1)
    evals, evecs = np.linalg.eigh(S)                     # ascending
    lam1_shared = float(evals[-1])
    U = evecs[:, -K_SHARED:]                             # (32,3) top-3
    UUt = U @ U.T
    P = np.eye(32) - UUt

    M0r = M0 @ P; M1r = M1 @ P
    proj_leak = float(np.max(np.abs(M0r @ U)))
    Sr = cross_cov(M0r, M1r)
    lam1_r = top_eig(Sr)

    # nulls on the projected residuals
    node_null, wa_null = [], []
    for k in range(N_PERM):
        rng = np.random.default_rng(SEED0 + k)
        node_null.append(top_eig(cross_cov(M0r, M1r[rng.permutation(M0r.shape[0])])))
        perm = np.arange(M0r.shape[0])
        for a in range(4):
            idx = np.where(act_idx == a)[0]
            perm[idx] = idx[rng.permutation(len(idx))]
        wa_null.append(top_eig(cross_cov(M0r, M1r[perm])))
    node_q95 = float(np.quantile(node_null, 0.95))
    wa_q95 = float(np.quantile(wa_null, 0.95))

    # per-source energy on the residual top mode (for the priority list if positive)
    vr = np.linalg.eigh(Sr)[1][:, -1]                    # residual top eigenvector (32,)
    node_load = (0.5 * (M0r + M1r)) @ vr                 # (83,) projection per node
    per_source = {}
    for s in range(22):
        e = node_load[src_idx == s]
        per_source[source_names[s]] = float(np.sqrt((e ** 2).mean())) if len(e) else 0.0
    top_sources = sorted(per_source.items(), key=lambda kv: -kv[1])[:6]

    pred_a = bool(n_material == MATERIAL_NODES
                  and abs(lam1_shared - LAM1_2658) <= LAM1_TOL
                  and float(np.max(np.abs(U.T @ U - np.eye(K_SHARED)))) < 1e-10
                  and proj_leak < 1e-10
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(lam1_r > node_q95)
    pred_c = bool(lam1_r > wa_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "shared_3dim_is_all_reliable_structure_private_stage_has_no_effect_space_target"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "residual_reliable_but_still_source_shared_not_private"
    elif not strong_null:
        verdict = "reliable_source_specific_private_structure_exists_priority_sources_reported"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_effect_space_shared_private_residual_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "effect_space_residual_reliability_preview_of_rung521_private_stage_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "k_shared": K_SHARED,
        "lambda1_shared": lam1_shared,
        "projection_leak_inf": proj_leak,
        "lambda1_residual": lam1_r,
        "residual_node_null_q95": node_q95,
        "residual_within_action_null_q95": wa_q95,
        "residual_top_mode_per_source_rms": per_source,
        "residual_top_sources": top_sources,
        "bars": {"material_target": MATERIAL_NODES, "k_shared": K_SHARED, "n_perm": N_PERM},
        'pred_a_instrument_reproduces_shared_subspace': pred_a,
        'pred_b_reliable_residual_structure_beyond_shared': pred_b,
        'pred_c_residual_structure_is_source_specific_private': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_material_nodes",
                               "lambda1_residual", "residual_node_null_q95",
                               "residual_within_action_null_q95", "residual_top_sources", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
