#!/usr/bin/env python3
"""Is §2661's "feeds BLOCK-6" real enrichment or base rate? (CPU self-red-team)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_shared_subspace_and_block6_count
# pred_b_block6_energy_enriched_above_base_rate
# pred_c_loaded_block6_circuits_are_coherent

Red-teams §2661's headline (both lanes anchor on it). Block-6 is 12/32 of the panel, so "feeds block-6" could
be base rate. Tests block-6 subspace energy vs a random-12-subset null (pred_b) and block-6 within-set
correlation coherence vs the same null (pred_c). Zero forwards. Preregistration:
polynomial_causal/MLP10_BLOCK6_ENRICHMENT_COHERENCE_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_BLOCK6_ENRICHMENT_COHERENCE_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_block6_enrichment_coherence_probe_results.json"
HASHES = {
    PREREG: "bfdd572433d43b1363dfb1505cbd8e2338bd8a4f45a5711b49b82b657f84d55f",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
K = 3
LAM1_2658 = 0.009330938687130093
LAM1_TOL = 5e-4
N_BLOCK6 = 12
N_SUBSET = 2000
SEED0 = 11000


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


def mean_abs_offdiag_corr(C, idx):
    sub = C[np.ix_(idx, idx)]
    d = np.sqrt(np.clip(np.diag(sub), 1e-30, None))
    R = sub / np.outer(d, d)
    m = len(idx)
    off = R[~np.eye(m, dtype=bool)]
    return float(np.mean(np.abs(off)))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_block6_enrichment_coherence_probe",
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
    w, V = np.linalg.eigh(S)
    lam1 = float(w[-1]); V3 = V[:, -K:]
    ortho = float(np.max(np.abs(V3.T @ V3 - np.eye(K))))

    energy = (V3 ** 2).sum(1)               # (32,), sums to K=3
    block6 = np.array([i for i, t in enumerate(tags) if t.startswith("r.6.")])
    n_b6 = int(len(block6))
    base_rate_energy = float(K * n_b6 / 32)
    E_B = float(energy[block6].sum())
    coh_B = mean_abs_offdiag_corr(S, block6)

    # random-12-subset null
    rng = np.random.default_rng(SEED0)
    null_E, null_coh = [], []
    for _ in range(N_SUBSET):
        idx = rng.choice(32, size=n_b6, replace=False)
        null_E.append(float(energy[idx].sum()))
        null_coh.append(mean_abs_offdiag_corr(S, idx))
    E_q95 = float(np.quantile(null_E, 0.95))
    coh_q95 = float(np.quantile(null_coh, 0.95))
    E_percentile = float((np.asarray(null_E) < E_B).mean())
    coh_percentile = float((np.asarray(null_coh) < coh_B).mean())

    # per-circuit enrichment (single-circuit view): top energy circuits with block flag
    order = np.argsort(-energy)
    top_energy = [[tags[i], float(energy[i]), tags[i].startswith("r.6.")] for i in order[:8]]

    pred_a = bool(n_material == MATERIAL_NODES and abs(lam1 - LAM1_2658) <= LAM1_TOL
                  and n_b6 == N_BLOCK6 and ortho < 1e-10 and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(E_B > E_q95)
    pred_c = bool(coh_B > coh_q95)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "block6_is_BASE_RATE_not_enrichment_qualify_2661_headline"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "block6_enriched_but_not_internally_coherent"
    elif not strong_null:
        verdict = "block6_genuinely_enriched_and_coherent_2661_headline_stands"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_block6_enrichment_coherence_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "enrichment_and_coherence_red_team_of_2661_block6_claim_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "n_block6_circuits": n_b6,
        "block6_subspace_energy": E_B,
        "base_rate_energy": base_rate_energy,
        "block6_energy_null_q95": E_q95,
        "block6_energy_percentile": E_percentile,
        "block6_coherence": coh_B,
        "block6_coherence_null_q95": coh_q95,
        "block6_coherence_percentile": coh_percentile,
        "top_energy_circuits": top_energy,
        "bars": {"material_target": MATERIAL_NODES, "k": K, "n_block6": N_BLOCK6, "n_subset": N_SUBSET},
        'pred_a_instrument_reproduces_shared_subspace_and_block6_count': pred_a,
        'pred_b_block6_energy_enriched_above_base_rate': pred_b,
        'pred_c_loaded_block6_circuits_are_coherent': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "block6_subspace_energy",
                               "base_rate_energy", "block6_energy_null_q95", "block6_energy_percentile",
                               "block6_coherence", "block6_coherence_null_q95", "top_energy_circuits",
                               "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
