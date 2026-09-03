#!/usr/bin/env python3
"""Does a NOISE-UNBIASED shared circuit-effect subspace exist at current N? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_material_count
# pred_b_reliable_shared_direction_exists_pooled
# pred_c_reusable_subspace_is_at_least_two_dimensional

Math-review move: §2657 showed per-node fingerprints are cross-half noise, which attenuates every grouping test
to ~0. The correct estimator pools nodes via the cross-half cross-covariance S=(M0^T M1 + M1^T M0)/2; because
the two document halves have INDEPENDENT noise, E[S]=signal covariance (noise-unbiased). Positive eigenvalues of
S are reliable shared circuit-effect directions. Zero forwards. Preregistration:
polynomial_causal/MLP10_SHARED_SUBSPACE_CROSS_HALF_COVARIANCE_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_SHARED_SUBSPACE_CROSS_HALF_COVARIANCE_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R520_RESULT = ROOT / "mlp10_source_star_causal_quotient_rung520_results.json"
OUT = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
HASHES = {
    PREREG: "f75e6aa1792f7fa98c36a03dffdfda8b68d93d3ea5050a5b6820ee2eefd4e0b1",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R520_RESULT: "1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b",
}
MATERIAL_NODES = 83
N_PERM = 200
SEED0 = 5090


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


def eigs_sym(M0, M1):
    S = (M0.T @ M1 + M1.T @ M0) / 2.0
    asym = float(np.max(np.abs(S - S.T)) / (np.max(np.abs(S)) + 1e-30))
    w = np.linalg.eigvalsh(S)[::-1]   # descending
    return S, w, asym


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_shared_subspace_cross_half_covariance_probe",
                          "model_loaded": False}, indent=2))
        return
    check_hashes()
    if OUT.exists():
        raise RuntimeError("output namespace already exists")

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    disc = b["collections"]["discovery"]
    cs = np.asarray(disc["circuit_sums"], dtype=np.float64)    # (4,23,2,2,32)
    cc = np.asarray(disc["circuit_counts"], dtype=np.float64)  # (2,2,32)
    task = np.asarray(disc["task"], dtype=np.float64)          # (4,23,248,6)
    tcnt = np.asarray(disc["task_counts"], dtype=np.float64)   # (248,6)

    # per-half effect (member-minus-control removal), validated reconstruction
    memMean = cs[:, :, :, 0, :] / cc[None, None, :, 0, :]
    ctrlMean = cs[:, :, :, 1, :] / cc[None, None, :, 1, :]
    eff_h = ((memMean - memMean[:, 0:1]) - (ctrlMean - ctrlMean[:, 0:1]))[:, 1:, :, :]  # (4,22,2,32)

    # material mask (pooled) — reproduces published count
    memP = cs[:, :, :, 0, :].sum(2) / cc[:, 0, :].sum(0)[None, None]
    ctrlP = cs[:, :, :, 1, :].sum(2) / cc[:, 1, :].sum(0)[None, None]
    effP = ((memP - memP[:, 0:1]) - (ctrlP - ctrlP[:, 0:1]))[:, 1:, :]
    circ_rms = np.sqrt((effP ** 2).mean(-1))
    tMean = task.sum(2) / tcnt.sum(0)[None, None]
    teffP = (tMean - tMean[:, 0:1])[:, 1:, :]
    task_norm = np.sqrt((teffP[:, :, 1:5] ** 2).sum(-1))
    material = ((circ_rms >= .0005) & (task_norm >= .00025)).reshape(4 * 22)
    n_material = int(material.sum())

    A = eff_h.reshape(4 * 22, 2, 32)[material]                 # (83,2,32)
    M0 = A[:, 0, :].copy(); M1 = A[:, 1, :].copy()
    # mean-centre circuits over nodes
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)

    S, w_real, asym = eigs_sym(M0, M1)
    lam1_real = float(w_real[0])

    # node-permutation null
    null_lam1 = []
    null_top = []
    for k in range(N_PERM):
        rng = np.random.default_rng(SEED0 + k)
        perm = rng.permutation(M1.shape[0])
        _, w, _ = eigs_sym(M0, M1[perm])
        null_lam1.append(float(w[0]))
        null_top.append(float(w[0]))
    lam1_q95 = float(np.quantile(null_lam1, 0.95))
    n_real_above = int(np.sum(w_real > lam1_q95))

    # secondary diagnostic: within-action permutation null (4 actions x up-to-22 sources)
    action_idx = (np.repeat(np.arange(4), 22))[material]
    wa_lam1 = []
    for k in range(N_PERM):
        rng = np.random.default_rng(SEED0 + 1000 + k)
        perm = np.arange(M1.shape[0])
        for a in range(4):
            idx = np.where(action_idx == a)[0]
            perm[idx] = idx[rng.permutation(len(idx))]
        _, w, _ = eigs_sym(M0, M1[perm])
        wa_lam1.append(float(w[0]))
    wa_q95 = float(np.quantile(wa_lam1, 0.95))

    pred_a = bool(n_material == MATERIAL_NODES and asym < 1e-12
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(lam1_real > lam1_q95)
    pred_c = bool(n_real_above >= 2)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "no_reliable_shared_subspace_even_pooled_grouping_absence_is_not_merely_power_raise_N"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "exactly_one_reliable_shared_direction_rank1_reuse_target_that_1d_object"
    elif not strong_null:
        verdict = "reusable_multidim_shared_circuit_effect_subspace_exists_now_seed_gpu_DAS_reuse_rung"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_shared_subspace_cross_half_covariance_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "noise_unbiased_shared_circuit_effect_subspace_dimensionality_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "cross_half_symmetry_rel": asym,
        "lambda1_real": lam1_real,
        "permutation_null_lambda1_q95": lam1_q95,
        "n_real_eigs_above_null_q95": n_real_above,
        "top8_real_eigs": [float(x) for x in w_real[:8]],
        "within_action_null_lambda1_q95": wa_q95,
        "lambda1_beats_within_action_null": bool(lam1_real > wa_q95),
        "term_reference_per_node_rho_2657": 0.016,
        "bars": {"material_target": MATERIAL_NODES, "n_perm": N_PERM, "subspace_dim_bar": 2},
        'pred_a_instrument_reproduces_material_count': pred_a,
        'pred_b_reliable_shared_direction_exists_pooled': pred_b,
        'pred_c_reusable_subspace_is_at_least_two_dimensional': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_material_nodes",
                               "lambda1_real", "permutation_null_lambda1_q95",
                               "n_real_eigs_above_null_q95", "top8_real_eigs",
                               "lambda1_beats_within_action_null", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
