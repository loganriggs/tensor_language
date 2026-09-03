#!/usr/bin/env python3
"""The document/pooling budget that turns the §2657 null into a testable object (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_rho0_and_material
# pred_b_single_node_order_of_magnitude_underpowered
# pred_c_pooling_is_the_cheaper_lever_and_explains_2658

Math-review move #2. Spearman-Brown converts §2657's per-node reliability rho0=0.016 into a document multiplier;
node subsampling confirms §2658's reliable subspace is a pooling effect and yields the node budget. Two levers
for Codex: raise per-node documents ~k(0.5)x, or pool >= m* nodes. Zero forwards. Preregistration:
polynomial_causal/MLP10_REQUIRED_N_AND_POOLING_BUDGET_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_REQUIRED_N_AND_POOLING_BUDGET_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2657 = ROOT / "mlp10_source_star_cross_half_stability_probe_results.json"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_required_n_and_pooling_budget_probe_results.json"
HASHES = {
    PREREG: "00d3742aff03bf12288eabd665c4bd1840af8b29adc10a5ec97679956926d9dd",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2657: "1bdb425e4da3f85e8da31e701b56dbf51191654f10c3a52c5f58108992532b0d",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
RHO0_REF = 0.016
RHO0_TOL = 0.003
SB_MULT_FLOOR_AT_HALF = 10.0
SUBSET_SIZES = (8, 16, 32, 64, 83)
N_DRAWS = 40
N_PERM = 100
SEED0 = 6000


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


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def sb_multiplier(rho0, rstar):
    return float(rstar * (1 - rho0) / (rho0 * (1 - rstar)))


def lam1_and_null(M0s, M1s, rng):
    def top(A, B):
        S = (A.T @ B + B.T @ A) / 2.0
        return float(np.linalg.eigvalsh(S)[-1])
    lam1 = top(M0s, M1s)
    nulls = [top(M0s, M1s[rng.permutation(M1s.shape[0])]) for _ in range(N_PERM)]
    return lam1, float(np.quantile(nulls, 0.95))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_required_n_and_pooling_budget_probe",
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

    A = eff_h.reshape(4 * 22, 2, 32)[material]        # (83,2,32)
    M0 = A[:, 0, :].copy(); M1 = A[:, 1, :].copy()
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)

    rho0 = np.array([corr(A[i, 0], A[i, 1]) for i in range(A.shape[0])])
    rho0_med = float(np.median(rho0))

    # Spearman-Brown document multipliers (use the positive median as the operative reliability)
    rho0_op = max(rho0_med, 1e-4)
    sb = {f"rho_star_{rs}": sb_multiplier(rho0_op, rs) for rs in (0.3, 0.5, 0.8)}

    # pooling curve: mean lambda1 vs mean null q95 over subsamples
    curve = []
    m_star = None
    for m in SUBSET_SIZES:
        lams, nq95s = [], []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(SEED0 + m * 1000 + d)
            idx = rng.choice(M0.shape[0], size=min(m, M0.shape[0]), replace=False)
            lam1, nq95 = lam1_and_null(M0[idx], M1[idx], rng)
            lams.append(lam1); nq95s.append(nq95)
        ml, mn = float(np.mean(lams)), float(np.mean(nq95s))
        ratio = ml / mn if mn > 0 else float("inf")
        detectable = bool(ml > mn)
        curve.append({"m": m, "mean_lambda1": ml, "mean_null_q95": mn,
                      "ratio": ratio, "detectable": detectable})
        if detectable and m_star is None:
            m_star = m
    ratio_grows = bool(curve[-1]["ratio"] > curve[0]["ratio"])

    pred_a = bool(n_material == MATERIAL_NODES and abs(rho0_med - RHO0_REF) <= RHO0_TOL
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(sb["rho_star_0.5"] >= SB_MULT_FLOOR_AT_HALF)
    pred_c = bool(m_star is not None and m_star <= MATERIAL_NODES
                  and curve[-1]["detectable"] and ratio_grows)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "single_node_only_mildly_underpowered_small_document_bump_suffices"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "pooling_model_inconsistent_with_2658_required_N_untrustworthy_flag_framework"
    elif not strong_null:
        verdict = "two_levers_delivered_raise_docs_k05x_or_pool_at_least_mstar_nodes"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_required_n_and_pooling_budget_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "estimation_budget_document_multiplier_and_node_pooling_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "rho0_median": rho0_med,
        "rho0_quartiles": [float(np.quantile(rho0, q)) for q in (0.25, 0.5, 0.75)],
        "spearman_brown_document_multipliers": sb,
        "pooling_curve": curve,
        "m_star_min_nodes_for_detectability": m_star,
        "pooled_rho_prediction_at_83": float(83 * rho0_op / (1 + 82 * rho0_op)),
        "bars": {"material_target": MATERIAL_NODES, "rho0_ref": RHO0_REF,
                 "sb_mult_floor_at_half": SB_MULT_FLOOR_AT_HALF, "subset_sizes": list(SUBSET_SIZES)},
        'pred_a_instrument_reproduces_rho0_and_material': pred_a,
        'pred_b_single_node_order_of_magnitude_underpowered': pred_b,
        'pred_c_pooling_is_the_cheaper_lever_and_explains_2658': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "n_material_nodes", "rho0_median",
                               "spearman_brown_document_multipliers", "m_star_min_nodes_for_detectability",
                               "pooling_curve", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
