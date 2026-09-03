#!/usr/bin/env python3
"""Is the block-6 shared summary REUSED across the four score implementations? (CPU probe, leave-one-action-out)

# BQGATE: EXPERIMENT
# pred_a_instrument_reproduces_shared_subspace_and_per_action_counts
# pred_b_block6_direction_transfers_to_every_action
# pred_c_shared_subspace_explains_majority_of_each_held_out_action

Leave-one-action-out transfer test of the §2658/§2661 shared subspace: learn the shared direction from three
score implementations (~62 nodes, above §2659's m*=16 floor) and test whether it carries reliable signal in the
held-out fourth. Answers Logan's compositional-reuse question in effect space. Zero forwards. Preregistration:
polynomial_causal/MLP10_CROSS_ACTION_SUBSPACE_REUSE_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP10_CROSS_ACTION_SUBSPACE_REUSE_PROBE_PREREGISTRATION.md"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"
R2658 = ROOT / "mlp10_shared_subspace_cross_half_covariance_probe_results.json"
OUT = ROOT / "mlp10_cross_action_subspace_reuse_probe_results.json"
HASHES = {
    PREREG: "2daad3155a6d838b8e5aa07b01cf4609974e6952fa46994ec3ad87fef739a52c",
    BUNDLE: "7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06",
    R2658: "1e8ade7c9acea5cbc83c1de511aa0b5bad323fea4b681a05dc450bdc6431120b",
}
MATERIAL_NODES = 83
PER_ACTION = [21, 20, 21, 21]
K = 3
LAM1_2658 = 0.009330938687130093
LAM1_TOL = 5e-4
FRAC_BAR = 0.50
N_PERM = 200
SEED0 = 9000


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


def pos_eig_sum(S):
    w = np.linalg.eigvalsh(S)
    return float(w[w > 0].sum())


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mlp10_cross_action_subspace_reuse_probe",
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
    material_grid = (circ_rms >= .0005) & (task_norm >= .00025)   # (4,22)
    per_action = [int(material_grid[a].sum()) for a in range(4)]
    n_material = int(material_grid.sum())

    # node arrays with action labels
    eff = eff_h                                    # (4,22,2,32)
    act_of, M0_list, M1_list = [], [], []
    for a in range(4):
        for s in range(22):
            if material_grid[a, s]:
                act_of.append(a); M0_list.append(eff[a, s, 0]); M1_list.append(eff[a, s, 1])
    act_of = np.array(act_of)
    M0 = np.array(M0_list); M1 = np.array(M1_list)     # (83,32)
    M0 -= M0.mean(0, keepdims=True); M1 -= M1.mean(0, keepdims=True)

    # pooled reference
    Spool = cross_cov(M0, M1)
    wp, Vp = np.linalg.eigh(Spool)
    pooled_lam1 = float(wp[-1]); pooled_v1 = Vp[:, -1]

    results = []
    all_b = True; all_c = True; ortho_ok = True
    for a in range(4):
        tr = act_of != a; te = act_of == a
        S_tr = cross_cov(M0[tr], M1[tr])
        w_tr, V_tr = np.linalg.eigh(S_tr)
        v1_tr = V_tr[:, -1]; V3_tr = V_tr[:, -K:]
        if float(np.max(np.abs(V3_tr.T @ V3_tr - np.eye(K)))) >= 1e-10:
            ortho_ok = False
        M0a, M1a = M0[te], M1[te]
        S_a = cross_cov(M0a, M1a)
        t1 = float(v1_tr @ S_a @ v1_tr)
        cap = float(np.clip(np.trace(V3_tr.T @ S_a @ V3_tr), 0, None))
        denom = pos_eig_sum(S_a)
        frac = cap / denom if denom > 0 else 0.0
        # null: permute held-out action half-1 pairing
        t1n, capn = [], []
        for k in range(N_PERM):
            rng = np.random.default_rng(SEED0 + a * 1000 + k)
            perm = rng.permutation(M1a.shape[0])
            S_perm = cross_cov(M0a, M1a[perm])
            t1n.append(float(v1_tr @ S_perm @ v1_tr))
            capn.append(float(np.clip(np.trace(V3_tr.T @ S_perm @ V3_tr), 0, None)))
        t1_q95 = float(np.quantile(t1n, 0.95)); cap_q95 = float(np.quantile(capn, 0.95))
        b_pass = bool(t1 > t1_q95)
        c_pass = bool(frac >= FRAC_BAR and cap > cap_q95)
        all_b = all_b and b_pass; all_c = all_c and c_pass
        results.append({"held_out_action": a, "n_test": int(te.sum()), "n_train": int(tr.sum()),
                        "t1": t1, "t1_null_q95": t1_q95, "b_pass": b_pass,
                        "captured_fraction": frac, "captured_trace": cap, "cap_null_q95": cap_q95,
                        "c_pass": c_pass, "cos_v1_pooled": abs(float(v1_tr @ pooled_v1))})

    pred_a = bool(n_material == MATERIAL_NODES and per_action == PER_ACTION
                  and abs(pooled_lam1 - LAM1_2658) <= LAM1_TOL and ortho_ok
                  and sha256(BUNDLE) == HASHES[BUNDLE])
    pred_b = bool(all_b)
    pred_c = bool(all_c)
    strong_null = bool(not (pred_a and pred_b and pred_c))

    if strong_null and pred_a and not pred_b:
        verdict = "shared_summary_does_not_transfer_to_every_score_implementation_partial_reuse"
    elif strong_null and pred_a and pred_b and not pred_c:
        verdict = "top_direction_reuses_across_actions_but_full_3dim_does_not"
    elif not strong_null:
        verdict = "block6_shared_summary_is_reused_across_all_four_score_implementations_supports_rung521_shared_projector"
    else:
        verdict = "instrument_invalid"

    result = {
        "status": "complete",
        "rung": "mlp10_cross_action_subspace_reuse_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "leave_one_action_out_transfer_of_the_shared_subspace_not_grouping_or_compression",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "n_material_nodes": n_material,
        "per_action_material": per_action,
        "pooled_lambda1": pooled_lam1,
        "per_held_out_action": results,
        "min_cos_v1_pooled": float(min(r["cos_v1_pooled"] for r in results)),
        "bars": {"material_target": MATERIAL_NODES, "per_action": PER_ACTION, "k": K,
                 "frac_bar": FRAC_BAR, "n_perm": N_PERM},
        'pred_a_instrument_reproduces_shared_subspace_and_per_action_counts': pred_a,
        'pred_b_block6_direction_transfers_to_every_action': pred_b,
        'pred_c_shared_subspace_explains_majority_of_each_held_out_action': pred_c,
        "strong_null": strong_null,
        "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "backwards": 0,
                            "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "per_action_material",
                               "per_held_out_action", "min_cos_v1_pooled", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
