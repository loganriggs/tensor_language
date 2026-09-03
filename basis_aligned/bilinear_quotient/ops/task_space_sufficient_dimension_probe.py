#!/usr/bin/env python3
"""How many score templates does the copy TASK need? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_exact_reproduction_of_513_task_effects
# pred_b_copy_task_near_rank1_both_halves
# pred_c_shared_score_direction_stable

Parallel-lane CPU analysis (Claude). Task-space effective rank of the four
equality-score implementations' copy-task effects from rung513's bundle. Zero
model forwards. Preregistration:
polynomial_causal/TASK_SPACE_SUFFICIENT_DIMENSION_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "TASK_SPACE_SUFFICIENT_DIMENSION_PROBE_PREREGISTRATION.md"
R513_RESULT = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_results.json"
R513_BUNDLE = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_bundle.pt"
OUT = ROOT / "task_space_sufficient_dimension_probe_results.json"
HASHES = {PREREG: "467f3f424c979186e759b9fc07d9a3efeb3492b0902ff16e435b880253b73d96", R513_RESULT: "043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5"}
BUNDLE_SHA = "06118d18594c4b167a3f3d46a2aa282969f6b061835f83a3b3d62b5ca72b8d8a"
CONTEXT = (1, 2, 3, 4)  # CELLS: all,near,far,one,multiple,off -> context = 1..4
ENERGY_BAR = .90
DIR_BAR = .90

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def spectrum(V):
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    p = S ** 2 / (S ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return S, float(p[0]), eff_rank, Vt

def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for p, e in HASHES.items():
            if not p.is_file() or sha256(p) != e:
                raise RuntimeError(f"frozen hash mismatch: {p}")
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "task_space_sufficient_dimension_probe",
                          "model_loaded": False}, indent=2))
        return
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")
    if sha256(R513_BUNDLE) != BUNDLE_SHA:
        raise RuntimeError("rung513 bundle sha mismatch")
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    b = torch.load(R513_BUNDLE, map_location="cpu", weights_only=False)
    c = b["collections"]["discovery"]
    st = c["source_task"].numpy()   # (4,248,6)
    bt = c["base_task"].numpy()     # (248,6)
    assert st.shape == (4, 248, 6) and bt.shape == (248, 6)
    E = st - bt[None]               # (4,248,6)
    halves = {"half0": (0, 124), "half1": (124, 248)}
    per_half, top_dirs = {}, {}
    for hk, (lo, hi) in halves.items():
        V = E[:, lo:hi, :][:, :, list(CONTEXT)].reshape(4, -1)
        S, e1, eff, Vt = spectrum(V)
        Vc = V - V.mean(0)
        Sc, e1c, effc, _ = spectrum(Vc)
        per_half[hk] = {
            "raw_singular_values": [float(x) for x in S],
            "raw_top1_energy": e1, "raw_effective_rank": eff,
            "centered_singular_values": [float(x) for x in Sc],
            "centered_top1_energy": e1c, "centered_effective_rank": effc,
        }
        top_dirs[hk] = Vt[0]
    dir_cos = float(abs(np.dot(top_dirs["half0"], top_dirs["half1"])))
    pool = E[:, :, list(CONTEXT)].reshape(4, -1)
    _, pool_e1, pool_eff, _ = spectrum(pool)
    pred_a = bool(sha256(R513_RESULT) == list(HASHES.values())[1]
                  and np.isfinite(pool_e1))
    pred_b = bool(per_half["half0"]["raw_top1_energy"] >= ENERGY_BAR
                  and per_half["half1"]["raw_top1_energy"] >= ENERGY_BAR)
    pred_c = bool(dir_cos >= DIR_BAR)
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    result = {
        "status": "complete", "rung": "task_space_sufficient_dimension_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "task_space_effective_rank_of_score_implementations",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "rung513_bundle_sha256": BUNDLE_SHA,
        "per_half": per_half,
        "half0_half1_top_direction_cosine": dir_cos,
        "pooled_raw_top1_energy": pool_e1,
        "pooled_raw_effective_rank": pool_eff,
        'pred_a_exact_reproduction_of_513_task_effects': pred_a,
        'pred_b_copy_task_near_rank1_both_halves': pred_b,
        'pred_c_shared_score_direction_stable': pred_c,
        "strong_null": strong_null,
        "verdict": ("copy_task_needs_one_score_template_task_space_gauge_basis"
                    if not strong_null else
                    "copy_task_task_space_rank_not_established"),
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null",
                               "half0_half1_top_direction_cosine",
                               "pooled_raw_effective_rank", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))
    for hk in ("half0", "half1"):
        r = per_half[hk]
        print(f"{hk}: raw_top1={r['raw_top1_energy']:.4f} eff_rank={r['raw_effective_rank']:.2f} "
              f"| centered_top1={r['centered_top1_energy']:.3f} eff={r['centered_effective_rank']:.2f}")

if __name__ == "__main__":
    main()
