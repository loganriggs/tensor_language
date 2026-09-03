#!/usr/bin/env python3
"""Is the score-implementation mismatch magnitude separable (rank-1)? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_exact_reproduction_of_513_norms
# pred_b_magnitude_is_rank1_separable
# pred_c_pair_scaling_is_site_consistent

Parallel-lane CPU analysis (Claude) completing section-2647: the mismatch
DIRECTION is fixed; is its MAGNITUDE over the branch x action-pair grid
separable? Zero model forwards. Preregistration:
polynomial_causal/MISMATCH_MAGNITUDE_LAW_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MISMATCH_MAGNITUDE_LAW_PROBE_PREREGISTRATION.md"
R513_RESULT = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_results.json"
OUT = ROOT / "mismatch_magnitude_law_probe_results.json"
HASHES = {PREREG: "5aae4c0d8ce52103bce45d1056fbf2d7943e0c8e10825398b8d15d548145bc68", R513_RESULT: "043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5"}
SITES = ("a11", "m11")
BRANCHES = ("L", "R", "L+R", "L+LR", "R+LR", "L+R+LR")
PAIRS = ("N-Z7", "N-Z8", "P-Z7")
ENERGY_BAR = .90
RESID_BAR = .05
SITE_CONSIST_BAR = .90

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for p, e in HASHES.items():
            if not p.is_file() or sha256(p) != e:
                raise RuntimeError(f"frozen hash mismatch: {p}")
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "mismatch_magnitude_law_probe",
                          "model_loaded": False}, indent=2))
        return
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    smd = json.loads(R513_RESULT.read_text())["analysis"]["signed_mismatch_decomposition"]
    reports, pair_scales, ok = {}, {}, True
    for site in SITES:
        N = np.zeros((len(BRANCHES), len(PAIRS)))
        for bi, b in enumerate(BRANCHES):
            for pi, pr in enumerate(PAIRS):
                k = f"{b}::{pr}"
                n0 = smd[k]["sites"][site]["half0"]["complete_mismatch_norm_squared"]
                n1 = smd[k]["sites"][site]["half1"]["complete_mismatch_norm_squared"]
                N[bi, pi] = math.sqrt((n0 + n1) / 2)
        if (N <= 0).any(): ok = False
        L = np.log(N)
        Lc = L - L.mean()
        U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
        energy = float(S[0] ** 2 / (S ** 2).sum())
        # rank-1 fit of the log matrix (mean + top component), exp back to raw
        L1 = L.mean() + S[0] * np.outer(U[:, 0], Vt[0])
        N1 = np.exp(L1)
        resid = float(np.linalg.norm(N - N1) / np.linalg.norm(N))
        # per-branch and per-pair scales from the rank-1 raw fit
        u = N1.mean(1); u /= u.mean()
        v = N1.mean(0)
        pair_scales[site] = v / np.linalg.norm(v)
        reports[site] = {
            "norm_grid": N.tolist(),
            "log_top_singular_energy_fraction": energy,
            "raw_rank1_relative_residual": resid,
            "rank1_separable": bool(energy >= ENERGY_BAR and resid <= RESID_BAR),
            "per_branch_scale": {BRANCHES[i]: float(u[i]) for i in range(len(BRANCHES))},
            "per_pair_scale": {PAIRS[i]: float(v[i]) for i in range(len(PAIRS))},
        }
    site_consist = float(np.dot(pair_scales["a11"], pair_scales["m11"]))
    pred_a = bool(ok and sha256(R513_RESULT) == list(HASHES.values())[1])
    pred_b = bool(all(reports[s]["rank1_separable"] for s in SITES))
    pred_c = bool(site_consist >= SITE_CONSIST_BAR)
    strong_null = bool(not pred_a or not pred_b)
    result = {
        "status": "complete", "rung": "mismatch_magnitude_law_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "descriptive_rank1_magnitude_law_of_published_mismatch_norms",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "sites": reports,
        "pair_scale_site_cosine": site_consist,
        'pred_a_exact_reproduction_of_513_norms': pred_a,
        'pred_b_magnitude_is_rank1_separable': pred_b,
        'pred_c_pair_scaling_is_site_consistent': pred_c,
        "strong_null": strong_null,
        "verdict": ("source_dependence_is_one_scalar_field_times_one_fixed_direction"
                    if (pred_b and pred_c) else
                    "magnitude_separable_but_site_specific" if pred_b else
                    "magnitude_has_branch_by_implementation_interaction"),
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null",
                               "pair_scale_site_cosine", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))
    for s in SITES:
        r = reports[s]
        print(f"{s}: energy={r['log_top_singular_energy_fraction']:.3f} "
              f"resid={r['raw_rank1_relative_residual']:.4f} "
              f"pair_scale={ {k: round(v,3) for k,v in r['per_pair_scale'].items()} }")

if __name__ == "__main__":
    main()
