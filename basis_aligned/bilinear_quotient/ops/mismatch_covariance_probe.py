#!/usr/bin/env python3
"""Is the score-implementation mismatch shape gauge-covariant? (CPU probe)

# BQGATE: EXPERIMENT
# pred_a_exact_reproduction_of_513_shares
# pred_b_mismatch_shape_is_gauge_covariant
# pred_c_stable_dominant_factor_subspace

Parallel-lane CPU analysis (Claude) on rung513's published signed-mismatch
decomposition. Zero model forwards. Preregistration:
polynomial_causal/MISMATCH_COVARIANCE_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MISMATCH_COVARIANCE_PROBE_PREREGISTRATION.md"
R513_RESULT = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_results.json"
OUT = ROOT / "mismatch_covariance_probe_results.json"
HASHES = {PREREG: "164bc70dbed5098829e3efa47d9de24323d11c35a5e892014265eb2bc70f714b", R513_RESULT: "043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5"}
SITES = ("a11", "m11")
PERM_SEEDS = tuple(20260907 + i for i in range(16))
COV_BAR = .70
MARGIN = .10
TOPK = 3
TOPK_AGREE_MIN = 12  # of 18

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
                          "rung": "mismatch_covariance_probe",
                          "model_loaded": False}, indent=2))
        return
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    smd = json.loads(R513_RESULT.read_text())["analysis"]["signed_mismatch_decomposition"]
    keys = list(smd.keys())
    assert len(keys) == 18, f"expected 18 keys, got {len(keys)}"
    site_reports = {}
    sum_ok = True
    for site in SITES:
        terms = list(smd[keys[0]]["sites"][site]["half0"]
                     ["term_signed_inner_product_fractions"].keys())
        rows, tops = [], []
        for k in keys:
            h0 = smd[k]["sites"][site]["half0"]["term_signed_inner_product_fractions"]
            h1 = smd[k]["sites"][site]["half1"]["term_signed_inner_product_fractions"]
            v = np.array([(h0[t] + h1[t]) / 2 for t in terms], dtype=np.float64)
            if abs(sum(h0.values()) - 1.0) > 1e-6 or abs(sum(h1.values()) - 1.0) > 1e-6:
                sum_ok = False
            rows.append(v)
            tops.append(frozenset(np.argsort(np.abs(v))[::-1][:TOPK].tolist()))
        M = np.array(rows)
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-30)
        cos = Mn @ Mn.T
        iu = np.triu_indices(len(keys), 1)
        mean_cos = float(cos[iu].mean())
        perm_means = []
        for seed in PERM_SEEDS:
            rng = np.random.default_rng(seed)
            P = np.array([r[rng.permutation(len(terms))] for r in Mn])
            Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-30)
            pc = Pn @ Pn.T
            perm_means.append(float(pc[iu].mean()))
        perm_q95 = float(np.quantile(perm_means, .95))
        from collections import Counter
        c = Counter()
        for t in tops: c.update(t)
        top3_global = [terms[i] for i, _ in c.most_common(TOPK)]
        top3_set = frozenset(terms.index(t) for t in top3_global)
        agree = sum(1 for t in tops if len(t & top3_set) >= 2)
        mean_fp = Mn.mean(0); mean_fp /= np.linalg.norm(mean_fp).clip(1e-30)
        site_reports[site] = {
            "terms": terms, "mean_pairwise_cosine": mean_cos,
            "perm_mean_pairwise_cosine_q95": perm_q95,
            "covariant": bool(mean_cos >= COV_BAR and mean_cos >= perm_q95 + MARGIN),
            "top3_global": top3_global,
            "keys_sharing_2of_top3": agree,
            "stable_subspace": bool(agree >= TOPK_AGREE_MIN),
            "mean_fingerprint_top": {terms[i]: float(mean_fp[i])
                                     for i in np.argsort(np.abs(mean_fp))[::-1][:6]},
        }
    pred_a = bool(sum_ok
                  and sha256(R513_RESULT) == list(HASHES.values())[1])
    pred_b = bool(all(site_reports[s]["covariant"] for s in SITES))
    pred_c = bool(all(site_reports[s]["stable_subspace"] for s in SITES))
    strong_null = bool(not pred_a or not pred_b)
    result = {
        "status": "complete", "rung": "mismatch_covariance_probe",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "descriptive_gauge_covariance_of_published_mismatch_shares",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "keys": keys, "sites": site_reports,
        'pred_a_exact_reproduction_of_513_shares': pred_a,
        'pred_b_mismatch_shape_is_gauge_covariant': pred_b,
        'pred_c_stable_dominant_factor_subspace': pred_c,
        "strong_null": strong_null,
        "verdict": ("mismatch_shape_is_gauge_covariant_low_dim_factor_subspace"
                    if not strong_null else
                    "mismatch_shape_is_key_idiosyncratic_unstructured"),
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "verdict", "strong_null", "runtime_s")
                      or k.startswith("pred_")}, indent=2, sort_keys=True))
    for s in SITES:
        r = site_reports[s]
        print(f"{s}: mean_cos={r['mean_pairwise_cosine']:.3f} "
              f"perm_q95={r['perm_mean_pairwise_cosine_q95']:.3f} "
              f"top3={r['top3_global']} agree={r['keys_sharing_2of_top3']}/18")

if __name__ == "__main__":
    main()
