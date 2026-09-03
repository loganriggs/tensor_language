#!/usr/bin/env python3
"""Bootstrap resolution of the section-2640 locality-tail uptick.

# BQGATE: EXPERIMENT
# pred_a_exact_deterministic_bootstrap_instrument
# pred_b_tail_uptick_has_definite_sign
# pred_c_bootstrap_self_consistent

Parallel-lane CPU analysis (Claude). Document-level bootstrap on section-2640's
frozen per-token bundle: is f(1/128) > f(1/64) real structure or estimator
noise? Zero model forwards. Preregistration:
polynomial_causal/LOCALITY_TAIL_UPTICK_BOOTSTRAP_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import torch
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "LOCALITY_TAIL_UPTICK_BOOTSTRAP_PREREGISTRATION.md"
R2640_RESULT = ROOT / "mlp1_write_locality_density_envelope_ext_results.json"
R2640_BUNDLE = ROOT / "mlp1_write_locality_density_envelope_ext_per_token.pt"
OUT = ROOT / "locality_tail_uptick_bootstrap_results.json"
HASHES = {
    PREREG: "05277c96b29df589367f07f0f29eb92cce0bcc63398bc732b05cc383363dd851",
    R2640_RESULT: "3cc9db75571137b46c292b4dae62d5e9c3ba88bd4ff261c722343267a040e4ea",
    R2640_BUNDLE: "fee1ff68d9699fd07f80fb63ebcc4721c22864cd2535d7dd018b79fb2587cf97",
}
BRANCHES = ("T", "I")
TAIL = {"d128": 0, "d64": 1}
HALVES = ((0, 250), (250, 500))
BOOT = 10000
SEED = 20260906
TOL = 1e-9

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def frac(recovery, x, mask):
    r = recovery[mask]; v = x[mask]
    return float((r * v).sum() / (v * v).sum().clamp_min(1e-30))

def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for p, e in HASHES.items():
            if not p.is_file() or sha256(p) != e:
                raise RuntimeError(f"frozen hash mismatch: {p}")
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "locality_tail_uptick_bootstrap",
                          "model_loaded": False}, indent=2))
        return
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")
    if OUT.exists():
        raise RuntimeError("output namespace already exists")
    receipt = json.loads(R2640_RESULT.read_text())
    if receipt.get("pred_a_exact_lawful_live_envelope_instrument") is not True:
        raise RuntimeError("2640 pred_a not true")
    b = torch.load(R2640_BUNDLE, map_location="cpu", weights_only=False)
    native = b["native"].double(); absent = b["absent"].double()
    arms = b["arms"].double(); masks = {k: b["masks"][k] for k in TAIL}
    assert tuple(native.shape) == (500, 256)
    assert tuple(arms.shape) == (2, 6, 500, 256)
    gen = torch.Generator().manual_seed(SEED)
    cells, point_ok = {}, True
    for bi, branch in enumerate(BRANCHES):
        x_full = absent[bi] - native
        for hi, (lo, hi_) in enumerate(HALVES):
            docs = list(range(lo, hi_))
            point = {}
            for dk, di in TAIL.items():
                rec = absent[bi] - arms[bi, di]
                m = masks[dk]
                point[dk] = frac(rec[lo:hi_], x_full[lo:hi_], m[lo:hi_])
                anchor = receipt["analysis"]["reports"][branch][f"half{hi}"]["profile"][
                    {"d128": 0, "d64": 1}[dk]]
                if abs(point[dk] - anchor) > TOL: point_ok = False
            deltas = torch.empty(BOOT, dtype=torch.float64)
            n = hi_ - lo
            idx_base = torch.arange(lo, hi_)
            for t in range(BOOT):
                pick = idx_base[torch.randint(0, n, (n,), generator=gen)]
                f128 = frac((absent[bi] - arms[bi, 0])[pick], x_full[pick], masks["d128"][pick])
                f64 = frac((absent[bi] - arms[bi, 1])[pick], x_full[pick], masks["d64"][pick])
                deltas[t] = f128 - f64
            q = torch.quantile(deltas, torch.tensor([.025, .5, .975], dtype=torch.float64))
            cells[f"{branch}_half{hi}"] = {
                "f_d128": point["d128"], "f_d64": point["d64"],
                "delta_point": point["d128"] - point["d64"],
                "boot_mean": float(deltas.mean()),
                "boot_q025": float(q[0]), "boot_median": float(q[1]),
                "boot_q975": float(q[2]),
                "p_positive": float((deltas > 0).double().mean()),
            }
    p_pos = [c["p_positive"] for c in cells.values()]
    pred_a = bool(point_ok
                  and sha256(R2640_BUNDLE) == receipt["bundle"]["sha256"])
    definite_pos = all(p >= .975 for p in p_pos)
    definite_neg = all(p <= .025 for p in p_pos)
    pred_b = bool(definite_pos or definite_neg)
    pred_c = bool(all(c["boot_q025"] <= c["delta_point"] <= c["boot_q975"]
                      for c in cells.values()))
    strong_null = bool(not pred_a or not pred_b)
    result = {
        "status": "complete", "rung": "locality_tail_uptick_bootstrap",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "bootstrap_resolution_of_published_locality_statistics",
        "source_hashes": {str(p): sha256(p) for p in HASHES},
        "bootstrap_resamples": BOOT, "seed": SEED,
        "cells": cells,
        "point_estimates_reproduce_2640": point_ok,
        'pred_a_exact_deterministic_bootstrap_instrument': pred_a,
        'pred_b_tail_uptick_has_definite_sign': pred_b,
        'pred_c_bootstrap_self_consistent': pred_c,
        "uptick_verdict": ("real_positive_isolated_edit" if definite_pos else
                           "real_negative" if definite_neg else
                           "within_sampling_noise_tail_is_flat"),
        "strong_null": strong_null,
        "execution_price": {"full_model_forwards": 0, "cpu_only": True},
        "next_step": ("certificate_hull_stands_tail_flat_within_noise" if strong_null
                      else "tail_uptick_is_structure_isolated_edit_reading"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("status", "uptick_verdict", "strong_null",
                               "next_step", "runtime_s") or k.startswith("pred_")},
                     indent=2, sort_keys=True))
    print(json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()}
                      for k, v in cells.items()}, indent=2))

if __name__ == "__main__":
    main()
