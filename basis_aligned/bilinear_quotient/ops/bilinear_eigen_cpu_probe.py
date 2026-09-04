#!/usr/bin/env python
"""bilinear_eigen_cpu_probe -- build the weight-space object of arXiv:2410.08417 for bilin18. CPU ONLY, no model run.

Pearce, Dooms, Rigg, Oramas & Sharkey (ICLR 2025 Spotlight, arXiv:2410.08417) propose that a gate-free bilinear MLP,
mlp(x) = Down(Left(x) o Right(x)) + b, folds into a third-order tensor whose contraction with an OUTPUT direction u gives a
SYMMETRIC matrix, and that eigendecomposing that matrix yields interpretable directions FROM WEIGHTS ALONE. bilin18 is exactly
that architecture (config: gated=False, squared_mlp=False), so the object is well defined here.

For output direction u, the bilinear form is  f_u(x) = <u, Down(Left(x) o Right(x))> = x^T A_u x  with
    A_u = Left^T diag(Down^T u) Right,        M_u = (A_u + A_u^T) / 2
and only the symmetric part M_u affects f_u. This script builds M_u for chosen blocks and output directions and records its
spectrum. It runs no forward pass and needs no GPU: that is the point of the method being tested.

The GPU half -- does |eigenvalue| predict CAUSAL damage? -- is preregistered separately
(CIRCUIT_BATTERY_BILINEAR_EIGEN_CAUSAL_PREREGISTRATION.md); SS2822-SS2826 measured that energy/magnitude rankings do NOT track
causal effect in this model, so that is a live question rather than a formality.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")
sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import fastload


def symmetric_form(blk, u):
    """M_u = sym(Left^T diag(Down^T u) Right) for output direction u. Pure weight algebra."""
    W = (blk.mlp.Down.weight.T.float() @ u.float())          # (hidden,)
    A = blk.mlp.Left.weight.T.float() @ (W[:, None] * blk.mlp.Right.weight.float())
    return 0.5 * (A + A.T)


def report(name, M, k=8):
    ev = torch.linalg.eigvalsh(M).flip(0)                    # descending
    a = ev.abs()
    tot = float(a.sum())
    order = torch.argsort(a, descending=True)
    top = a[order][:k]
    eff = float(a.sum() ** 2 / (a ** 2).sum())               # participation ratio of |eigenvalues|
    return {"name": name, "dim": int(M.shape[0]),
            "top8_abs_eigenvalues": [float(x) for x in top],
            "top8_share_of_abs_mass": float(top.sum() / max(tot, 1e-12)),
            "effective_rank_abs": eff,
            "n_positive": int((ev > 0).sum()), "n_negative": int((ev < 0).sum()),
            "spectral_norm": float(a.max()), "trace": float(ev.sum())}


def main():
    t0 = time.time()
    m = fastload.load_model_fast()          # CPU; no .to(cuda), no forward
    tok = R  # for paths only
    out = {"source": "arXiv:2410.08417 operational definition, applied to bilin18", "blocks": {}}
    # output directions: pooled numeric-answer axis and a random control of matched norm
    import circuit_battery_tasks as BANK
    ids = sorted({BANK.ENC.encode(s)[0] for s in [f" {i}" for i in range(0, 100)]
                  if len(BANK.ENC.encode(s)) == 1})
    WU = m.lm_head.weight.detach().float()
    u_num = WU[ids].mean(0)                                   # "a number goes here" axis
    u_num = u_num / u_num.norm()
    g = torch.Generator().manual_seed(2853)
    u_rand = torch.randn(WU.shape[1], generator=g); u_rand = u_rand / u_rand.norm()
    for layer in (8, 10, 11):
        blk = m.transformer.h[layer]
        for label, u in (("numeric_axis", u_num), ("random_axis", u_rand)):
            M = symmetric_form(blk, u)
            out["blocks"][f"mlp{layer}/{label}"] = report(f"mlp{layer}/{label}", M)
            print(f"mlp{layer:<2d} {label:12s} "
                  f"top1={out['blocks'][f'mlp{layer}/{label}']['top8_abs_eigenvalues'][0]:.4g} "
                  f"top8_share={out['blocks'][f'mlp{layer}/{label}']['top8_share_of_abs_mass']:.4f} "
                  f"eff_rank={out['blocks'][f'mlp{layer}/{label}']['effective_rank_abs']:.1f} "
                  f"pos/neg={out['blocks'][f'mlp{layer}/{label}']['n_positive']}/"
                  f"{out['blocks'][f'mlp{layer}/{label}']['n_negative']}", flush=True)
    out["seconds"] = time.time() - t0
    out["gpu_used"] = False
    p = R.ROOT / "bilinear_eigen_cpu_probe_results.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p.name} in {out['seconds']:.1f}s, no GPU")


if __name__ == "__main__":
    main()
