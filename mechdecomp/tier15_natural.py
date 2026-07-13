"""TIER 1.5 on NATURAL TEXT — closes two flagged items from the random-token run.

(1) The random-token induction batch had base L1H2 match −0.0223, ~20× weaker than the −0.434
    on the natural demo sequence. Ordering reproduced, magnitudes did not. Redo the joint-Δŝ gate
    on real induction sites mined from the val corpus, where the circuit is actually engaged.

(2) `logs/tier15_omp2.log`'s "OV-map R² 0.06–0.46 (lasso) → 0.89–0.97 (OMP)" recorded NO L0 for the
    lasso run and is therefore uncitable (the toy showed a fixed λ can sit at 7× ktrue). Re-measure
    lasso vs OMP at MATCHED L0, separating the two questions:
       (a) E-step only  — same OMP-learned dictionary, codes by lasso(λ) vs OMP(k)
       (b) full pipeline — dictionary learned under each E-step

Induction sites are mined, not synthesised: positions (j, q) with tok[j]==tok[q], q > j+1, and
tok[q+1]==tok[j+1] (the copy actually succeeds). src = j+1.

Guard: base match on mined sites must be strongly negative (order −0.4), else the sites are not
engaging the circuit and the run is thrown out rather than reported.

Run: TL_CORPUS=tiny python -m mechdecomp.tier15_natural
"""

import numpy as np
import torch

from lm_eval import load_model
from text_data import N_CTX, RUNS, val_windows

from .estep import solve_codes
from .release_d import omp_codes
from .tier15_contraction import H_IND, KSPARSE, M_ATOMS, head_write, l1_match, learn_dict

DEV = "cuda"


def mine_induction_sites(toks, max_sites=96):
    """(seq, q, j, src) where tok[j]==tok[q], tok[q+1]==tok[j+1], q>j+1."""
    sites = []
    T = toks.shape[1]
    for b in range(toks.shape[0]):
        row = toks[b]
        seen = {}
        for p in range(T - 1):
            t = int(row[p])
            if t in seen:
                j = seen[t]
                if p > j + 1 and int(row[p + 1]) == int(row[j + 1]):
                    sites.append((b, p, j, j + 1))
                    if len(sites) >= max_sites:
                        return sites
            seen[t] = p
    return sites


@torch.no_grad()
def match_at(model, h1, sites):
    """L1H2 match s(q, src) evaluated per-site (each site has its own q, src)."""
    pat = model.layers[1].pattern(h1)
    return torch.stack([pat[b, H_IND, q, s] for (b, q, _, s) in sites])


@torch.no_grad()
def main():
    model = load_model(RUNS / "attn2-seed0", None, DEV)
    L0, L1 = model.layers[0], model.layers[1]

    data, _ = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX] for w in range(64)]).astype(np.int64)
    toks = torch.from_numpy(buf).to(DEV)
    sites = mine_induction_sites(toks)
    print(f"mined {len(sites)} natural induction sites (tok[j]==tok[q], tok[q+1]==tok[j+1])\n", flush=True)

    h0 = model.embed(toks).float()
    h1 = L0(h0).float()
    base = match_at(model, h1, sites)
    print(f"BASE L1H2 match on mined sites = {base.mean():+.4f}  (random-token run: −0.0223)", flush=True)
    assert base.mean() < -0.1, f"GUARD FAIL: base match {base.mean():.4f} — sites do not engage the circuit"
    print("  [guard ok] circuit is engaged at these sites\n", flush=True)

    # ---------- causal ground truth ----------
    print("CAUSAL — zero each L0 head's write at each site's src:", flush=True)
    writes, us, OVs, true_ds = {}, {}, {}, {}
    for h in range(4):
        w, u, OV = head_write(model, h0, h)
        writes[h], us[h], OVs[h] = w, u, OV
        h1a = h1.clone()
        for (b, _, _, s) in sites:
            h1a[b, s, :] -= w[b, s, :]
        m = match_at(model, h1a, sites)
        true_ds[h] = float((m - base).mean())
        print(f"  zero L0H{h}@src → match {m.mean():+.4f}   Δs {true_ds[h]:+.4f}", flush=True)
    dom = max(range(4), key=lambda h: abs(true_ds[h]))
    assert dom == 3, f"GUARD FAIL: causal truth says L0H3, got L0H{dom}"
    print(f"  [guard ok] L0H3 dominant\n", flush=True)

    # ---------- dictionaries ----------
    n0 = L0.norm(h0).reshape(-1, h0.shape[-1]).float()
    scale = float(n0.shape[-1]) ** 0.5
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n0.shape[0], generator=g)[:10000]
    X0 = (n0[idx].T / scale).contiguous()
    print("decomposing L0 OV maps (OMP k=8 + validated M-step):", flush=True)
    Ds = {}
    for h in range(4):
        Ds[h], r2 = learn_dict(OVs[h], X0)
        print(f"  L0H{h}-OV  R2 {r2:.4f}", flush=True)

    # ---------- joint Δŝ gate on natural sites ----------
    print("\nJOINT GATE on natural sites (remove atom-reconstructed write):", flush=True)
    print("  head   true Δs    atom Δŝ", flush=True)
    est = {}
    for h in range(4):
        OV, D = OVs[h], Ds[h]
        h1r = h1.clone()
        srcs = sorted({s for (_, _, _, s) in sites})
        bb = torch.tensor([b for (b, _, _, _) in sites], device=DEV)
        for s in srcs:
            rows = torch.unique(bb[[i for i, (_, _, _, ss) in enumerate(sites) if ss == s]])
            uu = (us[h][rows, s, :].T / scale).contiguous()
            Y = OV @ uu
            C, _ = omp_codes(OV @ D, Y, KSPARSE)
            w_hat = (L0.scale * scale) * ((OV @ D) @ C).T
            h1r[rows, s, :] -= w_hat
        est[h] = float((match_at(model, h1r, sites) - base).mean())
        print(f"  L0H{h}  {true_ds[h]:+.4f}   {est[h]:+.4f}", flush=True)
    de = max(range(4), key=lambda h: abs(est[h]))
    nxt = max(abs(est[h]) for h in range(4) if h != 3)
    print(f"\n  JOINT GATE: argmax|Δŝ| = L0H{de} → {'PASS' if de == 3 else 'FAIL'}"
          f"   ({abs(est[3]):.4f} vs next {nxt:.4f}, {abs(est[3])/max(nxt,1e-9):.2f}x)", flush=True)

    # ---------- (2) matched-L0 lasso vs OMP ----------
    print("\nMATCHED-L0 lasso vs OMP on L0H3-OV (closes the uncitable 0.06-0.46 claim)", flush=True)
    W, D = OVs[3], Ds[3]
    Y = W @ X0
    den = ((Y - Y.mean(1, keepdim=True)) ** 2).sum()

    def r2c(C):
        return float(1 - (((W @ D) @ C - Y) ** 2).sum() / den)

    Comp, _ = omp_codes(W @ D, Y, KSPARSE)
    print(f"\n(a) E-STEP ONLY (same OMP-learned dictionary):", flush=True)
    print(f"  OMP  k={KSPARSE}            L0 {float((Comp.abs()>1e-8).sum(0).float().mean()):5.1f}   R2 {r2c(Comp):.4f}", flush=True)
    for lam in (0.0005, 0.002, 0.01, 0.05, 0.2, 0.8):
        Cl = solve_codes(W, D, X0, lam=lam) * (D.T @ X0)
        l0 = float((Cl.abs() > 1e-8).sum(0).float().mean())
        star = "  <- matched" if abs(l0 - KSPARSE) < 3 else ""
        print(f"  lasso λ={lam:<6}       L0 {l0:5.1f}   R2 {r2c(Cl):.4f}{star}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
