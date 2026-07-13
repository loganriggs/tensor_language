"""TIER 1.5 on the OWT models — does a causally-verified circuit even EXIST to recover?

The heartbeat lists runs_owt/{block2-dense-seed0, attn2-s120k-dense-seed0} as Tier-1.5 comparison
models. But an established result of the circuits program is that **OWT never forms induction
unaided** (only a copy-burst mixture installs it). Dense OWT models may therefore have NO induction
edge, in which case the decomposition gate is *inapplicable* — not failed.

So this script does not assume a circuit. It:
  1. mines natural induction sites from the OWT val stream;
  2. measures, for every L1 head, its match weight at those sites, and picks the head with the
     strongest (most negative) match — the induction-head candidate;
  3. measures the CAUSAL effect on that head of zeroing each L0 head's write at src;
  4. only if one L0 head dominates (>= DOMINANCE x the next) does it run the decomposition gate;
     otherwise it reports "no ground-truth edge — gate inapplicable" and stops.

This ordering matters: a decomposition gate against a circuit that does not exist would produce a
number that looks like a result. Ground truth first, method second.

Run: TL_CORPUS=owt python -m mechdecomp.tier15_owt [run_name]
"""

import sys

import numpy as np
import torch

from lm_eval import load_model
from text_data import N_CTX, RUNS, val_windows

from .release_d import omp_codes
from .tier15_contraction import KSPARSE, learn_dict

DEV = "cuda"
DOMINANCE = 2.0          # the top L0 head must be this many x the next to count as ground truth
N_WIN = 64


def mine_sites(toks, max_sites=128):
    sites = []
    for b in range(toks.shape[0]):
        row, seen = toks[b], {}
        for p in range(toks.shape[1] - 1):
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
def match_at(layer, h, sites, head):
    pat = layer.pattern(h)
    return torch.stack([pat[b, head, q, s] for (b, q, _, s) in sites])


@torch.no_grad()
def head_write(L0, h0, h):
    dh = L0.d_head
    n0 = L0.norm(h0)
    pat0 = L0.pattern(h0)[:, h]
    u = torch.einsum("bqk,bkd->bqd", pat0, n0)
    OV = (L0.o.weight[:, h * dh:(h + 1) * dh] @ L0.v.weight[h * dh:(h + 1) * dh, :]).float()
    return L0.scale * (u @ OV.T), u, OV


@torch.no_grad()
def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "attn2-s120k-dense-seed0"
    model = load_model(RUNS / run, None, DEV)
    print(f"=== {run} ===", flush=True)

    # locate the two attention layers (block2 interleaves MLPs)
    attn_idx = [i for i, l in enumerate(model.layers) if hasattr(l, "q1")]
    print(f"attention layers at {attn_idx} of {len(model.layers)}", flush=True)
    if len(attn_idx) < 2:
        print("fewer than 2 attention layers — no L0→L1 edge possible"); return
    i0, i1 = attn_idx[0], attn_idx[1]

    data, _ = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX] for w in range(N_WIN)]).astype(np.int64)
    toks = torch.from_numpy(buf).to(DEV)
    sites = mine_sites(toks)
    print(f"mined {len(sites)} natural induction sites", flush=True)
    if len(sites) < 20:
        print("too few sites — inconclusive"); return

    h0 = model.embed(toks).float()
    h = h0
    for i in range(i1):
        h = model.layers[i](h)
    h1 = h.float()                                # input to the second attention layer
    L0, L1 = model.layers[i0], model.layers[i1]
    nh = L1.n_head

    # ---------- 2. which L1 head, if any, is the induction head? ----------
    print("\nL1 head match weights at induction sites (most negative = strongest match):", flush=True)
    ms = {}
    for hd in range(nh):
        ms[hd] = float(match_at(L1, h1, sites, hd).mean())
        print(f"  L1H{hd}  match {ms[hd]:+.4f}", flush=True)
    cand = min(range(nh), key=lambda k: ms[k])
    print(f"  → candidate induction head: L1H{cand} (match {ms[cand]:+.4f})", flush=True)
    if ms[cand] > -0.02:
        print("\nNO INDUCTION HEAD: no L1 head forms a meaningful match at induction sites.")
        print("Ground truth does not exist ⇒ decomposition gate is INAPPLICABLE (not failed).")
        return

    # ---------- 3. causal ground truth ----------
    base = match_at(L1, h1, sites, cand)
    print(f"\nCAUSAL: zero each L0 head's write at src; effect on L1H{cand} match "
          f"(base {base.mean():+.4f}):", flush=True)
    ds, OVs, us = {}, {}, {}
    for hd in range(L0.n_head):
        w, u, OV = head_write(L0, h0, hd)
        OVs[hd], us[hd] = OV, u
        hx = h1.clone()
        for (b, _, _, s) in sites:
            hx[b, s, :] -= w[b, s, :]
        ds[hd] = float((match_at(L1, hx, sites, cand) - base).mean())
        print(f"  zero L0H{hd}@src → Δs {ds[hd]:+.4f}", flush=True)
    order = sorted(range(L0.n_head), key=lambda k: -abs(ds[k]))
    top, nxt = order[0], order[1]
    ratio = abs(ds[top]) / max(abs(ds[nxt]), 1e-9)
    print(f"  top L0H{top} ({abs(ds[top]):.4f}) vs next L0H{nxt} ({abs(ds[nxt]):.4f}) = {ratio:.2f}x", flush=True)
    if ratio < DOMINANCE:
        print(f"\nNO DOMINANT EDGE (< {DOMINANCE}x): the circuit is distributed across L0 heads.")
        print("Ground truth is not a single edge ⇒ the single-edge gate is INAPPLICABLE (not failed).")
        print(f"This is itself a finding: OWT-dense {run} has no clean L0→L1 induction edge.")
        return

    # ---------- 4. decomposition gate (only now) ----------
    print(f"\nGround truth established: L0H{top} → L1H{cand} at {ratio:.2f}x. Running the gate.", flush=True)
    n0 = L0.norm(h0).reshape(-1, h0.shape[-1]).float()
    sc = float(n0.shape[-1]) ** 0.5
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n0.shape[0], generator=g)[:10000]
    X0 = (n0[idx].T / sc).contiguous()
    est = {}
    for hd in range(L0.n_head):
        D, r2 = learn_dict(OVs[hd], X0)
        print(f"  L0H{hd}-OV  R2 {r2:.4f}", flush=True)
        hx = h1.clone()
        for s in sorted({s for (_, _, _, s) in sites}):
            rows = torch.unique(torch.tensor([b for (b, _, _, ss) in sites if ss == s], device=DEV))
            uu = (us[hd][rows, s, :].T / sc).contiguous()
            C, _ = omp_codes(OVs[hd] @ D, OVs[hd] @ uu, KSPARSE)
            hx[rows, s, :] -= (L0.scale * sc) * ((OVs[hd] @ D) @ C).T
        est[hd] = float((match_at(L1, hx, sites, cand) - base).mean())
    print(f"\n  head   true Δs     atom Δŝ", flush=True)
    for hd in range(L0.n_head):
        print(f"  L0H{hd}  {ds[hd]:+.4f}   {est[hd]:+.4f}", flush=True)
    de = max(range(L0.n_head), key=lambda k: abs(est[k]))
    nx = max(abs(est[k]) for k in range(L0.n_head) if k != de)
    print(f"\n  JOINT GATE: argmax|Δŝ| = L0H{de}, truth L0H{top} → "
          f"{'PASS' if de == top else 'FAIL'}  ({abs(est[de]):.4f} vs next {nx:.4f}, "
          f"{abs(est[de])/max(nx,1e-9):.2f}x)", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
