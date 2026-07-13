"""Seeded, distribution-level replication of the masked-projector vs SAE comparison.

Why: the tick-15 headline leaned partly on `irrepl top1` — a MAX over 96 sampled atoms. A max
statistic across non-identical runs already produced one retracted result in this program (the "12x").
So: 3 seeds per dictionary, full distributions (median / mean / p90), and a paired rank test on the
pooled per-atom losses. Also trains a STRONGER SAE (8k steps) so the baseline is not handicapped —
the weak 3k-step SAE already won, and a stronger one can only sharpen the verdict.

Reports, for each dictionary:
    R2(Wx) on held-out           (is the objective learned at all?)
    irreplaceability: median, mean, p90, max over probed atoms
    Mann-Whitney U vs masked-projector on the pooled per-atom loss distributions

Guards:
  G1  each SAE must beat random on ITS own objective (x-R2), else it is not a fair baseline.
  G2  every dictionary is evaluated with the SAME data, K, k, probe indices, and eval points.

Run: python -m mechdecomp.vs_sae_replicate
"""

import statistics as st

import torch
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .irreplaceability import learn, omp_chunked, r2_of
from .tail_atoms import collect
from .vs_sae import train_sae, x_r2

DEV = "cuda"
K_ATOMS, K_SPARSE = 1024, 16
N_EVAL, N_PROBE = 800, 64
SEEDS = (0, 1, 2)


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U -> normal-approx z and the common-language effect size P(a>b)."""
    n1, n2 = len(a), len(b)
    all_v = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(all_v):
        j = i
        while j + 1 < len(all_v) and all_v[j + 1][0] == all_v[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[t] = r
        i = j + 1
    R1 = sum(ranks[t] for t, (_, g) in enumerate(all_v) if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    z = (U1 - mu) / sd if sd else 0.0
    return z, U1 / (n1 * n2)


def irrepl_losses(W, D, Y, k, probe_idx):
    den = ((Y - Y.mean(1, keepdim=True)) ** 2).sum()
    WD = W @ D
    base = r2_of(WD, omp_chunked(WD, Y, k), Y, den)
    K = D.shape[1]
    out = []
    for j in probe_idx:
        keep = [i for i in range(K) if i != j]
        Ck = omp_chunked(WD[:, keep], Y, k)
        out.append(base - r2_of(WD[:, keep], Ck, Y, den))
    return base, out


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").to(DEV).eval()
    H, T, W = collect(model, tok)
    d_in = W.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(H.shape[0], generator=g)
    Xtr = H[perm[:10000]].T.contiguous().to(DEV)
    Xev = H[perm[10000:10000 + N_EVAL]].T.contiguous().to(DEV)
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xev = Xtr / sc, Xev / sc
    Y = W @ Xev
    gp = torch.Generator().manual_seed(3)
    probe = torch.randperm(K_ATOMS, generator=gp)[:N_PROBE].tolist()   # SAME indices for all dicts
    print(f"K={K_ATOMS} k={K_SPARSE}  probes={N_PROBE}  eval={N_EVAL}  seeds={SEEDS}\n", flush=True)

    pooled = {}
    r2s = {}
    for tag in ("masked-projector", "SAE-8k", "random"):
        pooled[tag], r2s[tag] = [], []

    for s in SEEDS:
        Dl = learn(W, Xtr, K_ATOMS, K_SPARSE, seed=s)
        sae = train_sae(Xtr, K_ATOMS, K_SPARSE, steps=8000)
        Ds = Fn.normalize(sae.W_dec.detach(), dim=0)
        gr = torch.Generator().manual_seed(100 + s)
        Dr = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=gr).to(DEV), dim=0)

        xs, xr = x_r2(Ds, Xev, K_SPARSE), x_r2(Dr, Xev, K_SPARSE)
        assert xs > xr + 0.05, f"G1 FAIL seed {s}: SAE x-R2 {xs:.3f} vs random {xr:.3f}"
        print(f"seed {s}: [G1 ok] SAE x-R2 {xs:.4f} > random {xr:.4f}", flush=True)

        for tag, D in (("masked-projector", Dl), ("SAE-8k", Ds), ("random", Dr)):
            base, losses = irrepl_losses(W, D, Y, K_SPARSE, probe)
            pooled[tag] += losses
            r2s[tag].append(base)
            print(f"   {tag:18s} R2(Wx) {base:.4f}   median irrepl {st.median(losses):.6f}"
                  f"   p90 {sorted(losses)[int(.9*len(losses))]:.6f}   max {max(losses):.6f}", flush=True)

    print("\n=== POOLED over seeds (distribution-level, not max) ===", flush=True)
    print(f"{'dictionary':18s} {'R2(Wx) mean±sd':>18s} {'irrepl median':>14s} {'mean':>10s} {'p90':>10s} {'max':>10s}", flush=True)
    for tag in ("masked-projector", "SAE-8k", "random"):
        v = sorted(pooled[tag])
        m, sd = st.mean(r2s[tag]), (st.stdev(r2s[tag]) if len(r2s[tag]) > 1 else 0.0)
        print(f"{tag:18s} {m:11.4f}±{sd:.4f} {st.median(v):14.6f} {st.mean(v):10.6f} "
              f"{v[int(.9*len(v))]:10.6f} {max(v):10.6f}", flush=True)

    print("\n=== Mann-Whitney U vs masked-projector (pooled per-atom losses) ===", flush=True)
    for tag in ("SAE-8k", "random"):
        z, cl = mannwhitney_u(pooled[tag], pooled["masked-projector"])
        verdict = "SAE higher" if z > 1.96 else ("lower" if z < -1.96 else "no significant difference")
        print(f"  {tag:16s} z = {z:+.2f}   P({tag} > MP) = {cl:.3f}   -> {verdict}", flush=True)
    print("\n  The spec's premise needs masked-projector STRICTLY BETTER than SAE-8k.", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
