"""THE CENTRAL CLAIM: does the weight-aware dictionary beat an activation-only SAE?

The spec proposes a *weight-activation* method: atoms are chosen for how they decompose W's ACTION
(reconstruct `Wx`), not for how they reconstruct `x`. The obvious baseline — never run in this program —
is a plain SAE trained on the same activations, whose decoder directions are then used as the
dictionary. If the weight-aware atoms are no better, the whole premise is unsupported.

Everything is matched: same activations, same K, same sparsity k, same evaluation.

  D_learned : masked-projector dictionary (OMP + Gauss-Seidel M-step on Wx)
  D_sae     : decoder directions of a top-k SAE trained on x  (activation-only)
  D_random  : unlearned random directions                      (floor)

Measured on held-out data, with criteria whose power was established on the Tier-1 toy:
  * R²(Wx)                — reconstruction of the map's action
  * irreplaceability      — drop atom from dictionary, RE-SELECT codes (toy: true/random 8.4x;
                            single-atom ablation had no power, 1.54x)
  * token purity          — top-40 activating positions; chance level measured

Guards:
  G1 the SAE must actually train: its x-reconstruction R² must beat random directions' x-R².
  G2 chance purity is measured from random positions, never assumed.

Run: python -m mechdecomp.vs_sae
"""

import statistics as st
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .irreplaceability import learn, omp_chunked, r2_of
from .tail_atoms import collect, K_ATOMS, K_SPARSE, N_PROBE, TOPN

DEV = "cuda"
N_EVAL, N_PURITY = 1000, 8000
SAE_STEPS = 3000


class TopKSAE(nn.Module):
    """Standard top-k SAE on x: z = TopK(W_enc(x - b)), x_hat = W_dec z + b."""

    def __init__(self, d, K, k):
        super().__init__()
        self.k = k
        self.b = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.randn(K, d) / d ** 0.5)
        self.W_dec = nn.Parameter(Fn.normalize(torch.randn(d, K), dim=0))

    def forward(self, x):
        pre = (x - self.b) @ self.W_enc.T
        v, i = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
        return z @ self.W_dec.T + self.b, z


def train_sae(X, K, k, steps=SAE_STEPS):
    """X: (d, N) columns are datapoints."""
    d, N = X.shape
    sae = TopKSAE(d, K, k).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    Xt = X.T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    for s in range(steps):
        idx = torch.randint(0, N, (512,), generator=g, device=DEV)
        xb = Xt[idx]
        xh, _ = sae(xb)
        loss = ((xh - xb) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sae.W_dec.data = Fn.normalize(sae.W_dec.data, dim=0)
    return sae


def x_r2(D, X, k):
    """How well does dictionary D sparse-code x itself (OMP, k atoms)? For guard G1."""
    C = omp_chunked(D, X, k)
    return float(1 - ((D @ C - X) ** 2).sum() / ((X - X.mean(1, keepdim=True)) ** 2).sum())


def irrepl(W, D, Y, k, n_probe, seed=3):
    den = ((Y - Y.mean(1, keepdim=True)) ** 2).sum()
    WD = W @ D
    base = r2_of(WD, omp_chunked(WD, Y, k), Y, den)
    K = D.shape[1]
    g = torch.Generator().manual_seed(seed)
    cand = torch.randperm(K, generator=g)[:n_probe].tolist()
    loss = {}
    for j in cand:
        keep = [i for i in range(K) if i != j]
        Ck = omp_chunked(WD[:, keep], Y, k)
        loss[j] = base - r2_of(WD[:, keep], Ck, Y, den)
    return base, loss


def purity_of(codes_row, Tpu, n=TOPN):
    v = codes_row.abs()
    if int((v > 1e-8).sum()) < n:
        return None
    idx = torch.topk(v, n).indices
    c = Counter(int(Tpu[i]) for i in idx)
    return c.most_common(1)[0][1] / n


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").to(DEV).eval()
    H, T, W = collect(model, tok)
    d_in = W.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(H.shape[0], generator=g)
    tr, ev = perm[:10000], perm[10000:10000 + N_EVAL]
    pu = perm[10000 + N_EVAL:10000 + N_EVAL + N_PURITY]
    Xtr = H[tr].T.contiguous().to(DEV); Xev = H[ev].T.contiguous().to(DEV)
    Xpu = H[pu].T.contiguous().to(DEV); Tpu = T[pu]
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xev, Xpu = Xtr / sc, Xev / sc, Xpu / sc
    Y, Ypu = W @ Xev, W @ Xpu

    print(f"W {tuple(W.shape)}  K={K_ATOMS}  k={K_SPARSE}   matched across all three dictionaries\n", flush=True)

    Dl = learn(W, Xtr, K_ATOMS, K_SPARSE)
    grd = torch.Generator().manual_seed(1)
    Dr = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=grd).to(DEV), dim=0)
    print("training top-k SAE on x (activation-only baseline)...", flush=True)
    sae = train_sae(Xtr, K_ATOMS, K_SPARSE)
    Ds = Fn.normalize(sae.W_dec.detach(), dim=0)

    # ---- G1: the SAE must actually have learned something about x ----
    r_sae_x, r_rnd_x, r_lrn_x = x_r2(Ds, Xev, K_SPARSE), x_r2(Dr, Xev, K_SPARSE), x_r2(Dl, Xev, K_SPARSE)
    print(f"G1  x-reconstruction R2 (the SAE's own objective): "
          f"SAE {r_sae_x:.4f}   random {r_rnd_x:.4f}   masked-projector {r_lrn_x:.4f}", flush=True)
    assert r_sae_x > r_rnd_x + 0.05, "G1 FAIL: the SAE did not train"
    print("  [G1 ok] SAE beats random on x, so it is a fair baseline\n", flush=True)

    ch = []
    for s in range(20):
        gsel = torch.Generator().manual_seed(100 + s)
        idx = torch.randperm(Tpu.shape[0], generator=gsel)[:TOPN]
        ch.append(Counter(int(Tpu[i]) for i in idx).most_common(1)[0][1] / TOPN)
    chance = st.mean(ch)
    print(f"G2  chance purity (measured) = {chance:.3f}\n", flush=True)

    print(f"{'dictionary':22s} {'R2(Wx)':>8s} {'irrepl mean':>12s} {'irrepl top1':>12s} {'purity top10':>13s}", flush=True)
    for tag, D in (("masked-projector", Dl), ("SAE decoder (x-only)", Ds), ("random", Dr)):
        base, loss = irrepl(W, D, Y, K_SPARSE, N_PROBE)
        WD = W @ D
        Cpu = omp_chunked(WD, Ypu, K_SPARSE)
        pur = {j: purity_of(Cpu[j], Tpu) for j in loss}
        pur = {j: p for j, p in pur.items() if p is not None}
        top10 = sorted(pur, key=lambda j: -loss[j])[:10]
        mp = st.mean(pur[j] for j in top10) if top10 else float("nan")
        print(f"{tag:22s} {base:8.4f} {st.mean(loss.values()):12.6f} {max(loss.values()):12.6f} {mp:13.3f}", flush=True)
    print(f"\n  chance purity {chance:.3f}.  Toy reference: true/random irreplaceability 8.4x.", flush=True)
    print("  The spec's premise requires masked-projector > SAE decoder on R2(Wx) and irreplaceability.", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
