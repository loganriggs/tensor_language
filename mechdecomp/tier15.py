"""Tier 1.5: decompose real bilinear-model maps (tiny attn2-seed0) and test circuit
recovery against the causally-verified ground truth (L0H3 → L1H2, both K branches).

Sites:
  - OV map of each L0 head:  OV_h = W_O[:, h·dh:(h+1)·dh] @ W_V[h·dh:(h+1)·dh, :]
    (d_model → d_model, rank ≤ d_head), input data = RMSNorm'd residuals entering L0.
  - K1/K2 maps of L1H2:      W_{k1}[h2 slice] (d_head × d_model), input data =
    RMSNorm'd residuals entering L1.

Contraction (spec §1.5): strength(j → k) = d_k^{K}ᵀ · OV_h · d_j^{OV}.
GATE: aggregate |contraction| through L0H3 into L1H2-K1/K2 must dominate the other
L0 heads (causal ground truth: zeroing L0H3@src collapses L1H2's match −.434→−.031;
other heads leave it intact).

Run: TL_CORPUS=tiny python -m mechdecomp.tier15
"""

import json

import numpy as np
import torch

sys_path_hack = None
from lm_eval import load_model
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

from .objective import r2
from .tier0 import train

DEV = "cuda"
N_POINTS = 10_000
M_ATOMS = 128
LAM = 0.01


def collect_activations(model, n_windows=150):
    """Normed residual inputs to L0 and to L1 across val windows."""
    data, _ = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1] for w in range(n_windows)]).astype(np.int64)
    b = torch.from_numpy(buf).cuda()
    x = b[:, :-1]
    with torch.no_grad():
        h0 = model.embed(x)                       # input to layer 0
        h1 = model.layers[0](h0)                  # input to layer 1
        n0 = model.layers[0].norm(h0).reshape(-1, h0.shape[-1])
        n1 = model.layers[1].norm(h1).reshape(-1, h1.shape[-1])
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n0.shape[0], generator=g)[:N_POINTS]
    scale = float(n0.shape[-1]) ** 0.5            # RMSNorm'd vectors have ||x|| ≈ √d
    return (n0[idx].T.float() / scale), (n1[idx].T.float() / scale)


def decompose(W, X, tag):
    torch.cuda.empty_cache()
    D, C = train(W, X, m=M_ATOMS, lam=LAM, rounds=12, verbose=False, prune_tol=1e-4)
    print(f"  {tag}: atoms {D.shape[1]} R2 {r2(W, D, C, X):.4f} "
          f"L0 {(C.abs() > 1e-8).sum(0).float().mean():.1f}", flush=True)
    return D, C


def main():
    model = load_model(RUNS / "attn2-seed0", None, DEV)
    dh = model.layers[0].d_head
    L0, L1 = model.layers[0], model.layers[1]
    X0, X1 = collect_activations(model)
    print(f"activations: X0 {tuple(X0.shape)} X1 {tuple(X1.shape)}", flush=True)

    OV = {h: (L0.o.weight[:, h * dh:(h + 1) * dh] @ L0.v.weight[h * dh:(h + 1) * dh, :]).detach().float()
          for h in range(4)}
    K = {"K1": L1.k1.weight[2 * dh:3 * dh, :].detach().float(),
         "K2": L1.k2.weight[2 * dh:3 * dh, :].detach().float()}

    dicts = {}
    for h in range(4):
        dicts[f"OV{h}"] = decompose(OV[h], X0, f"L0H{h}-OV")
    for name, Wk in K.items():
        dicts[name] = decompose(Wk, X1, f"L1H2-{name}")

    # contraction matrices + aggregate strengths
    out = {"aggregate": {}}
    for kname in ("K1", "K2"):
        Dk, Ck = dicts[kname]
        usek = Ck.abs().sum(1); usek = usek / usek.sum()
        for h in range(4):
            Dj, Cj = dicts[f"OV{h}"]
            usej = Cj.abs().sum(1); usej = usej / usej.sum()
            G = Dk.T @ OV[h] @ Dj                 # (m_k, m_j) contraction matrix
            # usage-weighted mean |contraction|
            agg = float((usek[:, None] * usej[None, :] * G.abs()).sum())
            out["aggregate"][f"L0H{h}->{kname}"] = round(agg, 5)
    print("\nusage-weighted |contraction| aggregates:")
    for k, v in out["aggregate"].items():
        print(f"  {k}: {v}")
    k1 = [out["aggregate"][f"L0H{h}->K1"] for h in range(4)]
    k2 = [out["aggregate"][f"L0H{h}->K2"] for h in range(4)]
    print(f"\nGATE (L0H3 dominant into both branches): "
          f"K1 {'PASS' if max(k1) == k1[3] else 'FAIL'} "
          f"(H3/{'max-other'} = {k1[3] / max(k1[:3]):.2f}x) | "
          f"K2 {'PASS' if max(k2) == k2[3] else 'FAIL'} "
          f"({k2[3] / max(k2[:3]):.2f}x)")
    (RUNS / "tier15_contraction.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
