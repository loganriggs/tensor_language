"""Adjudicating experiment (Logan, 2026-07-09): is the FEATURE BASIS a near-optimal point
of the masked-projector objective? If yes, the "structural failure" verdict is dead and the
GPT-2 result was an optimization/initialization failure.

SITE CORRECTNESS (the confound the earlier run missed):
  res-jb SAE features live at blocks.6.hook_resid_pre.
  A map's atoms must live in THAT map's input space.
  Layer-6 attention reads ln1(resid_pre); with TL's fold_ln the learnable scale is already
  folded into W_V, so the remaining op is  ln1(x) = (x - mean(x)) / std(x) = P x / std(x),
  with P = I - (1/d)·11ᵀ a FIXED linear centering projector and 1/std a per-datapoint scalar.
  ⇒ Work in ln1-normalized space:  X̃ = ln1_normalized  (the literal input to W_V).
     Feature directions map linearly:  d_j = normalize(P f_j).
     The per-datapoint 1/std is absorbed into the code c_ij (codes are free).
  W = full attention OV (all heads concatenated) = W_O @ W_V  — high rank, reads X̃.

TESTS
  (0) sanity: SAE reconstructs resid_pre (its own site) at the published L0/R².
  (1) EXPLICIT FEASIBLE POINT (Logan's construction): c_ij = s_ij / (d_jᵀ x̃_i), no solving.
      Measures R²(Wx̃) and L0 of the feature basis as a masked-projector solution.
  (2) PINNED-D E-step: D = feature dirs held fixed, codes solved by the lasso. Does the
      optimizer find a sparse high-R² solution when handed the right dictionary?
  (3) BASELINES: relevance-blind random SAE features; svd-init principal dirs; random dirs.
  (4) GATE-WARP FIX: L1 on the EFFECTIVE coefficient c·|dᵀx| rather than on c
      (the objective's real bias against feature dirs with modest projections).

Run: python -m mechdecomp.feasibility
"""

import torch
import torch.nn.functional as Fn

from .estep import solve_codes
from .objective import predict, r2

DEV = "cuda"
LAYER = 6


def collect(n_docs=200, n_tok=20000):
    from datasets import load_dataset
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2").to(DEV)
    h_pre, h_ln1 = f"blocks.{LAYER}.hook_resid_pre", f"blocks.{LAYER}.ln1.hook_normalized"
    pre, ln1 = [], []
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = model.to_tokens(d["text"])[:, :128]
            if ids.shape[1] < 8:
                continue
            _, c = model.run_with_cache(ids, names_filter=[h_pre, h_ln1])
            pre.append(c[h_pre][0]); ln1.append(c[h_ln1][0])
    P = torch.cat(pre).float(); L = torch.cat(ln1).float()
    idx = torch.randperm(P.shape[0], generator=torch.Generator().manual_seed(0))[:n_tok]
    return model, P[idx].contiguous(), L[idx].contiguous()


def full_ov(model):
    """W_O @ W_V concatenated over heads: (d_model, d_model), reads ln1(resid_pre)."""
    b = model.blocks[LAYER].attn
    W_V = b.W_V.detach().float()            # (n_head, d_model, d_head)
    W_O = b.W_O.detach().float()            # (n_head, d_head, d_model)
    return torch.einsum("hmd,hdn->mn", W_V, W_O).T.contiguous()   # (d_out, d_in)


def report(tag, W, D, C, X):
    R = r2(W, D, C, X)
    L0 = (C.abs() > 1e-8).sum(0).float().mean().item()
    print(f"  {tag:44s} R2 {R:7.4f}   L0 {L0:7.1f}   atoms {D.shape[1]}", flush=True)
    return R, L0


def main():
    from sae_lens import SAE
    sae = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{LAYER}.hook_resid_pre", device=DEV)
    sae = sae[0] if isinstance(sae, tuple) else sae
    model, Xpre, Xln1 = collect()
    Xpre, Xln1 = Xpre.to(DEV), Xln1.to(DEV)

    # (0) sanity: SAE at its own site
    with torch.no_grad():
        S = sae.encode(Xpre)                       # (N, n_feat) codes
        rec = sae.decode(S)
        r2_sae = 1 - ((rec - Xpre) ** 2).sum() / ((Xpre - Xpre.mean(0)) ** 2).sum()
        l0_sae = (S > 1e-6).sum(1).float().mean()
    print(f"(0) SAE at its own site: R2 {r2_sae:.4f}  L0 {l0_sae:.1f}\n", flush=True)

    W = full_ov(model)
    print(f"W = full OV {tuple(W.shape)} rank {torch.linalg.matrix_rank(W).item()}", flush=True)
    X = Xln1.T.contiguous()                        # (d_in, N) — literal input to W

    d = X.shape[0]
    P = torch.eye(d, device=DEV) - torch.ones(d, d, device=DEV) / d
    Fdir = Fn.normalize(sae.W_dec.detach().float() @ P.T, dim=1)   # centered feature dirs
    live = torch.where((S > 1e-6).sum(0) > 20)[0]                  # features used on this data
    print(f"live SAE features on this data: {len(live)}\n", flush=True)

    # ---- (1) EXPLICIT FEASIBLE POINT: c_ij = s_ij / (d_j · x̃_i) ----
    K = 2048
    order = live[torch.argsort(-(S[:, live] > 1e-6).sum(0))][:K]    # most-used features
    D = Fdir[order].T.contiguous()                                  # (d_in, K)
    A = D.T @ X                                                     # gates (K, N)
    Sk = S[:, order].T.contiguous()                                 # (K, N) SAE codes
    Cfeas = torch.where(Sk.abs() > 1e-6, Sk / torch.where(A.abs() > 1e-3, A, torch.full_like(A, float("nan"))), torch.zeros_like(Sk))
    Cfeas = torch.nan_to_num(Cfeas, nan=0.0, posinf=0.0, neginf=0.0)
    print("(1) EXPLICIT FEASIBLE POINT (SAE codes / gates, no solving):", flush=True)
    report("feature basis, c=s/(d·x)", W, D, Cfeas, X)

    # ---- (2) PINNED-D E-step (dictionary fixed to features, codes solved) ----
    print("\n(2) PINNED-D: dictionary = SAE features, codes solved by lasso:", flush=True)
    for lam in (0.005, 0.02, 0.05):
        C = solve_codes(W, D, X, lam)
        report(f"pinned SAE-feature dict (lam {lam})", W, D, C, X)

    # ---- (3) BASELINES ----
    print("\n(3) BASELINES (same K, codes solved, lam 0.02):", flush=True)
    g = torch.Generator().manual_seed(0)
    rnd_feat = Fdir[live[torch.randperm(len(live), generator=g)[:K]]].T.contiguous()
    report("random SAE features (relevance-blind)", W, rnd_feat, solve_codes(W, rnd_feat, X, 0.02), X)
    from .tier0 import svd_init
    Dsvd = svd_init(W, X, K).float()
    report("svd-init principal dirs", W, Dsvd, solve_codes(W, Dsvd, X, 0.02), X)
    Drnd = Fn.normalize(torch.randn(d, K, generator=g).to(DEV), dim=0)
    report("random directions", W, Drnd, solve_codes(W, Drnd, X, 0.02), X)

    # ---- (4) GATE-WARP FIX: L1 on effective coefficient c·|d·x| ----
    print("\n(4) GATE-WARPED L1 (penalize c only) vs EFFECTIVE-COEFF L1 (penalize c·|d·x|):", flush=True)
    C_std = solve_codes(W, D, X, 0.02)
    report("pinned dict, plain L1 on c", W, D, C_std, X)
    C_eff = solve_codes(W, D, X, 0.02, gate_weighted_l1=True)
    report("pinned dict, L1 on effective coeff", W, D, C_eff, X)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
