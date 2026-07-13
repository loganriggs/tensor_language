"""Feasibility test, CORRECTED. (feasibility.py v1 mis-constructed the point: it dropped the
decoder norms n_j, the per-datapoint 1/std_i from LayerNorm, and the decoder bias b_dec →
spurious R² = -2.38.)

Correct algebra. Let x = resid_pre, x̂ = SAE decode(encode(x)) = Σ_j s_j W_dec_j + b_dec.
Attention reads x̃ = ln1(x) = P x / std(x), P = centering projector (fold_ln puts γ in W_V).
Write d_j = P W_dec_j / n_j  (unit atoms),  n_j = ||P W_dec_j||. Then

    x̃_i = P x_i / std_i ≈ ( Σ_j s_ij n_j d_j  +  P b_dec ) / std_i
  ⇒ W x̃_i ≈ Σ_j (s_ij n_j / std_i) · W d_j   +   (1/std_i) · W P b_dec

The masked projector's coefficient on W d_j is c_ij·(d_jᵀ x̃_i), hence the FEASIBLE POINT is

    c_ij = s_ij · n_j / ( std_i · (d_jᵀ x̃_i) )      + one bias atom for W P b_dec

Its L0 equals the SAE's L0 (≈56). If its R² on the target W x̃ is high, then the FEATURE BASIS
is a near-optimal point of the masked-projector objective, and the "structural failure" claim
is dead — the earlier GPT-2 result (L0 294, R² 0.91, degrading solver) was optimization.

Reports, in order:
  (0) SAE sanity at its own site.
  (A) UPPER BOUND: push the SAE reconstruction through the true map — W·(P x̂/std). This is
      what any feature-basis solution can achieve; independent of the parameterization.
  (B) FEASIBLE POINT in masked-projector form, with and without the bias atom.
  (C) Found solutions for comparison: pinned-D lasso, svd-init, random dirs.
  (D) Gate-warp: plain L1 on c vs L1 on the effective coefficient c·|dᵀx|.

Run: python -m mechdecomp.feasibility2
"""

import torch
import torch.nn.functional as Fn

from .estep import solve_codes
from .objective import r2

DEV = "cuda"
LAYER = 6
K = 4096          # dictionary size (most-used SAE features)


def r2_of(Yhat, Y):
    return float(1 - ((Yhat - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


def main():
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    sae = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{LAYER}.hook_resid_pre", device=DEV)
    sae = sae[0] if isinstance(sae, tuple) else sae
    model = HookedTransformer.from_pretrained("gpt2").to(DEV)
    h_pre, h_ln1 = f"blocks.{LAYER}.hook_resid_pre", f"blocks.{LAYER}.ln1.hook_normalized"
    pre, ln1 = [], []
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(200), ds):
            ids = model.to_tokens(d["text"])[:, :128]
            if ids.shape[1] < 8:
                continue
            _, c = model.run_with_cache(ids, names_filter=[h_pre, h_ln1])
            pre.append(c[h_pre][0]); ln1.append(c[h_ln1][0])
    Xp = torch.cat(pre).float(); Xl = torch.cat(ln1).float()
    idx = torch.randperm(Xp.shape[0], generator=torch.Generator().manual_seed(0))[:20000]
    Xp, Xl = Xp[idx].to(DEV), Xl[idx].to(DEV)              # (N, d)

    with torch.no_grad():
        S = sae.encode(Xp)                                  # (N, n_feat)
        Xhat = sae.decode(S)
        print(f"(0) SAE at its own site: R2 {1 - ((Xhat-Xp)**2).sum()/((Xp-Xp.mean(0))**2).sum():.4f}"
              f"  L0 {(S>1e-6).sum(1).float().mean():.1f}", flush=True)

    b = model.blocks[LAYER].attn
    W = torch.einsum("hmd,hdn->mn", b.W_V.detach().float(), b.W_O.detach().float()).T.contiguous()
    dm = W.shape[1]
    P = torch.eye(dm, device=DEV) - torch.ones(dm, dm, device=DEV) / dm
    X = Xl.T.contiguous()                                   # (d_in, N) literal input to W
    Y = W @ X                                               # target
    print(f"W = full OV {tuple(W.shape)} rank {torch.linalg.matrix_rank(W).item()}\n", flush=True)

    # std_i recovered from the LN identity: ln1(x) = P x / std  ⇒ std_i = ||P x_i|| / ||ln1(x_i)||
    Pxp = (P @ Xp.T)                                        # (d, N)
    std = Pxp.norm(dim=0) / X.norm(dim=0).clamp_min(1e-8)   # (N,)

    # ---- (A) UPPER BOUND: SAE reconstruction pushed through the true map ----
    with torch.no_grad():
        ln1_hat = (P @ Xhat.T) / std[None, :]               # (d, N)
        print("(A) UPPER BOUND — SAE reconstruction through the true map:", flush=True)
        print(f"  W·(P·x̂/std)  vs  W·x̃ :  R2 {r2_of(W @ ln1_hat, Y):7.4f}   "
              f"L0 {(S>1e-6).sum(1).float().mean():.1f}\n", flush=True)

    # ---- (B) FEASIBLE POINT in masked-projector form ----
    W_dec = sae.W_dec.detach().float()                      # (n_feat, d)
    b_dec = sae.b_dec.detach().float()                      # (d,)
    used = torch.argsort(-(S > 1e-6).sum(0))[:K]
    PW = (P @ W_dec[used].T)                                # (d, K)
    n_j = PW.norm(dim=0).clamp_min(1e-8)
    D = (PW / n_j).contiguous()                             # unit atoms (d, K)
    A = D.T @ X                                             # gates (K, N)
    Sk = S[:, used].T.contiguous()                          # (K, N)
    # c_ij = s_ij n_j / (std_i * a_ij)   (zero where the code is zero)
    num = Sk * n_j[:, None] / std[None, :]
    C = torch.where(Sk.abs() > 1e-6, num / torch.where(A.abs() > 1e-4, A, torch.full_like(A, 1e-4)),
                    torch.zeros_like(Sk))
    Yhat_nb = (W @ D) @ (C * A)
    bias_term = (W @ (P @ b_dec))[:, None] / std[None, :]
    print("(B) FEASIBLE POINT  c = s·n/(std·gate):", flush=True)
    print(f"  masked-projector, NO bias atom     R2 {r2_of(Yhat_nb, Y):7.4f}   "
          f"L0 {(C.abs()>1e-8).sum(0).float().mean():7.1f}", flush=True)
    print(f"  + bias atom (W·P·b_dec/std)        R2 {r2_of(Yhat_nb + bias_term, Y):7.4f}   "
          f"L0 {(C.abs()>1e-8).sum(0).float().mean()+1:7.1f}\n", flush=True)

    # ---- (C) found solutions, same dictionary size ----
    print("(C) SOLVER-FOUND solutions (lam 0.02):", flush=True)
    Cp = solve_codes(W, D, X, 0.02)
    print(f"  pinned SAE-feature dict            R2 {r2(W,D,Cp,X):7.4f}   "
          f"L0 {(Cp.abs()>1e-8).sum(0).float().mean():7.1f}", flush=True)
    from .tier0 import svd_init
    Ds = svd_init(W, X, K).float()
    Cs = solve_codes(W, Ds, X, 0.02)
    print(f"  svd-init principal dirs            R2 {r2(W,Ds,Cs,X):7.4f}   "
          f"L0 {(Cs.abs()>1e-8).sum(0).float().mean():7.1f}", flush=True)
    g = torch.Generator().manual_seed(0)
    Dr = Fn.normalize(torch.randn(dm, K, generator=g).to(DEV), dim=0)
    Cr = solve_codes(W, Dr, X, 0.02)
    print(f"  random directions                  R2 {r2(W,Dr,Cr,X):7.4f}   "
          f"L0 {(Cr.abs()>1e-8).sum(0).float().mean():7.1f}\n", flush=True)

    # ---- (D) gate-warp fix ----
    print("(D) GATE-WARP (pinned feature dict):", flush=True)
    Cg = solve_codes(W, D, X, 0.02, gate_weighted_l1=True)
    print(f"  plain L1 on c                      R2 {r2(W,D,Cp,X):7.4f}   L0 {(Cp.abs()>1e-8).sum(0).float().mean():7.1f}", flush=True)
    print(f"  L1 on effective coeff c·|d·x|      R2 {r2(W,D,Cg,X):7.4f}   L0 {(Cg.abs()>1e-8).sum(0).float().mean():7.1f}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
