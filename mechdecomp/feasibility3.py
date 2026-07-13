"""Feasibility, exact + MATCHED-L0 (the comparison that actually adjudicates).

Fixes in v3:
  * v2's (B) still divided by near-zero gates (c = s·n/(std·a), clamped) and truncated to
    K=4096 features, so it did NOT reproduce (A). But the objective only ever uses the
    PRODUCT c_ij·a_ij. So realize the feasible point by setting that product directly:
        (C ⊙ A)_ij := s_ij · n_j / std_i          over ALL features
    Then Ŷ = (W D)(C⊙A) = W·(P(x̂ − b_dec)/std) exactly; +bias atom ⇒ equals (A).
  * The decisive test is R² at MATCHED L0. The feature basis sits at L0≈56; the solver's
    0.969 sits at L0≈565. Sweep λ so every dictionary is evaluated at comparable L0, then
    compare R². If the feature basis DOMINATES at matched L0, the "structural" claim dies:
    a better sparse solution exists and the optimizer never finds it.

Run: python -m mechdecomp.feasibility3
"""

import torch
import torch.nn.functional as Fn

from .estep import solve_codes
from .objective import r2

DEV = "cuda"
LAYER = 6


def r2_of(Yhat, Y):
    return float(1 - ((Yhat - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


def sweep(W, D, X, tag, targets=(57, 120, 300), lams=(20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.05)):
    """R² at (approximately) matched L0 values."""
    got = []
    for lam in lams:
        C = solve_codes(W, D, X, lam)
        L0 = (C.abs() > 1e-8).sum(0).float().mean().item()
        got.append((L0, r2(W, D, C, X), lam))
    pts = "  ".join(f"L0 {l:.0f}:R2 {r:.3f}" for l, r, _ in sorted(got))
    print(f"  {tag:30s} frontier → {pts}", flush=True)
    for t in targets:
        best = min(got, key=lambda z: abs(z[0] - t))
        print(f"      nearest L0 {t:4d} → L0 {best[0]:6.1f}  R2 {best[1]:7.4f}  (λ {best[2]})", flush=True)
    return got


def main():
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    sae = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{LAYER}.hook_resid_pre", device=DEV)
    sae = sae[0] if isinstance(sae, tuple) else sae
    model = HookedTransformer.from_pretrained("gpt2").to(DEV)
    hp, hl = f"blocks.{LAYER}.hook_resid_pre", f"blocks.{LAYER}.ln1.hook_normalized"
    pre, ln1 = [], []
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(200), ds):
            ids = model.to_tokens(d["text"])[:, :128]
            if ids.shape[1] < 8:
                continue
            _, c = model.run_with_cache(ids, names_filter=[hp, hl])
            pre.append(c[hp][0]); ln1.append(c[hl][0])
    Xp = torch.cat(pre).float(); Xl = torch.cat(ln1).float()
    idx = torch.randperm(Xp.shape[0], generator=torch.Generator().manual_seed(0))[:12000]
    Xp, Xl = Xp[idx].to(DEV), Xl[idx].to(DEV)

    with torch.no_grad():
        S = sae.encode(Xp); Xhat = sae.decode(S)
    l0_sae = (S > 1e-6).sum(1).float().mean().item()
    print(f"(0) SAE at own site: R2 {1-((Xhat-Xp)**2).sum()/((Xp-Xp.mean(0))**2).sum():.4f}  L0 {l0_sae:.1f}\n", flush=True)

    b = model.blocks[LAYER].attn
    W = torch.einsum("hmd,hdn->mn", b.W_V.detach().float(), b.W_O.detach().float()).T.contiguous()
    dm = W.shape[1]
    P = torch.eye(dm, device=DEV) - torch.ones(dm, dm, device=DEV) / dm
    X = Xl.T.contiguous(); Y = W @ X
    std = (P @ Xp.T).norm(dim=0) / X.norm(dim=0).clamp_min(1e-8)

    # ---- (A) upper bound ----
    ub = r2_of(W @ ((P @ Xhat.T) / std[None, :]), Y)
    print(f"(A) UPPER BOUND  W·(P x̂/std) vs W·x̃ :  R2 {ub:.4f}  at L0 {l0_sae:.1f}\n", flush=True)

    # ---- (B) EXACT feasible point: set the product (C⊙A) directly, ALL features ----
    W_dec = sae.W_dec.detach().float(); b_dec = sae.b_dec.detach().float()
    PW = (P @ W_dec.T); n_j = PW.norm(dim=0).clamp_min(1e-8)
    Dfull = (PW / n_j).contiguous()                      # (d, n_feat) unit atoms
    # chunked over datapoints: the (n_feat, N) effective-coeff matrix is too big to hold
    WD = W @ Dfull                                        # (d_out, n_feat)
    bias_vec = W @ (P @ b_dec)                            # (d_out,)
    Yhat = torch.empty_like(Y); bias = torch.empty_like(Y); l0_sum = 0.0
    CH = 2000
    for i in range(0, Y.shape[1], CH):
        sl = slice(i, min(i + CH, Y.shape[1]))
        Mc = (S[sl].T * n_j[:, None]) / std[sl][None, :]  # (n_feat, chunk)
        Yhat[:, sl] = WD @ Mc
        bias[:, sl] = bias_vec[:, None] / std[sl][None, :]
        l0_sum += (Mc.abs() > 1e-8).sum(0).float().sum().item()
        del Mc
    l0_feas = l0_sum / Y.shape[1]
    print("(B) EXACT feasible point (product set directly, all features):", flush=True)
    print(f"  no bias atom   R2 {r2_of(Yhat, Y):7.4f}  L0 {l0_feas:6.1f}", flush=True)
    print(f"  + bias atom    R2 {r2_of(Yhat + bias, Y):7.4f}  L0 {l0_feas+1:6.1f}   (should equal (A))\n", flush=True)

    # ---- (C) MATCHED-L0 comparison ----
    K = 2048
    used = torch.argsort(-(S > 1e-6).sum(0))[:K]
    Dfeat = Dfull[:, used].contiguous()
    from .tier0 import svd_init
    Dsvd = svd_init(W, X, K).float().contiguous()
    g = torch.Generator().manual_seed(0)
    Drnd = Fn.normalize(torch.randn(dm, K, generator=g).float().to(DEV), dim=0)
    print("(C) MATCHED-L0: R² of each dictionary at comparable sparsity", flush=True)
    sweep(W, Dfeat, X, "SAE-feature dict (pinned)")
    sweep(W, Dsvd, X, "svd-init principal dirs")
    sweep(W, Drnd, X, "random directions")
    print(f"\n  reference: EXACT feature-basis point  L0 {l0_feas+1:.0f}  R2 {r2_of(Yhat+bias, Y):.4f}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
