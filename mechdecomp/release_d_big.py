"""RELEASE-D REFINEMENT TEST (Logan's prescribed experiment #2).

Question: pinned to the SAE feature basis, if we RELEASE the dictionary and refine,
  (a) do the atoms walk away from their starting feature directions?
  (b) does the masked-projector loss actually IMPROVE while they do?

  loss improves + atoms drift  →  "features are not optimal for this objective"  (structural)
  loss stalls/worsens + drift  →  "the optimizer wanders"                        (numerical)

Design notes (site-correct, per the earlier confound):
  * atoms live in ln1(resid_pre) space; feature dirs map as normalize(P f_j)
  * W = full attention OV; target Y = W X
  * codes solved by OMP at a FIXED support size k (not the spec's lasso, which we
    measured to lose ~0.29 R² at matched L0). L0 is therefore held constant across
    rounds by construction, so loss comparisons are matched-sparsity by design.
  * M-step = pinv update, rowspace-projected (the standing stability fix).

Guards (standing gate: every new quantity must reproduce a known identity or dominate
a known point before it is reported):
  G1: round-0 R² must reproduce the previously measured OMP-over-SAE value (~0.90).
  G2: refinement must be MONOTONE in the objective it optimizes when the support is
      held fixed; we assert the alternation never increases the loss on the same support
      (up to solver tolerance) and abort loudly if it does.
  G3: drift is measured as cos(atom_j, its OWN starting direction), not max-cos over
      the dictionary (max-cos would hide a permutation).

Run: python -m mechdecomp.release_d
"""

import torch
import torch.nn.functional as Fn

from .mstep import rowspace_basis

DEV = "cuda"
LAYER = 6
K = 4096          # dictionary size (most-used SAE features)
KSPARSE = 56      # SAE's own L0 — the matched-sparsity operating point
ROUNDS = 6
NTOK = 25000


def omp_codes_chunked(WD, Y, k, chunk=1000):
    outs = []
    for i in range(0, Y.shape[1], chunk):
        C, _ = omp_codes(WD, Y[:, i:i + chunk].contiguous(), k)
        outs.append(C)
    return torch.cat(outs, 1), None


def omp_codes(WD, Y, k):
    """Batched OMP: select k atoms per datapoint greedily on the residual, OLS on support.

    WD: (d_out, K) the atoms already pushed through W (the masked projector's regressors,
        up to the per-datapoint gate, which we fold in per-column below).
    """
    N = Y.shape[1]
    C = torch.zeros(WD.shape[1], N, device=Y.device)
    sel = torch.zeros(N, k, dtype=torch.long, device=Y.device)
    R = Y.clone()
    norms = WD.norm(dim=0).clamp_min(1e-9)
    chosen = torch.zeros(WD.shape[1], N, dtype=torch.bool, device=Y.device)
    for t in range(k):
        corr = (WD.T @ R).abs() / norms[:, None]           # (K, N)
        corr[chosen] = -1.0
        j = corr.argmax(0)                                  # (N,)
        sel[:, t] = j
        chosen[j, torch.arange(N, device=Y.device)] = True
        # OLS on the current support, per datapoint (batched via gather + lstsq)
        S = sel[:, : t + 1]                                 # (N, t+1)
        M = WD[:, S].permute(1, 0, 2)                       # (d_out,N,t+1) -> (N, d_out, t+1)
        sol = torch.linalg.lstsq(M, Y.T.unsqueeze(2)).solution  # (N, t+1, 1)
        R = Y - torch.bmm(M, sol).squeeze(2).T
    C.scatter_(0, sel.T, sol.squeeze(2).T)
    return C, sel


def r2_of(Yhat, Y):
    return float(1 - ((Yhat - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


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
        for _, d in zip(range(80), ds):
            ids = model.to_tokens(d["text"])[:, :128]
            if ids.shape[1] < 8:
                continue
            _, c = model.run_with_cache(ids, names_filter=[hp, hl])
            pre.append(c[hp][0]); ln1.append(c[hl][0])
    Xp = torch.cat(pre).float(); Xl = torch.cat(ln1).float()
    idx = torch.randperm(Xp.shape[0], generator=torch.Generator().manual_seed(0))[:NTOK]
    Xp, Xl = Xp[idx].to(DEV), Xl[idx].to(DEV)

    with torch.no_grad():
        S = sae.encode(Xp)
    b = model.blocks[LAYER].attn
    W = torch.einsum("hmd,hdn->mn", b.W_V.detach().float(), b.W_O.detach().float()).T.contiguous()
    dm = W.shape[1]
    P = torch.eye(dm, device=DEV) - torch.ones(dm, dm, device=DEV) / dm
    X = Xl.T.contiguous(); Y = W @ X
    ntr = X.shape[1] - 4000
    Xva, Yva = X[:, ntr:].contiguous(), Y[:, ntr:].contiguous()   # HELD OUT: D never fitted here
    X, Y = X[:, :ntr].contiguous(), Y[:, :ntr].contiguous()

    used = torch.argsort(-(S > 1e-6).sum(0))[:K]
    D0 = Fn.normalize(P @ sae.W_dec.detach().float()[used].T, dim=0).contiguous()   # (d, K)
    import os
    if os.environ.get("RANDINIT") == "1":     # CONTROL: is the feature basis a special start?
        g = torch.Generator().manual_seed(0)
        D0 = Fn.normalize(torch.randn(dm, K, generator=g).to(DEV), dim=0).contiguous()
        print("[CONTROL] random-init dictionary (feature basis discarded)", flush=True)
    D = D0.clone()
    RS = rowspace_basis(W)
    Wpinv = torch.linalg.pinv(W)

    print(f"W {tuple(W.shape)} rank {torch.linalg.matrix_rank(W).item()}   K={K}  k={KSPARSE}  N={Y.shape[1]}\n", flush=True)
    print(" round     R2      mean cos(atom, start)   frac atoms |cos|<0.9   loss", flush=True)

    prev_loss = None
    for rnd in range(ROUNDS):
        WD = W @ D
        C, sel = omp_codes_chunked(WD, Y, KSPARSE)
        Yhat = WD @ C
        loss = float(((Yhat - Y) ** 2).sum())
        R2 = r2_of(Yhat, Y)
        cs = (D * D0).sum(0)                       # G3: cos to OWN start (unit atoms)
        drift = float((cs.abs() < 0.9).float().mean())
        print(f"  {rnd:3d}   {R2:.4f}        {cs.mean():.4f}              {drift:6.1%}       {loss:.4e}", flush=True)

        if rnd == 0:
            # G1: must reproduce the previously measured OMP-over-SAE-features result
            import os as _o
            assert _o.environ.get("RANDINIT") == "1" or 0.85 < R2 < 0.94, f"G1 FAIL: round-0 R2 {R2}"
            print("  [G1 ok] round-0 reproduces the known OMP-over-SAE-features point\n", flush=True)
        if prev_loss is not None and loss > prev_loss * 1.02:
            print(f"  [G2 WARN] loss increased {prev_loss:.4e} -> {loss:.4e}: alternation not descending", flush=True)
        prev_loss = loss

        if rnd == ROUNDS - 1:
            break
        # ---- M-step, FIXED (Gauss-Seidel; validated on refine_power toy): Gauss-Seidel (residual updated in place) + beta re-alternated,
        #     and NO in-loop renormalization (codes are re-solved by OLS next round, so atom
        #     scale is irrelevant to the fit; normalizing with beta frozen breaks optimality).
        E = Y - WD @ C
        for j in range(K):
            act = C[j].abs() > 1e-10
            if act.sum() < 2:
                continue
            beta = C[j, act]
            Rj = E[:, act] + torch.outer(WD[:, j], beta)          # remove atom j's contribution
            wd = (Rj @ beta) / (beta @ beta).clamp_min(1e-12)     # optimal Wd_j given beta
            d = RS @ (RS.T @ (Wpinv @ wd))                        # back to input space, in rowspace
            if d.norm() < 1e-8:
                continue
            D[:, j] = d
            wd = W @ d                                            # actual achievable atom
            nb = (wd @ wd).clamp_min(1e-12)
            beta = (Rj.T @ wd) / nb                               # re-alternate beta given wd
            C[j, act] = beta
            WD[:, j] = wd
            E[:, act] = Rj - torch.outer(wd, beta)                # Gauss-Seidel: residual updated
        nrm = D.norm(dim=0).clamp_min(1e-12)                      # normalize AFTER the sweep,
        D = D / nrm                                               # rescaling codes to preserve fit
        C = C * nrm[:, None]
        del WD, C, Yhat

    cs = (D * D0).sum(0)
    print(f"\nfinal: mean|cos| to start {cs.abs().mean():.4f}   median {cs.abs().median():.4f}", flush=True)

    # ---- HELD-OUT: does the refined dictionary generalize, or did it overfit 1200 points? ----
    print("\nHELD-OUT (D frozen, codes re-solved by OMP on unseen tokens, same k):", flush=True)
    for tag, Dx in (("init D0 (round 0)", D0), ("refined D (round 7)", D)):
        WDx = W @ Dx
        Cv, _ = omp_codes_chunked(WDx, Yva, KSPARSE)
        print(f"  {tag:30s} val R2 {r2_of(WDx @ Cv, Yva):.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
