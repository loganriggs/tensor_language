"""E9: ITERATED REDUCTION — fit with L1, zero the bottom 10%, refit, repeat.  Magnitude vs ATTRIBUTION.

Logan's ask. Two prune criteria per weight w of the transcoder factors:
    magnitude    score = |w|
    attribution  score = |w * dL_fid/dw|      (how much fidelity that weight is actually buying)
Each round: fit on L_fid + L1 -> score the SURVIVING weights -> zero the bottom PRUNE_FRAC -> refit. The mask
accumulates and is never undone (iterative magnitude/attribution pruning), so the trajectory traces out a
fidelity-vs-density curve, and the surviving support IS the discovered structure.

Run in two places:
  TOY  — planted 3-sparse ground truth: we know the true support, so we can score SUPPORT RECOVERY (F1) and
         check the method actually finds the planted blocks. CONTROLS: random pruning (must be worse) and
         one-shot pruning to the same density (must be worse than iterated, or iteration buys nothing).
  REAL — the L8 bilinear MLP of the 500M bilinear GPT, under the RIDGED REAL METRIC from E8 (FINDING 6):
         Sigma_t = (1-t)*Sigma_real + t*(tr/d)*I,  with the real non-zero mean (non-central Wick).
         This is the metric F5/FINDING 10 was missing — the real input distribution has effective dim 23.5
         of 1152, so an isotropic Lambda was scoring the layer on ~1128 directions the model never visits.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, tensor_inner_mean
from e1_synthetic_recovery import make_gt, rank1_mmcs, D_IN, K_SP, DEV

torch.set_default_dtype(torch.float32)
ROUNDS, STEPS0, STEPS, PRUNE, LAM = 22, 1500, 400, 0.10, 3e-3
MET = "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders/real_metric.pt"


def ridge(Sigma, t=0.05):
    d = Sigma.shape[0]
    return (1 - t) * Sigma + t * (torch.diagonal(Sigma).sum() / d) * torch.eye(d, device=Sigma.device)


def iterate(Dg, Lg, Rg, Sig, mu, r_tc, seed, crit, rounds=ROUNDS):
    """crit: 'mag' | 'attr' | 'rand' | 'oneshot'.  Returns the (density, tensor-sim) trajectory + final factors."""
    K, d = Dg.shape[0], Lg.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed)
    Lt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K, r_tc, generator=g, device=DEV) / r_tc ** .5).requires_grad_()
    aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu).detach()
    mL = torch.ones_like(Lt); mR = torch.ones_like(Rt)          # accumulating survival masks (never undone)
    opt = torch.optim.Adam([Lt, Rt, Dt], 3e-3)
    traj = []

    def fit(n):
        for _ in range(n):
            loss = fid_loss_mean(Dg, Lg, Rg, Dt, Lt * mL, Rt * mR, Sig, mu, aa=aa) \
                 + LAM * ((Lt * mL).abs().mean() + (Rt * mR).abs().mean())
            loss.backward(); opt.step(); opt.zero_grad()

    def tsim():
        with torch.no_grad():
            return 1 - float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt * mL, Rt * mR, Sig, mu, aa=aa))

    def scores():
        """|w| or |w * dL/dw| on the SURVIVING weights."""
        if crit == "rand":
            return torch.rand_like(Lt), torch.rand_like(Rt)
        if crit in ("mag", "oneshot"):
            return (Lt * mL).abs().detach(), (Rt * mR).abs().detach()
        loss = fid_loss_mean(Dg, Lg, Rg, Dt, Lt * mL, Rt * mR, Sig, mu, aa=aa)   # ATTRIBUTION
        gL, gR = torch.autograd.grad(loss, [Lt, Rt])
        return (Lt * gL).abs().detach(), (Rt * gR).abs().detach()

    fit(STEPS0)
    traj.append((1.0, tsim()))
    target = (1 - PRUNE) ** rounds                                # the density iterated pruning will reach
    for rd in range(rounds):
        keep = target if crit == "oneshot" else (1 - PRUNE) ** (rd + 1)
        sL, sR = scores()
        s = torch.cat([sL[mL > 0].flatten(), sR[mR > 0].flatten()])   # rank only the survivors
        n_kill = int((1 - keep) * (mL.numel() + mR.numel())) - int((mL == 0).sum() + (mR == 0).sum())
        if n_kill > 0 and n_kill < s.numel():
            thr = s.kthvalue(n_kill).values
            mL = mL * (sL > thr).float(); mR = mR * (sR > thr).float()
        with torch.no_grad():
            Lt.mul_(mL); Rt.mul_(mR)
        fit(STEPS if crit != "oneshot" else STEPS * rounds if rd == rounds - 1 else 0)
        dens = float(((mL > 0).sum() + (mR > 0).sum()) / (mL.numel() + mR.numel()))
        traj.append((dens, tsim()))
        if crit == "oneshot": break
    return traj, (Dt.detach(), (Lt * mL).detach(), (Rt * mR).detach()), (mL, mR)


def support_f1(M, Lg):
    """Did we find the PLANTED support? F1 over the union of gt rows' nonzero coords (rows are gauge-permuted,
    so compare the COORDINATE-USE profile: which input coords the layer reads at all, weighted by rows)."""
    pred = (M[:, 1:].abs().sum(0) > 0).float()                    # coords used by the transcoder
    true = (Lg[:, 1:].abs().sum(0) > 1e-8).float()
    tp = float((pred * true).sum()); fp = float((pred * (1 - true)).sum()); fn = float(((1 - pred) * true).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


if __name__ == "__main__":
    print("E9  ITERATED REDUCTION: fit(L1) -> zero the bottom 10% -> refit -> repeat\n")

    # ---------------- TOY: we know the truth, so the method can be validated (and can fail) ----------------
    print("  --- TOY (planted rank-8, 3-sparse ground truth; transcoder rank 32) ---")
    d = D_IN + 1
    Sig = torch.eye(d, device=DEV); Sig[0, 0] = 0.0
    mu = torch.zeros(d, device=DEV); mu[0] = 1.0
    out = {}
    for crit in ["mag", "attr", "rand", "oneshot"]:
        tr, fac, msk = [], [], []
        for s in range(5):
            Dg, Lg, Rg = make_gt(s)
            t, f, m = iterate(Dg, Lg, Rg, Sig, mu, 32, s, crit)
            tr.append(t); fac.append(rank1_mmcs(Dg, Lg, Rg, *f)); msk.append(support_f1(f[1], Lg))
        out[crit] = (tr, np.mean(fac), np.mean(msk))
    print(f"  {'density':>9s} " + " ".join(f"{c:>12s}" for c in ["mag", "attr", "rand"]))
    n = len(out["mag"][0][0])
    for i in range(0, n, 3):
        dens = np.mean([t[i][0] for t in out["mag"][0]])
        row = " ".join(f"{np.mean([t[i][1] for t in out[c][0]]):12.3f}" for c in ["mag", "attr", "rand"])
        print(f"  {dens:9.3f} {row}")
    print(f"\n  final (density {np.mean([t[-1][0] for t in out['mag'][0]]):.3f}):")
    for c in ["mag", "attr", "rand"]:
        ts = np.mean([t[-1][1] for t in out[c][0]])
        print(f"    {c:8s}  tensor-sim {ts:6.3f}   gt-recovery {out[c][1]:.3f}   support-F1 {out[c][2]:.3f}")
    o = out["oneshot"]
    print(f"    {'oneshot':8s}  tensor-sim {np.mean([t[-1][1] for t in o[0]]):6.3f}   gt-recovery {o[1]:.3f}"
          f"   support-F1 {o[2]:.3f}   <- CONTROL: prune to the same density in ONE step")
    print(f"    (gt support = {K_SP} coords/row of {D_IN}; random-init gt-recovery ~0.066 = chance)")

    # ---------------- REAL: the same procedure, under the RIDGED REAL metric ----------------
    print("\n  --- REAL L8 bilinear MLP (r=4608, d=1152), RIDGED REAL metric (t=0.05) ---")
    M = torch.load(MET, weights_only=False)
    Sig_r = ridge(M["Sigma"].to(DEV), 0.05); mu_r = M["mu"].to(DEV)
    Dr = M["Dp"].to(DEV)
    import json as _j
    from huggingface_hub import hf_hub_download
    import jacclust.tt_model as TT
    cfg = _j.load(open(hf_hub_download("Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd", "config.json")))
    cfg.pop("step", None)
    mm = TT.GPT(TT.GPTConfig(**cfg)).eval()
    mm.load_state_dict(torch.load(hf_hub_download("Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
                                                  "pytorch_model.bin"), map_location="cpu", weights_only=True))
    Lr = mm.transformer.h[8].mlp.Left.weight.detach().float().to(DEV)
    Rr = mm.transformer.h[8].mlp.Right.weight.detach().float().to(DEV)
    print(f"  effective input dim of the real Sigma = {M['eff_dim']:.1f} of 1152  (F5 used isotropic = all 1152)")
    print(f"  {'density':>9s} {'mag':>12s} {'attr':>12s} {'rand':>12s}")
    trs = {}
    for crit in ["mag", "attr", "rand"]:
        trs[crit], _, _ = iterate(Dr, Lr, Rr, Sig_r, mu_r, 256, 0, crit, rounds=ROUNDS)
    for i in range(0, len(trs["mag"]), 3):
        print(f"  {trs['mag'][i][0]:9.3f} " + " ".join(f"{trs[c][i][1]:12.3f}" for c in ["mag", "attr", "rand"]))
    i = len(trs["mag"]) - 1
    print(f"  {trs['mag'][i][0]:9.3f} " + " ".join(f"{trs[c][i][1]:12.3f}" for c in ["mag", "attr", "rand"]))
    print("DONE")
