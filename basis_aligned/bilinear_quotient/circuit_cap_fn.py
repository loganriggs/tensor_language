# circuit_cap_fn: FALSE-NEGATIVE ANALYSIS OF THE MEMBERSHIP CLASSIFIER (S1503:
# rho .739 but FN rate .62 at rank-64). Rank sweep of the score subspace (16/64/
# 128/256): does more of the ensemble's output space capture the damage tail?
# Per-token arrays SAVED (circuit_cap_fn_tokens.npz) and the score-vs-damage
# scatter PLOTTED (plots/cap_classifier_scatter.png) per the plotting default.
# Only two causal arms needed (clean, ensemble-removal) — damage is re-measured
# once, scores are weights-only.
# Original header: TARGET-SIDE DISTRIBUTION + CLASSIFIER-GRADED GENERALIZATION
# (S1502: v1 masked on the PREVIOUS token; the certified behavior is TARGET-side.
# User: grade membership prediction by false positives and false negatives.)
# Circuit: capitalized-continuation prediction. Distribution: positions whose
# TARGET token is a capitalized word (' X...' with uppercase start, from vocab
# regex). Mechanism: the 13-head capitalization ensemble (L13-17) + mlp0's
# low-rank interaction subspace.
# Generalization leg: weight-only score s_w = fraction of lm_head row w inside the
# ensemble's OUTPUT subspace (top-64 SVD of the ensemble heads' c_proj image
# slices); ground truth = per-target-token CE rise under ensemble removal;
# graded by rank correlation + false-negative rate on the top-damaged tokens.
# Registered predictions:
#   pred_a Spearman rises monotonically with score-subspace rank (16<=64<=128<=256).
#   pred_b FN rate at rank-256 <= .45 (more output directions capture the tail).
#   pred_c FP rate at rank-256 stays <= .30 (the gain is not bought with noise).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_cap_fn_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENSEMBLE = {13: [0, 5], 14: [4, 6, 7], 15: [3], 16: [0, 3, 4, 5], 17: [0, 1, 2]}
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENS_C = {(L, h): CONSTS[f'head{L}.{h}'].to(DEV).float()
         for L, hs in ENSEMBLE.items() for h in hs}


@torch.no_grad()
def block_pat(at, xin, B):
    cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
    q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
    q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat.masked_fill(~tril, 0.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    MEANR = cl.fineweb_rows(24, skip=80)[:, :T + 1].contiguous()

    # offset-averaged patterns for the extraction background
    ACC = {L: torch.zeros(9, T, T) for L in range(18)}
    nb = 0
    for i in range(0, 24, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            ACC[L] += pat.float().mean(0).cpu()
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    KERNS = {}
    for L in range(18):
        mp = (ACC[L] / nb).to(DEV)
        kern = torch.zeros_like(mp)
        for d_ in range(T):
            idxs = torch.arange(d_, T)
            kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
        KERNS[L] = kern
    print("patterns cached", flush=True)

    # mlp0 interaction subspace (top-8, RMS-whitened composed block-1 reads)
    FR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 480, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(blk.attn, xin, idx.shape[0])
        v = blk.attn.c_v(xin).view(-1, T, 9, 128)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(v.dtype), v)
        xx = xm + blk.attn.c_proj(y.reshape(-1, T, D))
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    MU = a1 / n0
    RMS = (a2 / n0).clamp_min(1e-12).sqrt()
    at1 = H[1].attn
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    STACK = torch.cat([at1.c_q.weight.float().to(DEV) @ Wd0,
                       at1.c_k.weight.float().to(DEV) @ Wd0,
                       at1.c_q2.weight.float().to(DEV) @ Wd0,
                       at1.c_k2.weight.float().to(DEV) @ Wd0,
                       at1.c_v.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Left.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Right.weight.float().to(DEV) @ Wd0], 0)
    _, _, Vt = torch.linalg.svd(STACK * RMS.unsqueeze(0), full_matrices=False)
    W8 = Vt[:8]
    print("subspace built", flush=True)

    import tiktoken, re
    ENC = tiktoken.get_encoding('gpt2')
    CAPSET = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        s = ENC.decode([t])
        if re.match(r'^ [A-Z]', s):
            CAPSET[t] = True
    print(f"target class size: {int(CAPSET.sum())} tokens", flush=True)

    # ensemble output image (all ranks from one SVD)
    cols = []
    for L, hs in ENSEMBLE.items():
        W = H[L].attn.c_proj.weight.float().to(DEV)
        for hh in hs:
            cols.append(W[:, hh * 128:(hh + 1) * 128])
    Eimg = torch.cat(cols, 1)
    Ue, Se, _ = torch.svd_lowrank(Eimg, q=300, niter=4)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    WUn = WU.norm(dim=1).clamp_min(1e-6)

    @torch.no_grad()
    def fwd(idx, rm_ens=False, rm_sub=False, bg=False, ens_exact=False):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            if bg:
                newp = KERNS[L].unsqueeze(0).expand(B, -1, -1, -1) \
                    .to(pat.dtype).clone()
                if ens_exact and L in ENSEMBLE:
                    for hh in ENSEMBLE[L]:
                        newp[:, hh] = pat[:, hh]
                pat = newp
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            if rm_ens and L in ENSEMBLE:
                y = y.clone()
                for hh in ENSEMBLE[L]:
                    y[:, :, hh, :] = ENS_C[(L, hh)].to(y.dtype)
            x = xm + at.c_proj(y.reshape(B, T, D))
            z = F.rms_norm(x, (D,))
            if L == 0 and rm_sub:
                h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
                hw = (h0 - MU) / RMS
                comp = ((hw @ W8.T) @ W8) * RMS
                x = x + (blk.mlp(z).float() - comp @ Wd0.T).to(x.dtype)
            else:
                x = x + blk.mlp(z)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)

    def ce_run(**kw):
        s_ = 0.0; n_ = 0; sc = 0.0; nc = 0
        tsum = torch.zeros(50257); tn = torch.zeros(50257)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, **kw).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cls = CAPSET.to(DEV)[tg] & mk
            s_ += float(ce[mk & ~cls].sum()); n_ += int((mk & ~cls).sum())
            sc += float(ce[cls].sum()); nc += int(cls.sum())
            tgf = tg.cpu().reshape(-1)
            cef = (ce * (mk & cls).float()).cpu().reshape(-1)
            mkf = (mk & cls).cpu().reshape(-1).float()
            tsum.index_add_(0, tgf, cef)
            tn.index_add_(0, tgf, mkf)
        return {'global': s_ / max(n_, 1), 'cls': sc / max(nc, 1)}, tsum, tn

    r0, ts_c, tn_c = ce_run()
    r1, ts_r, tn_r = ce_run(rm_ens=True)
    print("both arms measured", flush=True)
    okk = tn_c >= 30
    rise_w = torch.where(okk, ts_r / tn_r.clamp_min(1) - ts_c / tn_c.clamp_min(1),
                         torch.zeros(50257))
    toks = torch.nonzero(okk).flatten()
    rv = rise_w[toks]

    res = {'ranks': {}}
    scores_by_rank = {}
    for rk in (16, 64, 128, 256):
        P = Ue[:, :rk]
        s_all = (WU @ P).norm(dim=1) / WUn
        sv = s_all.cpu()[toks]
        scores_by_rank[rk] = sv
        rs = torch.argsort(torch.argsort(sv)).float()
        rr = torch.argsort(torch.argsort(rv)).float()
        n = len(toks)
        rho = 1 - 6 * float(((rs - rr) ** 2).sum()) / max(n * (n * n - 1), 1)
        q75 = float(sv.quantile(0.75))
        top_dmg = rv.argsort(descending=True)[:50]
        fn = float((sv[top_dmg] < q75).float().mean())
        med_r = float(rv.median())
        top_sc = sv.argsort(descending=True)[:50]
        fp = float((rv[top_sc] <= med_r).float().mean())
        res['ranks'][rk] = {'spearman': round(rho, 3), 'fn': round(fn, 3),
                            'fp': round(fp, 3)}
        print(rk, res['ranks'][rk], flush=True)

    import numpy as np
    np.savez(PT + 'circuit_cap_fn_tokens.npz',
             tokens=toks.numpy(), rise=rv.numpy(),
             s64=scores_by_rank[64].numpy(), s256=scores_by_rank[256].numpy(),
             count=tn_c[toks].numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, rk in enumerate((64, 256)):
        sv = scores_by_rank[rk]
        top_dmg = set(rv.argsort(descending=True)[:50].tolist())
        colors = ['#b45309' if i in top_dmg else '#2563eb'
                  for i in range(len(toks))]
        ax[j].scatter(sv, rv, s=8, c=colors, alpha=.6)
        ax[j].set_xlabel(f'weights-only score (rank {rk})')
        ax[j].set_ylabel('measured CE rise under ensemble removal')
        ax[j].set_title(f"rank {rk}: rho {res['ranks'][rk]['spearman']}, "
                        f"FN {res['ranks'][rk]['fn']}", fontsize=10)
    fig.suptitle('Capitalization circuit: membership score vs measured damage '
                 '(orange = top-50 damaged)', fontsize=11)
    plt.tight_layout()
    plt.savefig(PT + 'plots/cap_classifier_scatter.png', dpi=140)
    print('plot saved', flush=True)

    rhos = [res['ranks'][rk]['spearman'] for rk in (16, 64, 128, 256)]
    pa = all(rhos[i] <= rhos[i + 1] + 1e-9 for i in range(3))
    pb = res['ranks'][256]['fn'] <= 0.45
    pc = res['ranks'][256]['fp'] <= 0.30
    out = {'summary': res, 'arm_ce': {'clean': {k: round(v, 4) for k, v in r0.items()},
                                      'rm_ens': {k: round(v, 4) for k, v in r1.items()}},
           'pred_a_rho_monotone': bool(pa), 'pred_b_fn256_le_45': bool(pb),
           'pred_c_fp256_le_30': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
