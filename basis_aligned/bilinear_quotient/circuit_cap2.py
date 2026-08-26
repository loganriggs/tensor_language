# circuit_cap2: TARGET-SIDE DISTRIBUTION + CLASSIFIER-GRADED GENERALIZATION
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
#   pred_a with the TARGET-side class, ensemble removal is selective: class rise
#          >= 3x global rise.
#   pred_b extraction on the simplified background recovers class >= 2x global.
#   pred_c classifier: Spearman(s_w, per-token rise) >= .3 over targets with >= 30
#          class occurrences, AND false-negative rate <= .5 (of the 50 most-damaged
#          target tokens, at least half sit in the top quartile of s_w).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_cap2_results.json'
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

    # ensemble OUTPUT subspace from weights: c_proj image slices of ensemble heads
    cols = []
    for L, hs in ENSEMBLE.items():
        W = H[L].attn.c_proj.weight.float().to(DEV)      # [D, D] (in = 9*128)
        for hh in hs:
            cols.append(W[:, hh * 128:(hh + 1) * 128])
    Eimg = torch.cat(cols, 1)                            # [D, 13*128]
    Ue, Se, _ = torch.svd_lowrank(Eimg, q=80, niter=4)
    P64 = Ue[:, :64]                                     # output subspace basis
    WU = m.lm_head.weight.float().to(DEV)                # [V, D]
    s_w = (WU @ P64).norm(dim=1) / WU.norm(dim=1).clamp_min(1e-6)
    print("weight scores built", flush=True)

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

    res = {}
    PT_STORE = {}
    for nm, kw in (('clean', {}), ('rm_ens', {'rm_ens': True}),
                   ('rm_sub', {'rm_sub': True}),
                   ('rm_both', {'rm_ens': True, 'rm_sub': True}),
                   ('bg', {'bg': True}),
                   ('bg_ens', {'bg': True, 'ens_exact': True})):
        r_, tsum, tn = ce_run(**kw)
        res[nm] = {k: round(v, 4) for k, v in r_.items()}
        if nm in ('clean', 'rm_ens'):
            PT_STORE[nm] = (tsum, tn)
        print(nm, res[nm], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    c = res['clean']
    rise = lambda nm, k: res[nm][k] - c[k]
    rec = lambda k: (res['bg'][k] - res['bg_ens'][k]) / max(res['bg'][k] - c[k], 1e-6)

    # per-target-token rise (>= 30 class occurrences)
    ts_c, tn_c = PT_STORE['clean']; ts_r, tn_r = PT_STORE['rm_ens']
    okk = tn_c >= 30
    rise_w = torch.where(okk, ts_r / tn_r.clamp_min(1) - ts_c / tn_c.clamp_min(1),
                         torch.zeros(50257))
    toks = torch.nonzero(okk).flatten()
    sv = s_w.cpu()[toks]; rv = rise_w[toks]
    rs = torch.argsort(torch.argsort(sv)).float()
    rr = torch.argsort(torch.argsort(rv)).float()
    n = len(toks)
    rho = 1 - 6 * float(((rs - rr) ** 2).sum()) / max(n * (n * n - 1), 1)
    top_damaged = toks[rv.argsort(descending=True)[:50]]
    q75 = float(s_w.cpu()[toks].quantile(0.75))
    fn_rate = float((s_w.cpu()[top_damaged] < q75).float().mean())
    top_score = toks[sv.argsort(descending=True)[:50]]
    med_rise = float(rv.median())
    fp_rate = float((rise_w[top_score] <= med_rise).float().mean())

    pa = rise('rm_ens', 'cls') >= 3 * max(rise('rm_ens', 'global'), 1e-6)
    pb = rec('cls') >= 2 * max(rec('global'), 1e-6) and rec('global') > 0
    pc = rho >= 0.3 and fn_rate <= 0.5
    out = {'res': res,
           'rises': {nm: {k: round(rise(nm, k), 4) for k in c}
                     for nm in ('rm_ens', 'rm_sub', 'rm_both')},
           'extraction_recovery': {k: round(rec(k), 4) for k in c},
           'classifier': {'spearman': round(rho, 3), 'n_tokens': n,
                          'fn_rate_top50_damaged': round(fn_rate, 3),
                          'fp_rate_top50_scored': round(fp_rate, 3)},
           'pred_a_removal_selective_3x': bool(pa),
           'pred_b_extraction_selective_2x': bool(pb),
           'pred_c_classifier': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"rises {out['rises']}")
    print(f"classifier {out['classifier']}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
