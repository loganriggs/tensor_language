# circuit_digits: FULL THREE-PROPERTY SUITE FOR THE DIGITS CIRCUIT, WITH THE TWO
# S1504/05 FIXES: (1) ensemble = top-8 heads by the WEIGHTS-ONLY score computed
# in-script (the method that won the screen); (2) extraction background = whitened
# rank-32 QK at ALL layers (median fid .92) instead of offset-averaged patterns —
# the S1504 non-vacuity failure was the crude background corrupting the restored
# heads' inputs. Class: digit targets (multi-token, so the classifier leg is
# non-vacuous). Distribution: positions whose TARGET token is a number.
# Registered predictions:
#   pred_a 8-head removal selectivity >= 5x on digit targets.
#   pred_b extraction on the rank-32 background: class recovery >= 2x global AND
#          global recovery > 0 (the background fix works).
#   pred_c classifier Spearman >= .4 over digit tokens with >= 30 occurrences.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_digits_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENSEMBLE = None  # filled in-script from the weights-only score (top-8)
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENS_C = {}   # filled after the ensemble is chosen


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
        if re.match(r'^ ?[0-9]+$', ENC.decode([t])):
            CAPSET[t] = True
    print(f"target class size: {int(CAPSET.sum())} tokens", flush=True)

    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u_cls = WU[CAPSET.to(DEV)].mean(0)
    u_cls = u_cls / u_cls.norm()
    sc = torch.zeros(18, 9)
    for L in range(18):
        W = H[L].attn.c_proj.weight.float().to(DEV)
        for hh in range(9):
            sc[L, hh] = float((u_cls @ W[:, hh * 128:(hh + 1) * 128]).norm())
    flat = sc.flatten().argsort(descending=True)[:8]
    global ENSEMBLE
    ENSEMBLE = {}
    for i in flat:
        L, hh = int(i) // 9, int(i) % 9
        ENSEMBLE.setdefault(L, []).append(hh)
        ENS_C[(L, hh)] = CONSTS[f'head{L}.{hh}'].to(DEV).float()
    print("ensemble:", {L: hs for L, hs in ENSEMBLE.items()}, flush=True)

    # membership score for the classifier leg (rank-64 output subspace)
    cols = []
    for L, hs in ENSEMBLE.items():
        W = H[L].attn.c_proj.weight.float().to(DEV)
        for hh in hs:
            cols.append(W[:, hh * 128:(hh + 1) * 128])
    Eimg = torch.cat(cols, 1)
    Ue, _, _ = torch.svd_lowrank(Eimg, q=80, niter=4)
    P64 = Ue[:, :64]
    s_w = (WU @ P64).norm(dim=1) / WU.norm(dim=1).clamp_min(1e-6)

    # rank-32 whitened QK background for ALL layers (extraction arms)
    CR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    XACC = {L: torch.zeros(D, D, device=DEV) for L in range(18)}
    ncov = 0
    for i in range(0, 96, 8):
        idx = CR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            Xf = xin.float().reshape(-1, D)
            XACC[L] += Xf.T @ Xf
            x, v1_ = blk(x, v1_, x0)
        ncov += idx.shape[0] * T
    def trunc_perhead(W, r_, Wh, Whi):
        Wf = W.float().to(DEV).view(9, 128, D)
        out = torch.zeros_like(Wf)
        for h in range(9):
            U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
            out[h] = ((U[:, :r_] * S[:r_]) @ Vt[:r_]) @ Whi
        return out.view(9 * 128, D)
    TWALL = {}
    for L in range(18):
        Sg = XACC[L] / ncov
        ev, Vv = torch.linalg.eigh(Sg)
        ev = ev.clamp_min(1e-6)
        Wh = Vv @ torch.diag(ev.sqrt()) @ Vv.T
        Whi = Vv @ torch.diag(ev.rsqrt()) @ Vv.T
        at = H[L].attn
        TWALL[L] = {'q': trunc_perhead(at.c_q.weight, 32, Wh, Whi),
                    'k': trunc_perhead(at.c_k.weight, 32, Wh, Whi),
                    'q2': trunc_perhead(at.c_q2.weight, 32, Wh, Whi),
                    'k2': trunc_perhead(at.c_k2.weight, 32, Wh, Whi)}
        print(f"bg maps L{L}", flush=True)

    @torch.no_grad()
    def fwd(idx, rm_ens=False, rm_sub=False, bg=False, ens_exact=False):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            if bg:
                TW = TWALL[L]
                qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
                kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
                q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
                k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
                if ens_exact and L in ENSEMBLE:
                    qf = at.c_q(xin).view(B, T, 9, 128).float()
                    kf = at.c_k(xin).view(B, T, 9, 128).float()
                    q2f = at.c_q2(xin).view(B, T, 9, 128).float()
                    k2f = at.c_k2(xin).view(B, T, 9, 128).float()
                    for hh in ENSEMBLE[L]:
                        qp[:, :, hh] = qf[:, :, hh]; kp[:, :, hh] = kf[:, :, hh]
                        q2p[:, :, hh] = q2f[:, :, hh]; k2p[:, :, hh] = k2f[:, :, hh]
            else:
                qp = at.c_q(xin).view(B, T, 9, 128).float()
                kp = at.c_k(xin).view(B, T, 9, 128).float()
                q2p = at.c_q2(xin).view(B, T, 9, 128).float()
                k2p = at.c_k2(xin).view(B, T, 9, 128).float()
            cos, sin = at.rotary(qp)
            q = are(F.rms_norm(qp, (128,)), cos, sin)
            k = are(F.rms_norm(kp, (128,)), cos, sin)
            q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
            k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
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
            x = x + blk.mlp(F.rms_norm(x, (D,)))
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

    pa = rise('rm_ens', 'cls') >= 5 * max(rise('rm_ens', 'global'), 1e-6)
    pb = rec('cls') >= 2 * max(rec('global'), 1e-6) and rec('global') > 0
    pc = rho >= 0.4
    out = {'res': res,
           'rises': {nm: {k: round(rise(nm, k), 4) for k in c}
                     for nm in ('rm_ens',)},
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
