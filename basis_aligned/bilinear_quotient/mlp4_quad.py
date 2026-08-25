# mlp4_quad: CHASE THE QUADRATIC RESIDUAL WITH FUNCTION-AWARE DIRECTIONS (§1428:
# lin5 .679, ~32% unexplained; §1425: variance PCs fail — use the FUNCTION's own
# directions). Input dirs = top-64 right singular vectors of the lin2 ridge map's
# per-component blocks (attn4, mlp3). Quadratic features = all pairwise products of
# the 128 projected coords (128*129/2 = 8256), ridged onto the lin5 residual.
# Arms: lin5 (refit) / +quad_full / +quad_cross (only attn4 x mlp3 products, 4096) /
# +quad_rand (same-count random-direction null). Fit skip=80, EVAL HELD OUT skip=7000.
#
# Registered predictions:
#   pred_a lin5+quad_full >= .78 of stake held-out.
#   pred_b quad_full's gain over lin5 >= 2x quad_rand's gain.
#   pred_c quad_cross alone carries >= 60% of quad_full's gain.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_quad_results.json'
NFIT = 960; NEV = 960
K = 64
H = m.transformer.h
STAND = {'mode': None, 'tensor': None}
CAP = {'on': False, 'store': None}


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def mlp4_hook(mod, args, output):
    if STAND['mode'] is None:
        return None
    return STAND['tensor'].to(output.dtype)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def capture(rows):
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'm3', 'm4', 'a4')}
    for i in range(0, rows.shape[0], 8):
        fwd(rows[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    return {n: torch.cat(v) for n, v in CAP['store'].items()}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(n))
             for L, n in ((0, 'm0'), (1, 'm1'), (2, 'm2'), (3, 'm3'), (4, 'm4'))]
    hooks.append(H[4].attn.c_proj.register_forward_hook(cap_hook_for('a4')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    X2f = torch.cat([FT['a4'], FT['m3']], -1).reshape(-1, 2 * D)
    X5f = torch.cat([FT['a4'], FT['m0'], FT['m1'], FT['m2'], FT['m3']], -1).reshape(-1, 5 * D)
    Yf = FT['m4'].reshape(-1, D)

    def ridge(X, Y):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm = Xg.mean(0); ym = Yg.mean(0)
        Xc = Xg - xm; Yc = Yg - ym
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV), Xc.T @ Yc)
        return W, xm, ym

    W2, xm2, ym2 = ridge(X2f, Yf)
    W5, xm5, ym5 = ridge(X5f, Yf)
    gmean = Yf.mean(0).to(DEV)
    print("ridges fit", flush=True)

    RQ = 64
    Ua4 = torch.linalg.svd(W2[:D], full_matrices=False)[0][:, :RQ]      # [D, 64]
    Um3 = torch.linalg.svd(W2[D:], full_matrices=False)[0][:, :RQ]
    gq = torch.Generator().manual_seed(31)
    Ra4 = torch.linalg.qr(torch.randn(D, RQ, generator=gq))[0].to(DEV)
    Rm3 = torch.linalg.qr(torch.randn(D, RQ, generator=gq))[0].to(DEV)

    def quad_feats(Ea4, Em3, Ba, Bm, cross_only=False):
        a = (Ea4.to(DEV) - xm2[:D]) @ Ba                                # [N, 64]
        b = (Em3.to(DEV) - xm2[D:]) @ Bm
        cross = torch.einsum('ni,nj->nij', a, b).reshape(a.shape[0], -1)
        if cross_only:
            return cross
        iu = torch.triu_indices(RQ, RQ)
        qa = torch.einsum('ni,nj->nij', a, a)[:, iu[0], iu[1]]
        qb = torch.einsum('ni,nj->nij', b, b)[:, iu[0], iu[1]]
        return torch.cat([cross, qa, qb], 1)

    # fit quad ridges on the lin5 residual (chunked normal equations)
    def fit_quad(Ba, Bm, cross_only):
        nfeat = RQ * RQ if cross_only else RQ * RQ + RQ * (RQ + 1)
        XtX = torch.zeros(nfeat, nfeat, device=DEV)
        XtY = torch.zeros(nfeat, D, device=DEV)
        xs = torch.zeros(nfeat, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        NPOS = FT['m4'].shape[0] * T
        CH = 16384
        A4 = FT['a4'].reshape(-1, D); M3 = FT['m3'].reshape(-1, D)
        X5 = X5f; Y = Yf
        for i in range(0, NPOS, CH):
            q = quad_feats(A4[i:i + CH], M3[i:i + CH], Ba, Bm, cross_only)
            resid = (Y[i:i + CH].to(DEV) - (ym5 + (X5[i:i + CH].to(DEV) - xm5) @ W5))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); n += q.shape[0]
        XtX -= torch.outer(xs, xs) / n; XtY -= torch.outer(xs, ys) / n
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(nfeat, device=DEV), XtY)
        return Wq, xs / n, ys / n
    Wqf, qmf, rmf = fit_quad(Ua4, Um3, False)
    Wqc, qmc, rmc = fit_quad(Ua4, Um3, True)
    Wqr, qmr, rmr = fit_quad(Ra4, Rm3, False)
    print("quad ridges fit", flush=True)

    mh = H[4].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'm3', 'm4', 'a4')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                Xe5 = torch.cat([E['a4'], E['m0'], E['m1'], E['m2'], E['m3']],
                                -1).reshape(-1, 5 * D).to(DEV)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin5':
                    st = lin5p.view(B, T, D)
                else:
                    A4e = E['a4'].reshape(-1, D); M3e = E['m3'].reshape(-1, D)
                    if mode == 'quad_full':
                        q = quad_feats(A4e, M3e, Ua4, Um3, False)
                        st = (lin5p + rmf + (q - qmf) @ Wqf).view(B, T, D)
                    elif mode == 'quad_cross':
                        q = quad_feats(A4e, M3e, Ua4, Um3, True)
                        st = (lin5p + rmc + (q - qmc) @ Wqc).view(B, T, D)
                    else:  # quad_rand
                        q = quad_feats(A4e, M3e, Ra4, Rm3, False)
                        st = (lin5p + rmr + (q - qmr) @ Wqr).view(B, T, D)
                STAND['mode'] = 'on'
                STAND['tensor'] = st
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            m_ = torch.ones_like(tg, dtype=torch.bool); m_[:, :64] = False
            s_ += float(ce[m_].sum()); n_ += int(m_.sum())
        STAND['mode'] = None
        return s_ / max(n_, 1)

    res = {}
    for mode in (None, 'mean', 'lin5', 'quad_full', 'quad_cross', 'quad_rand'):
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    r5, rf, rc, rr = rec('lin5'), rec('quad_full'), rec('quad_cross'), rec('quad_rand')
    gain_f = rf - r5; gain_c = rc - r5; gain_r = rr - r5
    pa = rf >= 0.78
    pb = gain_f >= 2.0 * max(gain_r, 1e-4)
    pc = gain_c >= 0.60 * max(gain_f, 1e-4)
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin5': round(r5, 4), 'quad_full': round(rf, 4),
                        'quad_cross': round(rc, 4), 'quad_rand': round(rr, 4)},
           'gains': {'full': round(gain_f, 4), 'cross': round(gain_c, 4),
                     'rand': round(gain_r, 4)},
           'pred_a_quad_78': bool(pa), 'pred_b_beats_null': bool(pb),
           'pred_c_cross_carries': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin5 {r5:.3f} quad {rf:.3f} cross {rc:.3f} rand {rr:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
