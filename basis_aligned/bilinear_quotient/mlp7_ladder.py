# mlp7_ladder: THE LADDER WALKED TO mlp7 (board #3: .056 CE unexplained, NO stand-in
# yet). Same template as mlp4/5/6 (S1432 format): arms for mlp7's output (everything
# else live, held-out): lin2 = ridge from [attn7, mlp6]; linall = ridge from
# [attn7, mlp0..mlp6]; linall + r=256/F=8192 sampled-pair quadratic over [attn7; mlp6].
# Fit skip=80, EVAL skip=7000. Frozen anchors from the 198-sweep score fid_opt too.
#
# Registered predictions (mirroring the mid-ladder pattern at 5/6):
#   pred_a lin2 [attn7, mlp6] >= .50 of mlp7's held-out stake.
#   pred_b linall >= .60.
#   pred_c the quadratic adds >= .03 over linall.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp7_ladder_results.json'
NFIT = 960; NEV = 960
H = m.transformer.h
STAND = {'mode': None, 'tensor': None}
CAP = {'on': False, 'store': None}
NAMES = ('m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'a7')


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def stand_hook(mod, args, output):
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in NAMES}
    for i in range(0, rows.shape[0], 8):
        fwd(rows[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    return {n: torch.cat(v) for n, v in CAP['store'].items()}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(f'm{L}')) for L in range(8)]
    hooks.append(H[7].attn.c_proj.register_forward_hook(cap_hook_for('a7')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    MLPS = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']
    X2f = torch.cat([FT['a7'], FT['m6']], -1).reshape(-1, 2 * D)
    XAf = torch.cat([FT['a7']] + [FT[n] for n in MLPS], -1).reshape(-1, 8 * D)
    Yf = FT['m7'].reshape(-1, D)

    def ridge(X, Y):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm = Xg.mean(0); ym = Yg.mean(0)
        Xc = Xg - xm; Yc = Yg - ym
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV), Xc.T @ Yc)
        return W, xm, ym

    W2, xm2, ym2 = ridge(X2f, Yf)
    WA, xmA, ymA = ridge(XAf, Yf)
    gmean = Yf.mean(0).to(DEV)
    print("ridges fit", flush=True)

    QN = 'quad_r256_F8192'
    gq = torch.Generator().manual_seed(31 + 256)
    Bj = torch.linalg.qr(torch.randn(2 * D, 256, generator=gq))[0].to(DEV)
    gp = torch.Generator().manual_seed(101 + 8192 + 256)
    pi = torch.randint(0, 256, (8192,), generator=gp).to(DEV)
    pj = torch.randint(0, 256, (8192,), generator=gp).to(DEV)

    def quad_feats(Ea, Em):
        Xj = torch.cat([Ea.to(DEV), Em.to(DEV)], -1) - xm2
        z = Xj @ Bj
        return z[:, pi] * z[:, pj]

    def fit_quad():
        Fn = 8192
        XtX = torch.zeros(Fn, Fn, device=DEV)
        XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        NPOS = FT['m7'].shape[0] * T
        CH = 16384
        Aa = FT['a7'].reshape(-1, D); Mm = FT['m6'].reshape(-1, D)
        for i in range(0, NPOS, CH):
            q = quad_feats(Aa[i:i + CH], Mm[i:i + CH])
            resid = (Yf[i:i + CH].to(DEV) - (ymA + (XAf[i:i + CH].to(DEV) - xmA) @ WA))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); n += q.shape[0]
        XtX -= torch.outer(xs, xs) / n; XtY -= torch.outer(xs, ys) / n
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(Fn, device=DEV), XtY)
        return Wq, xs / n, ys / n

    QFIT = fit_quad()
    print("quad fit", flush=True)

    mh = H[7].mlp.register_forward_hook(stand_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in NAMES}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                XeA = torch.cat([E['a7']] + [E[n] for n in MLPS],
                                -1).reshape(-1, 8 * D).to(DEV)
                linAp = ymA + (XeA - xmA) @ WA
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin2':
                    Xe2 = torch.cat([E['a7'], E['m6']], -1).reshape(-1, 2 * D).to(DEV)
                    st = (ym2 + (Xe2 - xm2) @ W2).view(B, T, D)
                elif mode == 'linall':
                    st = linAp.view(B, T, D)
                else:
                    Wq, qm, rm = QFIT
                    q = quad_feats(E['a7'].reshape(-1, D), E['m6'].reshape(-1, D))
                    st = (linAp + rm + (q - qm) @ Wq).view(B, T, D)
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
    for mode in [None, 'mean', 'lin2', 'linall', QN]:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp7']
    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    fid = lambda a: (sw['ce_opt'] - res[a]) / max(sw['ce_opt'] - res['None'], 1e-6)
    r2, rall, rq = rec('lin2'), rec('linall'), rec(QN)
    pa = r2 >= 0.50
    pb = rall >= 0.60
    pc = (rq - rall) >= 0.03
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin2': round(r2, 4), 'linall': round(rall, 4),
                        'quad_r256': round(rq, 4)},
           'fid_opt': {k: round(fid(k), 4) for k in ('lin2', 'linall', QN)},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_lin2_50': bool(pa), 'pred_b_linall_60': bool(pb),
           'pred_c_quad_adds_03': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin2 {r2:.3f} linall {rall:.3f} quad {rq:.3f} (stake {stake:.4f})")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
