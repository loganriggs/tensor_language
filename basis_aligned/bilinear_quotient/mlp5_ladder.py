# mlp5_ladder: THE mlp4 DESCRIPTION LADDER WALKED TO mlp5 (§1427 inputs: attn5 + mlp4;
# §1432: the ladder format = lin2 / lin-all / +quadratic). Arms for mlp5's output
# (everything else live, held-out): lin2 = ridge from [attn5, mlp4]; linall = ridge
# from [attn5, mlp0..mlp4]; linall + r=256/F=8192 sampled-pair quadratic over
# [attn5; mlp4]. Fit skip=80, EVAL skip=7000.
#
# Registered predictions:
#   pred_a lin2 [attn5, mlp4] >= .50 of mlp5's held-out stake.
#   pred_b linall >= .60.
#   pred_c the quadratic adds >= .03 over linall.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp5_ladder_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'a5')}
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
             for L, n in ((0, 'm0'), (1, 'm1'), (2, 'm2'), (3, 'm3'), (4, 'm4'),
                          (5, 'm5'))]
    hooks.append(H[5].attn.c_proj.register_forward_hook(cap_hook_for('a5')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    X2f = torch.cat([FT['a5'], FT['m4']], -1).reshape(-1, 2 * D)
    X5f = torch.cat([FT['a5'], FT['m0'], FT['m1'], FT['m2'], FT['m3'], FT['m4']],
                    -1).reshape(-1, 6 * D)
    Yf = FT['m5'].reshape(-1, D)

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

    ARMS = [('quad_r256_F8192', 256, 8192)]
    QSPEC = {}
    for name, r, kind in ARMS:
        gq = torch.Generator().manual_seed(31 + r)
        Bj = torch.linalg.qr(torch.randn(2 * D, r, generator=gq))[0].to(DEV)
        if kind == 'squares':
            pi = torch.arange(r).to(DEV); pj = torch.arange(r).to(DEV)
        else:
            Fn = kind
            gp = torch.Generator().manual_seed(101 + Fn + r)
            pi = torch.randint(0, r, (Fn,), generator=gp).to(DEV)
            pj = torch.randint(0, r, (Fn,), generator=gp).to(DEV)
        QSPEC[name] = (Bj, pi, pj)

    def quad_feats(Ea4, Em3, name):
        Bj, pi, pj = QSPEC[name]
        Xj = torch.cat([Ea4.to(DEV), Em3.to(DEV)], -1) - xm2
        z = Xj @ Bj
        return z[:, pi] * z[:, pj]

    def fit_quad(name):
        Fn = QSPEC[name][1].shape[0]
        XtX = torch.zeros(Fn, Fn, device=DEV)
        XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        NPOS = FT['m5'].shape[0] * T
        CH = 16384
        A4 = FT['a5'].reshape(-1, D); M3 = FT['m4'].reshape(-1, D)
        for i in range(0, NPOS, CH):
            q = quad_feats(A4[i:i + CH], M3[i:i + CH], name)
            resid = (Yf[i:i + CH].to(DEV) - (ym5 + (X5f[i:i + CH].to(DEV) - xm5) @ W5))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); n += q.shape[0]
        XtX -= torch.outer(xs, xs) / n; XtY -= torch.outer(xs, ys) / n
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(Fn, device=DEV), XtY)
        return Wq, xs / n, ys / n
    QFIT = {}
    for name, _, _ in ARMS:
        QFIT[name] = fit_quad(name)
        print(f"fit {name}", flush=True)

    mh = H[5].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'a5')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                Xe5 = torch.cat([E['a5'], E['m0'], E['m1'], E['m2'], E['m3'],
                                 E['m4']], -1).reshape(-1, 6 * D).to(DEV)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin2':
                    Xe2 = torch.cat([E['a5'], E['m4']], -1).reshape(-1, 2 * D).to(DEV)
                    st = (ym2 + (Xe2 - xm2) @ W2).view(B, T, D)
                elif mode == 'linall':
                    st = lin5p.view(B, T, D)
                else:
                    A4e = E['a5'].reshape(-1, D); M3e = E['m4'].reshape(-1, D)
                    Wq, qm, rm = QFIT[mode]
                    q = quad_feats(A4e, M3e, mode)
                    st = (lin5p + rm + (q - qm) @ Wq).view(B, T, D)
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
    for mode in [None, 'mean', 'lin2', 'linall'] + [a[0] for a in ARMS]:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    r2, rall, rq = rec('lin2'), rec('linall'), rec('quad_r256_F8192')
    pa = r2 >= 0.50
    pb = rall >= 0.60
    pc = (rq - rall) >= 0.03
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin2': round(r2, 4), 'linall': round(rall, 4),
                        'quad_r256': round(rq, 4)},
           'pred_a_lin2_50': bool(pa), 'pred_b_linall_60': bool(pb),
           'pred_c_quad_adds_03': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin2 {r2:.3f} linall {rall:.3f} quad {rq:.3f} (stake {stake:.4f})")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
