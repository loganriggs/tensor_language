# mlp4_quad_curve: CAPACITY OR DIRECTION? (§1429: the quadratic residual is
# direction-agnostic — random 64-dim projections beat function-aware bases.) Sweep A:
# direction-pool rank r in {32, 64, 128, 256} random joint dims over centered
# [attn4; mlp3], feature count FIXED at F=8192 sampled index-pair products. Sweep B:
# r fixed at 64, F in {4096, 8192, 16384}. All ridged onto the lin5 residual; fit
# skip=80, EVAL HELD OUT skip=7000.
#
# Registered predictions:
#   pred_a isotropy: recovery FLAT in r at fixed F (max-min <= .02 across sweep A).
#   pred_b capacity-limited: each F doubling at r=64 gains >= .015 recovery.
#   pred_c r=64 / F=16384 reaches >= .77.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_quad_curve_results.json'
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

    ARMS = [('r32_F8192', 32, 8192), ('r64_F8192', 64, 8192),
            ('r128_F8192', 128, 8192), ('r256_F8192', 256, 8192),
            ('r64_F4096', 64, 4096), ('r64_F16384', 64, 16384)]
    QSPEC = {}
    for name, r, Fn in ARMS:
        gq = torch.Generator().manual_seed(31 + r)
        Bj = torch.linalg.qr(torch.randn(2 * D, r, generator=gq))[0].to(DEV)
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
        NPOS = FT['m4'].shape[0] * T
        CH = 16384
        A4 = FT['a4'].reshape(-1, D); M3 = FT['m3'].reshape(-1, D)
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
    for mode in [None, 'mean', 'lin5'] + [a[0] for a in ARMS]:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    recs = {a[0]: round(rec(a[0]), 4) for a in ARMS}
    r5 = rec('lin5')
    sweepA = [recs['r32_F8192'], recs['r64_F8192'], recs['r128_F8192'], recs['r256_F8192']]
    pa = (max(sweepA) - min(sweepA)) <= 0.02
    g1 = recs['r64_F8192'] - recs['r64_F4096']
    g2 = recs['r64_F16384'] - recs['r64_F8192']
    pb = g1 >= 0.015 and g2 >= 0.015
    pc = recs['r64_F16384'] >= 0.77
    out = {'ce': res, 'stake': round(stake, 4), 'lin5': round(r5, 4),
           'recoveries': recs, 'sweepA_spread': round(max(sweepA) - min(sweepA), 4),
           'F_doubling_gains': [round(g1, 4), round(g2, 4)],
           'pred_a_isotropy': bool(pa), 'pred_b_capacity_limited': bool(pb),
           'pred_c_F16384_77': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin5 {r5:.3f} | {recs}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
