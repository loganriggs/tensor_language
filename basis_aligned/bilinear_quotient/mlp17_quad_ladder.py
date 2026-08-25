# mlp17_quad_ladder: LADDER FOR THE TOP READOUT MODULE (board: mlp17 .053 unexplained;
# benchmark: fitted linear read .84). Arms for mlp17's output (held-out, FROZEN
# anchors): linresid = ridge from mlp17's own normed INPUT (reproduces the linear
# read); +quad r=256/F=8192 sampled pairs over the same input. Input captured via
# forward pre-hook. Fit skip=80, EVAL skip=7000.
#
# Registered predictions:
#   pred_a linresid >= .80 fid_opt (reproduces the S1131 read under this harness).
#   pred_b +quad adds >= .04 (top modules are categorical/bilinear — quadratic terms
#          should matter more here than in the mids).
#   pred_c combined >= .88.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_quad_ladder_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m17', 'xin')}
    for i in range(0, rows.shape[0], 8):
        fwd(rows[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    return {n: torch.cat(v) for n, v in CAP['store'].items()}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[17].mlp.register_forward_hook(cap_hook_for('m17')),
             H[17].mlp.register_forward_pre_hook(
                 lambda mod, args: CAP['store']['xin'].append(
                     args[0].detach().float().cpu()) if CAP['on'] else None)]
    FT = capture(FITR)
    print("fit capture done", flush=True)

    X2f = FT['xin'].reshape(-1, D)
    X5f = X2f
    Yf = FT['m17'].reshape(-1, D)

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
        Bj = torch.linalg.qr(torch.randn(D, r, generator=gq))[0].to(DEV)
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
        Xj = Ea4.to(DEV) - xm2
        z = Xj @ Bj
        return z[:, pi] * z[:, pj]

    def fit_quad(name):
        Fn = QSPEC[name][1].shape[0]
        XtX = torch.zeros(Fn, Fn, device=DEV)
        XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        NPOS = FT['m17'].shape[0] * T
        CH = 16384
        A4 = FT['xin'].reshape(-1, D); M3 = FT['xin'].reshape(-1, D) * 0
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

    mh = H[17].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m17', 'xin')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                Xe5 = E['xin'].reshape(-1, D).to(DEV)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin2':
                    st = lin5p.view(B, T, D)
                elif mode == 'linall':
                    st = lin5p.view(B, T, D)
                else:
                    A4e = E['xin'].reshape(-1, D); M3e = A4e * 0
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

    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp17']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    r2, rall, rq = fid(res['lin2']), fid(res['linall']), fid(res['quad_r256_F8192'])
    stake = res['mean'] - res['None']
    pa = r2 >= 0.80
    pb = (rq - r2) >= 0.04
    pc = rq >= 0.88
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin2': round(r2, 4), 'linall': round(rall, 4),
                        'quad_r256': round(rq, 4)},
           'pred_a_linread_80': bool(pa), 'pred_b_quad_04': bool(pb),
           'pred_c_88': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin2 {r2:.3f} linall {rall:.3f} quad {rq:.3f} (stake {stake:.4f})")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
