# mlp2_rank_curve: THE FIRST BITS-VS-FIDELITY FRONTIER (user question: how do we
# measure simplicity? Answer: description length at policy B=16 bits/param — and
# mlp2's lin2 map at 42.5 Mbit vs the module's own 255 Mbit is our first true glass
# plank). SVD-truncate the lin2 ridge W (2304 x 1152) at ranks {8, 16, 32, 64, 128,
# 256, 512, full}; fidelity per rank on FROZEN sweep anchors; bits = r x (2304+1152)
# x 16 + biases. Fit skip=80, EVAL skip=7000.
#
# Registered predictions:
#   pred_a rank-64 (3.5 Mbit) holds >= .80 fid_opt.
#   pred_b rank-256 (14 Mbit) >= .90 (within .02 of full's .920).
#   pred_c the fidelity-per-bit knee is at rank <= 64 (marginal fid per doubling
#          drops below half its previous value beyond 64).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp2_rank_curve_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'a2')}
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
             for L, n in ((0, 'm0'), (1, 'm1'), (2, 'm2'))]
    hooks.append(H[2].attn.c_proj.register_forward_hook(cap_hook_for('a2')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    X2f = torch.cat([FT['a2'], FT['m1']], -1).reshape(-1, 2 * D)
    X5f = torch.cat([FT['a2'], FT['m0'], FT['m1']], -1).reshape(-1, 3 * D)
    Yf = FT['m2'].reshape(-1, D)

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

    ARMS = []
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
        NPOS = FT['m2'].shape[0] * T
        CH = 16384
        A4 = FT['a2'].reshape(-1, D); M3 = FT['m1'].reshape(-1, D)
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
    RANKS = [8, 16, 32, 64, 128, 256, 512, 1152]
    U_, S_, Vt_ = torch.linalg.svd(W2, full_matrices=False)
    WR = {r: (U_[:, :r] * S_[:r]) @ Vt_[:r] for r in RANKS}
    print("rank family built", flush=True)

    mh = H[2].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'm2', 'a2')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                Xe5 = torch.cat([E['a2'], E['m0'], E['m1']],
                                -1).reshape(-1, 3 * D).to(DEV)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                else:
                    rk = int(mode.split('rank')[1])
                    Xe2 = torch.cat([E['a2'], E['m1']], -1).reshape(-1, 2 * D).to(DEV)
                    st = (ym2 + (Xe2 - xm2) @ WR[rk]).view(B, T, D)
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
    for mode in [None, 'mean'] + [f'rank{r}' for r in RANKS]:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp2']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    fids = {r: round(fid(res[f'rank{r}']), 4) for r in RANKS}
    bits = {r: r * (2 * D + D) * 16 for r in RANKS}
    pa = fids[64] >= 0.80
    pb = fids[256] >= 0.90
    gains = {r: fids[r] - fids[rp] for r, rp in zip(RANKS[1:], RANKS[:-1])}
    knee_ok = all(gains.get(r2_, 1) < 0.5 * max(gains.get(r1_, 1e-6), 1e-6)
                  for r1_, r2_ in [(64, 128)])
    pc = knee_ok
    out = {'ce': res, 'fid_opt_by_rank': fids,
           'mbits_by_rank': {r: round(b / 1e6, 2) for r, b in bits.items()},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_r64_80': bool(pa), 'pred_b_r256_90': bool(pb),
           'pred_c_knee_le_64': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
