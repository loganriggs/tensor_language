# mlp17_channel: BOARD #3 ATTACKED THROUGH ITS OWN CHANNEL (S1476: mlp16->mlp17 is
# the one deep edge with real signal — .1075 CE, beyond-rank-32 only .0044). Build
# mlp17's quadratic features FROM THE CHANNEL'S OWN 32 DIRECTIONS instead of random
# projections: s = ((h16 - mu)/rms) @ V32 (V32 = top whitened right-vectors of
# [Left17; Right17] @ Down16), features = 32 squares + 496 crosses = 528, fit on the
# residual of a linall ridge (sequential refit). Reference arm: the ladder's random
# r256/F8192 quadratic over [attn17, mlp16]. Everything held-out, frozen anchors.
# Current mlp17 best: .856 (linread+quad, S1443).
#
# Registered predictions:
#   pred_a linall + channel-quad >= .90 fid_opt.
#   pred_b channel-quad gain >= random-quad gain with 15x fewer features.
#   pred_c channel-quad adds >= .04 over linall.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_channel_results.json'
NFIT = 480; NEV = 960
H = m.transformer.h
STAND = {'mode': None, 'tensor': None}
CAP = {'on': False, 'store': None}
MNAMES = [f'm{L}' for L in range(17)]
NAMES = MNAMES + ['m17', 'a17', 'z16']


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def cap_pre_hook(name):
    def hook(mod, args):
        if CAP['on']:
            CAP['store'][name].append(args[0].detach().float().cpu())
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
def h16_of(z16):
    """z16 [N, D] cpu -> h16 [N, HD] on DEV in chunks handled by caller."""
    blk = H[16]
    zg = z16.to(DEV)
    return blk.mlp.Left(zg).float() * blk.mlp.Right(zg).float()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(f'm{L}')) for L in range(17)]
    hooks.append(H[17].mlp.register_forward_hook(cap_hook_for('m17')))
    hooks.append(H[17].attn.c_proj.register_forward_hook(cap_hook_for('a17')))
    hooks.append(H[16].mlp.register_forward_pre_hook(cap_pre_hook('z16')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    # channel stats + directions
    Z16f = FT['z16'].reshape(-1, D)
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, Z16f.shape[0], 16384):
        h = h16_of(Z16f[i:i + 16384])
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    mu16 = a1 / n0
    rms16 = (a2 / n0).clamp_min(1e-12).sqrt()
    Wd16 = H[16].mlp.Down.weight.float().to(DEV)
    C = torch.cat([H[17].mlp.Left.weight.float().to(DEV) @ Wd16,
                   H[17].mlp.Right.weight.float().to(DEV) @ Wd16], 0)
    Cw = C * rms16.unsqueeze(0)
    U_, S_, V_ = torch.svd_lowrank(Cw, q=64, niter=4)
    V32 = V_[:, :32]                                    # [HD, 32]
    print("channel dirs built", flush=True)

    iu = torch.triu_indices(32, 32)

    def chan_feats(z16_cpu):
        outs = []
        for i in range(0, z16_cpu.shape[0], 16384):
            h = h16_of(z16_cpu[i:i + 16384])
            s = ((h - mu16) / rms16) @ V32              # [n, 32]
            q = s[:, iu[0]] * s[:, iu[1]]               # 528 features
            outs.append(q)
        return torch.cat(outs)

    Yf = FT['m17'].reshape(-1, D)
    XAf = torch.cat([FT['a17']] + [FT[n] for n in MNAMES], -1).reshape(-1, 18 * D)

    def ridge_chunked(X, Y):
        n, d = X.shape
        xm = X.mean(0).to(DEV); ym = Y.mean(0).to(DEV)
        XtX = torch.zeros(d, d, device=DEV)
        XtY = torch.zeros(d, Y.shape[1], device=DEV)
        CH = 8192
        for i in range(0, n, CH):
            Xc = X[i:i + CH].to(DEV) - xm
            Yc = Y[i:i + CH].to(DEV) - ym
            XtX += Xc.T @ Xc; XtY += Xc.T @ Yc
            del Xc, Yc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(d, device=DEV), XtY)
        return W, xm, ym

    WA, xmA, ymA = ridge_chunked(XAf, Yf)
    print("linall ridge fit", flush=True)

    def fit_on_resid(feats_fn, Fn):
        XtX = torch.zeros(Fn, Fn, device=DEV)
        XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        CH = 16384
        NPOS = Yf.shape[0]
        for i in range(0, NPOS, CH):
            q = feats_fn(i, i + CH)
            resid = (Yf[i:i + CH].to(DEV)
                     - (ymA + (XAf[i:i + CH].to(DEV) - xmA) @ WA))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); n += q.shape[0]
        XtX -= torch.outer(xs, xs) / n; XtY -= torch.outer(xs, ys) / n
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(Fn, device=DEV), XtY)
        return Wq, xs / n, ys / n

    QCH = fit_on_resid(lambda a, b: chan_feats(Z16f[a:b]), 528)
    print("channel quad fit", flush=True)

    gq = torch.Generator().manual_seed(31 + 256)
    Bj = torch.linalg.qr(torch.randn(2 * D, 256, generator=gq))[0].to(DEV)
    gp = torch.Generator().manual_seed(101 + 8192 + 256)
    pi = torch.randint(0, 256, (8192,), generator=gp).to(DEV)
    pj = torch.randint(0, 256, (8192,), generator=gp).to(DEV)
    X2f = torch.cat([FT['a17'], FT['m16']], -1).reshape(-1, 2 * D)
    xm2 = X2f.mean(0).to(DEV)

    def rand_feats(a, b):
        Xj = X2f[a:b].to(DEV) - xm2
        z = Xj @ Bj
        return z[:, pi] * z[:, pj]

    QRD = fit_on_resid(rand_feats, 8192)
    print("random quad fit", flush=True)
    gmean = Yf.mean(0).to(DEV)

    mh = H[17].mlp.register_forward_hook(stand_hook)

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
                XeA = torch.cat([E['a17']] + [E[n] for n in MNAMES],
                                -1).reshape(-1, 18 * D).to(DEV)
                linAp = ymA + (XeA - xmA) @ WA
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'linall':
                    st = linAp.view(B, T, D)
                elif mode == 'chanquad':
                    Wq, qm, rm = QCH
                    q = chan_feats(E['z16'].reshape(-1, D))
                    st = (linAp + rm + (q - qm) @ Wq).view(B, T, D)
                else:
                    Wq, qm, rm = QRD
                    Xe2 = torch.cat([E['a17'], E['m16']], -1).reshape(-1, 2 * D)
                    z = (Xe2.to(DEV) - xm2) @ Bj
                    q = z[:, pi] * z[:, pj]
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
    for mode in [None, 'mean', 'linall', 'chanquad', 'randquad']:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp17']
    fid = lambda a: (sw['ce_opt'] - res[a]) / max(sw['ce_opt'] - res['None'], 1e-6)
    fids = {k: round(fid(k), 4) for k in ('linall', 'chanquad', 'randquad')}
    g_ch = fids['chanquad'] - fids['linall']
    g_rd = fids['randquad'] - fids['linall']
    pa = fids['chanquad'] >= 0.90
    pb = g_ch >= g_rd
    pc = g_ch >= 0.04
    out = {'ce': res, 'fid_opt': fids,
           'gains': {'chanquad': round(g_ch, 4), 'randquad': round(g_rd, 4)},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_chan_90': bool(pa), 'pred_b_chan_ge_rand': bool(pb),
           'pred_c_chan_adds_04': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids} gains ch {g_ch:.4f} rd {g_rd:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
