# deep_mid_sweep: THE REMAINING DEEP-MID MLPS (10-15) IN ONE AMORTIZED RUN (user
# parallelization directive; mlp7/8/9 all showed the same profile: linall ~.46-.48,
# quad adds <= .05 — this fills the last six zoo-only MLP seeds and TESTS whether the
# profile is universal in 10-15). One capture pass (m0..m15, a10..a15); per target L:
# linall ridge from [aL, m0..m{L-1}] (chunked) + r256/F8192 sampled-pair quadratic
# over [aL, m{L-1}] on the linall residual. Frozen anchors. NEV=960.
#
# Registered predictions:
#   pred_a median linall recovery across mlp10-15 >= .40.
#   pred_b median quadratic ADDITION <= .03 (the deep-mid flatness generalizes).
#   pred_c median best fid_opt >= .45 (seeds land ~.5, like 7/8/9).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_mid_sweep_results.json'
NFIT = 480; NEV = 960
H = m.transformer.h
TARGETS = [10, 11, 12, 13, 14, 15]
STAND = {'L': None, 'tensor': None}
CAP = {'on': False, 'store': None}
NAMES = [f'm{L}' for L in range(16)] + [f'a{L}' for L in TARGETS]


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def stand_hook_for(L):
    def hook(mod, args, output):
        if STAND['L'] == L:
            return STAND['tensor'].to(output.dtype)
        return None
    return hook


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


def ridge_chunked(X, Y):
    n, d = X.shape
    xm = X.mean(0).to(DEV); ym = Y.mean(0).to(DEV)
    XtX = torch.zeros(d, d, device=DEV)
    XtY = torch.zeros(d, Y.shape[1], device=DEV)
    CH = 16384
    for i in range(0, n, CH):
        Xc = X[i:i + CH].to(DEV) - xm
        Yc = Y[i:i + CH].to(DEV) - ym
        XtX += Xc.T @ Xc; XtY += Xc.T @ Yc
        del Xc, Yc
    lam = 0.01 * float(torch.diagonal(XtX).mean())
    W = torch.linalg.solve(XtX + lam * torch.eye(d, device=DEV), XtY)
    return W, xm, ym


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(f'm{L}')) for L in range(16)]
    hooks += [H[L].attn.c_proj.register_forward_hook(cap_hook_for(f'a{L}'))
              for L in TARGETS]
    shooks = [H[L].mlp.register_forward_hook(stand_hook_for(L)) for L in TARGETS]
    FT = capture(FITR)
    print("capture done", flush=True)

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    gq = torch.Generator().manual_seed(31 + 256)
    Bj = torch.linalg.qr(torch.randn(2 * D, 256, generator=gq))[0].to(DEV)
    gp = torch.Generator().manual_seed(101 + 8192 + 256)
    pi = torch.randint(0, 256, (8192,), generator=gp).to(DEV)
    pj = torch.randint(0, 256, (8192,), generator=gp).to(DEV)

    FITS = {}
    for L in TARGETS:
        blocks = [f'a{L}'] + [f'm{j}' for j in range(L)]
        XA = torch.cat([FT[b] for b in blocks], -1).reshape(-1, (L + 1) * D)
        Yf = FT[f'm{L}'].reshape(-1, D)
        WA, xmA, ymA = ridge_chunked(XA, Yf)
        X2 = torch.cat([FT[f'a{L}'], FT[f'm{L - 1}']], -1).reshape(-1, 2 * D)
        xm2 = X2.mean(0).to(DEV)
        Fn = 8192
        XtX = torch.zeros(Fn, Fn, device=DEV)
        XtY = torch.zeros(Fn, D, device=DEV)
        xs = torch.zeros(Fn, device=DEV); ys = torch.zeros(D, device=DEV); n = 0
        CH = 16384
        for i in range(0, Yf.shape[0], CH):
            z = (X2[i:i + CH].to(DEV) - xm2) @ Bj
            q = z[:, pi] * z[:, pj]
            resid = (Yf[i:i + CH].to(DEV)
                     - (ymA + (XA[i:i + CH].to(DEV) - xmA) @ WA))
            XtX += q.T @ q; XtY += q.T @ resid
            xs += q.sum(0); ys += resid.sum(0); n += q.shape[0]
        XtX -= torch.outer(xs, xs) / n; XtY -= torch.outer(xs, ys) / n
        lam = 0.05 * float(torch.diagonal(XtX).mean())
        Wq = torch.linalg.solve(XtX + lam * torch.eye(Fn, device=DEV), XtY)
        FITS[L] = (WA, xmA, ymA, xm2, Wq, xs / n, ys / n,
                   Yf.mean(0).to(DEV), blocks)
        del XA, X2
        print(f"fits done L{L}", flush=True)

    def ce_run(L, mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['L'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n_2: [] for n_2 in NAMES}
                STAND['L'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n_2: torch.cat(v) for n_2, v in CAP['store'].items()}
                B = idx.shape[0]
                WA, xmA, ymA, xm2, Wq, qm, rm, gmean, blocks = FITS[L]
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                else:
                    XeA = torch.cat([E[b] for b in blocks],
                                    -1).reshape(-1, (L + 1) * D).to(DEV)
                    linp = ymA + (XeA - xmA) @ WA
                    if mode == 'linall':
                        st = linp.view(B, T, D)
                    else:
                        Xe2 = torch.cat([E[f'a{L}'], E[f'm{L - 1}']],
                                        -1).reshape(-1, 2 * D).to(DEV)
                        z = (Xe2 - xm2) @ Bj
                        q = z[:, pi] * z[:, pj]
                        st = (linp + rm + (q - qm) @ Wq).view(B, T, D)
                STAND['L'] = L
                STAND['tensor'] = st
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        STAND['L'] = None
        return s_ / max(n_, 1)

    clean = ce_run(TARGETS[0], None)
    res = {'clean': round(clean, 4)}
    recs = {}; adds = {}; fids = {}
    for L in TARGETS:
        cm = ce_run(L, 'mean'); cl_ = ce_run(L, 'linall'); cq = ce_run(L, 'quad')
        res[f'mlp{L}'] = {'mean': round(cm, 4), 'linall': round(cl_, 4),
                          'quad': round(cq, 4)}
        stake = cm - clean
        recs[L] = (cm - cl_) / max(stake, 1e-6)
        adds[L] = (cl_ - cq) / max(stake, 1e-6)
        a = sw[f'mlp{L}']
        fids[L] = (a['ce_opt'] - cq) / max(a['ce_opt'] - clean, 1e-6)
        print(f"mlp{L}: stake {stake:.4f} linall {recs[L]:.3f} "
              f"quad+ {adds[L]:.3f} fid {fids[L]:.3f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks + shooks:
        hk.remove()

    import statistics
    med_lin = statistics.median(recs.values())
    med_add = statistics.median(adds.values())
    med_fid = statistics.median(fids.values())
    pa = med_lin >= 0.40
    pb = med_add <= 0.03
    pc = med_fid >= 0.45
    out = {'ce': res,
           'linall_recovery': {L: round(v, 4) for L, v in recs.items()},
           'quad_addition': {L: round(v, 4) for L, v in adds.items()},
           'fid_opt': {L: round(v, 4) for L, v in fids.items()},
           'medians': {'linall': round(med_lin, 4), 'quad_add': round(med_add, 4),
                       'fid': round(med_fid, 4)},
           'pred_a_medlin_40': bool(pa), 'pred_b_medadd_le_03': bool(pb),
           'pred_c_medfid_45': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"medians lin {med_lin:.3f} add {med_add:.3f} fid {med_fid:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
