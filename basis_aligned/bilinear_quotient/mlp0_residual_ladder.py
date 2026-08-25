# mlp0_residual_ladder: PRIORITY TARGET #2 AFTER THE BOARD UPDATE (mlp0 residual =
# .091 nats; token map fidelity ~.90 per benchmark). Stand-ins for mlp0's output
# (held-out): (1) tok = per-token table; (2) tok+lin = table + ridge from
# [attn0, embed] on the TABLE RESIDUAL; (3) +quad (r=256/F=8192). ANCHOR FIX (S1438
# lesson): the mean arm is NOT recomputed — fidelities are scored against the FROZEN
# sweep anchors for mlp0 (in-script gmean kept only as a cross-check arm, computed
# BEFORE residualization). Fit skip=80, EVAL skip=7000.
#
# Registered predictions (scored on frozen fid_opt):
#   pred_a tok table >= .85 fid_opt (reproduces ~.90 within harness differences).
#   pred_b +ridge gains >= .02 fid_opt.
#   pred_c tok+ridge+quad >= .93 fid_opt.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_residual_ladder_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'a0')}
    for i in range(0, rows.shape[0], 8):
        fwd(rows[i:i + 8, :-1].to(DEV).contiguous())
    CAP['on'] = False
    return {n: torch.cat(v) for n, v in CAP['store'].items()}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    hooks = [H[0].mlp.register_forward_hook(cap_hook_for('m0'))]
    hooks.append(H[0].attn.c_proj.register_forward_hook(cap_hook_for('a0')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    # embed stream x0 for fit rows (computed directly, no hook)
    X0parts = []
    with torch.no_grad():
        for i in range(0, NFIT, 8):
            x0 = F.rms_norm(m.transformer.wte(FITR[i:i + 8, :-1].to(DEV)), (D,))
            X0parts.append(x0.float().cpu())
    X0f = torch.cat(X0parts)
    X2f = torch.cat([FT['a0'], X0f], -1).reshape(-1, 2 * D)
    X5f = X2f
    Yf = FT['m0'].reshape(-1, D)
    TRUE_GMEAN = Yf.mean(0).to(DEV)        # BEFORE residualization (S1438 fix)
    toksF = FITR[:, :-1].reshape(-1)
    tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
    tsum.index_add_(0, toksF, Yf); tcnt.index_add_(0, toksF, torch.ones(toksF.shape[0]))
    TOKTAB = torch.where(tcnt.unsqueeze(1) > 0, tsum / tcnt.clamp_min(1).unsqueeze(1),
                         Yf.mean(0).unsqueeze(0)).to(DEV)
    Yf = Yf - TOKTAB[toksF].cpu()          # ridge + quad fit the TABLE RESIDUAL

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
        NPOS = FT['m0'].shape[0] * T
        CH = 16384
        A4 = FT['a0'].reshape(-1, D); M3 = X0f.reshape(-1, D)
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

    mh = H[0].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'a0')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                tokse = idx.reshape(-1).cpu()
                base_t = TOKTAB[tokse].view(B, T, D)
                x0e = F.rms_norm(m.transformer.wte(idx), (D,)).float()
                Xe5 = torch.cat([E['a0'].to(DEV), x0e.reshape(-1, T, D).cpu().to(DEV)
                                 .reshape(E['a0'].shape)], -1).reshape(-1, 2 * D)
                lin5p = ym5 + (Xe5 - xm5) @ W5
                if mode == 'mean':
                    st = TRUE_GMEAN.expand(B, T, D)
                elif mode == 'lin2':          # tok table alone
                    st = base_t
                elif mode == 'linall':        # tok + residual ridge
                    st = base_t + lin5p.view(B, T, D)
                else:                         # + quad
                    A4e = E['a0'].reshape(-1, D); M3e = x0e.reshape(-1, D).cpu()
                    Wq, qm, rm = QFIT[mode]
                    q = quad_feats(A4e, M3e, mode)
                    st = base_t + (lin5p + rm + (q - qm) @ Wq).view(B, T, D)
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
    ANCH_MEAN = 3.8523; ANCH_OPT = None
    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp0']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    f2, fall, fq = fid(res['lin2']), fid(res['linall']), fid(res['quad_r256_F8192'])
    pa = f2 >= 0.85
    pb = (fall - f2) >= 0.02
    pc = fq >= 0.93
    out = {'ce': res, 'stake': round(stake, 4),
           'fid_opt': {'tok': round(f2, 4), 'tok_ridge': round(fall, 4),
                       'tok_ridge_quad': round(fq, 4)},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_tok_85': bool(pa), 'pred_b_ridge_02': bool(pb),
           'pred_c_full_93': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin2 {r2:.3f} linall {rall:.3f} quad {rq:.3f} (stake {stake:.4f})")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
