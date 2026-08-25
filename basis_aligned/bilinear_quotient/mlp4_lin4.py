# mlp4_lin4: RIDGE STAND-INS WITH THE CORRECTED DIET (§1427: mlp4's marginal inputs
# are attn4 + mlp3; §1424: lin3 from front MLPs = .61 held-out). Arms: lin2 = ridge
# from [attn4out, mlp3out]; lin5 = ridge from [attn4out, mlp0out, mlp1out, mlp2out,
# mlp3out]. Fit skip=80, EVAL HELD OUT skip=7000.
#
# Registered predictions:
#   pred_a lin5 recovers >= .70 of mlp4's stake held-out.
#   pred_b lin2 (two named components) recovers >= .50.
#   pred_c lin5 beats §1424's lin3 by >= .05.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_lin4_results.json'
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
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin2':
                    Xe = torch.cat([E['a4'], E['m3']], -1).reshape(-1, 2 * D).to(DEV)
                    st = (ym2 + (Xe - xm2) @ W2).view(B, T, D)
                else:  # lin5
                    Xe = torch.cat([E['a4'], E['m0'], E['m1'], E['m2'], E['m3']],
                                   -1).reshape(-1, 5 * D).to(DEV)
                    st = (ym5 + (Xe - xm5) @ W5).view(B, T, D)
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
    for mode in (None, 'mean', 'lin2', 'lin5'):
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    r2, r5 = rec('lin2'), rec('lin5')
    LIN3 = 0.6116
    pa = r5 >= 0.70
    pb = r2 >= 0.50
    pc = (r5 - LIN3) >= 0.05
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin2': round(r2, 4), 'lin5': round(r5, 4), 'lin3_ref': LIN3},
           'pred_a_lin5_70': bool(pa), 'pred_b_lin2_50': bool(pb),
           'pred_c_beats_lin3': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | lin2 {r2:.3f} lin5 {r5:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
