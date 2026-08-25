# mlp4_from_inputs: STAND-INS AS FUNCTIONS OF UPSTREAM COMPONENT OUTPUTS (§1422-23:
# mlp4 reads {mlp0, mlp2, mlp3}, ignores lexicon/topic/attention). Fit on skip=80 rows,
# EVAL HELD OUT on skip=7000 (benchmark rule 1). Stand-ins for mlp4's output, everything
# else live: (a) lin3 = ridge from concat [mlp0out, mlp2out, mlp3out] (3456 -> D);
# (b) lin0 = ridge from mlp0out alone; (c) class64 = K=64 k-means over mlp0out, table
# of mean mlp4-out per class ("64 class codes"). Ridge lambda = .01 x mean diag.
#
# Registered predictions:
#   pred_a lin3 recovers >= .50 of mlp4's stake held-out.
#   pred_b lin0 recovery >= .60 of lin3's (mlp0 dominance carries to construction).
#   pred_c class64 recovers >= .25 (a 64-entry code table is a real partial description).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_from_inputs_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm2', 'm3', 'm4')}
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
             for L, n in ((0, 'm0'), (2, 'm2'), (3, 'm3'), (4, 'm4'))]
    FT = capture(FITR)
    print("fit capture done", flush=True)

    Xf = torch.cat([FT['m0'], FT['m2'], FT['m3']], -1).reshape(-1, 3 * D)
    X0f = FT['m0'].reshape(-1, D)
    Yf = FT['m4'].reshape(-1, D)

    def ridge(X, Y):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm = Xg.mean(0); ym = Yg.mean(0)
        Xc = Xg - xm; Yc = Yg - ym
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV), Xc.T @ Yc)
        return W, xm, ym

    W3, xm3, ym3 = ridge(Xf, Yf)
    W0, xm0, ym0 = ridge(X0f, Yf)
    print("ridges fit", flush=True)

    g = torch.Generator().manual_seed(11)
    Xk = X0f.to(DEV)
    cent = Xk[torch.randperm(Xk.shape[0], generator=g)[:K]].clone()
    for it in range(25):
        lab = torch.cdist(Xk, cent).argmin(1)
        for k in range(K):
            sel = lab == k
            if int(sel.sum()) > 0:
                cent[k] = Xk[sel].mean(0)
    ctab = torch.zeros(K, D, device=DEV); ccnt = torch.zeros(K, device=DEV)
    ctab.index_add_(0, lab, Yf.to(DEV)); ccnt.index_add_(0, lab, torch.ones_like(lab, dtype=torch.float))
    gmean = Yf.mean(0).to(DEV)
    ctab = torch.where(ccnt.unsqueeze(1) > 0, ctab / ccnt.clamp_min(1).unsqueeze(1),
                       gmean.unsqueeze(0))
    print("class table fit; sizes:", torch.bincount(lab, minlength=K).min().item(),
          "min /", torch.bincount(lab, minlength=K).max().item(), "max", flush=True)

    mh = H[4].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm2', 'm3', 'm4')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'lin3':
                    Xe = torch.cat([E['m0'], E['m2'], E['m3']], -1).reshape(-1, 3 * D).to(DEV)
                    st = (ym3 + (Xe - xm3) @ W3).view(B, T, D)
                elif mode == 'lin0':
                    Xe = E['m0'].reshape(-1, D).to(DEV)
                    st = (ym0 + (Xe - xm0) @ W0).view(B, T, D)
                else:  # class64
                    Xe = E['m0'].reshape(-1, D).to(DEV)
                    labe = torch.cdist(Xe, cent).argmin(1)
                    st = ctab[labe].view(B, T, D)
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
    for mode in (None, 'mean', 'lin3', 'lin0', 'class64'):
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    stake = res['mean'] - res['None']
    rec = lambda a: (res['mean'] - res[a]) / max(stake, 1e-6)
    r3, r0, rc = rec('lin3'), rec('lin0'), rec('class64')
    pa = r3 >= 0.50
    pb = r0 >= 0.60 * max(r3, 1e-4)
    pc = rc >= 0.25
    out = {'ce': res, 'stake': round(stake, 4),
           'recovery': {'lin3': round(r3, 4), 'lin0': round(r0, 4),
                        'class64': round(rc, 4)},
           'pred_a_lin3_50': bool(pa), 'pred_b_lin0_carries': bool(pb),
           'pred_c_class64_25': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | lin3 {r3:.3f} lin0 {r0:.3f} class64 {rc:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
