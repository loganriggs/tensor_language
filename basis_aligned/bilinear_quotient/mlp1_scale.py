# mlp1_scale: SCALE THE BOARD-#1 PLANK (mlp1 = .181 unexplained at .975; the
# verified plank is tier2000 + rank-128 ridge = .9507. Mechanical scaling arms:
# tier8000 exact rows + rank-64 tail; full residual ridge; rank-256 ridge.
# Fit skip=80, EVAL skip=7000, frozen anchors.
#
# Registered predictions:
#   pred_a tier8000 + full ridge >= .965 fid_opt.
#   pred_b tier8000 + rank-256 ridge >= .960.
#   pred_c rank-256 beats the verified .9507 by >= .008.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_scale_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'a1')}
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
             for L, n in ((0, 'm0'), (1, 'm1'))]
    hooks.append(H[1].attn.c_proj.register_forward_hook(cap_hook_for('a1')))
    FT = capture(FITR)
    print("fit capture done", flush=True)

    Yf = FT['m1'].reshape(-1, D)
    toksF = FITR[:, :-1].reshape(-1)
    tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
    tsum.index_add_(0, toksF, Yf); tcnt.index_add_(0, toksF, torch.ones(toksF.shape[0]))
    gmean_t = Yf.mean(0)
    TOKTAB = torch.where(tcnt.unsqueeze(1) > 0, tsum / tcnt.clamp_min(1).unsqueeze(1),
                         gmean_t.unsqueeze(0)).to(DEV)
    Tc = TOKTAB - gmean_t.to(DEV)
    U_, S_, Vt_ = torch.linalg.svd(Tc, full_matrices=False)
    base64 = gmean_t.to(DEV) + (U_[:, :64] * S_[:64]) @ Vt_[:64]
    freq_order = tcnt.argsort(descending=True)
    TIER = base64.clone(); keep = freq_order[:8000]; TIER[keep] = TOKTAB[keep]
    toksF2 = FITR[:, :-1].reshape(-1)
    RES = Yf - TIER[toksF2].cpu()          # residual of the TIERED table
    X2f = torch.cat([FT['a1'], FT['m0']], -1).reshape(-1, 2 * D)
    Xg = X2f.to(DEV); xm2 = Xg.mean(0)
    Yg = RES.to(DEV); ym2 = Yg.mean(0)
    Xc = Xg - xm2; Yc = Yg - ym2
    XtX = Xc.T @ Xc
    lam = 0.01 * float(torch.diagonal(XtX).mean())
    W2 = torch.linalg.solve(XtX + lam * torch.eye(2 * D, device=DEV), Xc.T @ Yc)
    Ur, Sr, Vtr = torch.linalg.svd(W2, full_matrices=False)
    W128 = (Ur[:, :256] * Sr[:256]) @ Vtr[:256]
    MBITS = {'tier': (8000 * D + 50257 * 64 + 64 * D) * 16 / 1e6,
             'tier_ridge': (8000 * D + 50257 * 64 + 64 * D + 2 * D * D) * 16 / 1e6,
             'tier_ridge128': (8000 * D + 50257 * 64 + 64 * D + 256 * 3 * D) * 16 / 1e6}

    gmean = gmean_t.to(DEV)
    print("variants built", flush=True)


    mh = H[1].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0', 'm1', 'a1')}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                tokse = idx.reshape(-1).cpu()
                base_t = TIER[tokse].view(B, T, D)
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                elif mode == 'tier':
                    st = base_t
                else:
                    Xe = torch.cat([E['a1'], E['m0']], -1).reshape(-1, 2 * D).to(DEV)
                    Wm = W2 if mode == 'tier_ridge' else W128
                    st = base_t + (ym2 + (Xe - xm2) @ Wm).view(B, T, D)
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
    for mode in [None, 'mean', 'tier', 'tier_ridge', 'tier_ridge128']:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp1']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    fids = {k: round(fid(res[k]), 4) for k in ('tier', 'tier_ridge', 'tier_ridge128')}
    pa = fids['tier_ridge'] >= 0.965
    pb = fids['tier_ridge128'] >= 0.960
    pc = (fids['tier_ridge128'] - 0.9507) >= 0.008
    out = {'ce': res, 'fid_opt': fids, 'mbits': {k: round(v, 1) for k, v in MBITS.items()},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_t8k_ridge_965': bool(pa), 'pred_b_t8k_r256_960': bool(pb),
           'pred_c_beats_9507_by_008': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    _j.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
