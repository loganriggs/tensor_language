# deep_tail_rank_b: RANK-64-TRUNCATED RIDGE TAIL (S1535 pool: the hybrid's wide
# ridge cost 190-380 Mbit — disqualifying. Truncate the fitted ridge to rank 64 via
# SVD: price becomes 256-core (14 Mbit) + 64x((L+1)D + D) floats (~15-25 Mbit) —
# ~35 Mbit total, 3x cheaper than units-1024. Same capture/fit machinery; the only
# new arm is the truncated-tail hybrid.
#
# Registered predictions:
#   pred_a the rank-64 tail retains >= .90x the full-ridge hybrid's recovery at
#          every target in this part.
#   pred_b price <= 45 Mbit/module (computed in-script).
#   pred_c the truncated hybrid beats units-1024 recovery at >= half the targets
#          (at ~60% of units-1024's price).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_tail_rank_b_results.json'
NFIT = 480; NEV = 960
H = m.transformer.h
TARGETS = [9, 14, 15]
STAND = {'L': None, 'tensor': None}
CAP = {'on': False, 'store': None}
NAMES = [f'm{L}' for L in range(18)] + [f'a{L}' for L in TARGETS] \
    + [f'z{L}' for L in TARGETS]


def cap_hook_for(name):
    def hook(mod, args, output):
        if CAP['on']:
            CAP['store'][name].append(output.detach().float().cpu())
        return None
    return hook


def cap_pre_hook_for(name):
    def hook(mod, args):
        if CAP['on']:
            CAP['store'][name].append(args[0].detach().float().cpu())
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

    hooks = [H[L].mlp.register_forward_hook(cap_hook_for(f'm{L}')) for L in range(18)]
    hooks += [H[L].attn.c_proj.register_forward_hook(cap_hook_for(f'a{L}'))
              for L in TARGETS]
    hooks += [H[L].mlp.register_forward_pre_hook(cap_pre_hook_for(f'z{L}'))
              for L in TARGETS]
    shooks = [H[L].mlp.register_forward_hook(stand_hook_for(L)) for L in TARGETS]
    FT = capture(FITR)
    print("capture done", flush=True)

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    K = 256
    FITS = {}
    for L in TARGETS:
        blk = H[L]
        Zf = FT[f'z{L}'].reshape(-1, D)
        CH = 16384
        HDL = blk.mlp.Left.weight.shape[0]
        a1 = torch.zeros(HDL, device=DEV); a2 = torch.zeros(HDL, device=DEV); n0 = 0
        for i in range(0, Zf.shape[0], CH):
            zz = Zf[i:i + CH].to(DEV)
            h = blk.mlp.Left(zz).float() * blk.mlp.Right(zz).float()
            a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
        mu = a1 / n0
        hs = (a2 / n0 - mu * mu).clamp_min(0).sqrt()
        score = hs * blk.mlp.Down.weight.float().norm(dim=0)
        topu = score.argsort(descending=True)[:K]
        Wl = blk.mlp.Left.weight.float()[topu]
        Wr = blk.mlp.Right.weight.float()[topu]
        Wd = blk.mlp.Down.weight.float()[:, topu]
        bb_ = blk.mlp.Down_bias.detach().float()

        def core_out(z_cpu):
            outs = []
            for i in range(0, z_cpu.shape[0], CH):
                zz = z_cpu[i:i + CH].to(DEV)
                h = (zz @ Wl.T).float() * (zz @ Wr.T).float()
                outs.append((h @ Wd.T + bb_).cpu())
            return torch.cat(outs)

        Yf = FT[f'm{L}'].reshape(-1, D)
        CORE = core_out(Zf)
        RES = Yf - CORE
        blocks = [f'a{L}'] + [f'm{j}' for j in range(L)]
        XA = torch.cat([FT[b] for b in blocks], -1).reshape(-1, (L + 1) * D)
        WA, xmA, ymA = ridge_chunked(XA, RES)
        Uw, Sw, Vw = torch.svd_lowrank(WA, q=80, niter=4)
        WA64 = (Uw[:, :64] * Sw[:64]) @ Vw[:, :64].T
        FITS[L] = (Wl, Wr, Wd, bb_, WA, xmA, ymA, Yf.mean(0).to(DEV), blocks, WA64)
        del XA, CORE, RES
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
                Wl, Wr, Wd, bb_, WA, xmA, ymA, gmean, blocks, WA64 = FITS[L]
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                else:
                    ze = E[f'z{L}'].reshape(-1, D).to(DEV)
                    h = (ze @ Wl.T).float() * (ze @ Wr.T).float()
                    core = h @ Wd.T + bb_
                    if mode == 'core':
                        st = core.view(B, T, D)
                    else:
                        XeA = torch.cat([E[b] for b in blocks],
                                        -1).reshape(-1, len(blocks) * D).to(DEV)
                        rid = ymA + (XeA - xmA) @ WA
                        if mode == 'hybrid64':
                            rid64 = ymA + (XeA - xmA) @ WA64
                            st = (core + rid64).view(B, T, D)
                        else:
                            st = (core + rid).view(B, T, D)
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
    HYB_REF = {4: .7585, 5: .6935, 6: .6576, 7: .5734, 8: .5549, 9: .5641,
               10: .4945, 11: .5305, 12: .5085, 13: .4971, 14: .4119, 15: .5239}
    U1024_REF = {4: .7032, 5: .6577, 6: .7098, 7: .6119, 8: .6446, 9: .6474,
                 10: .5939, 11: .6072, 12: .5818, 13: .5661, 14: .417, 15: .5147}
    recs = {}
    for L in TARGETS:
        cm = ce_run(L, 'mean')
        ch = ce_run(L, 'hybrid64')
        stake = cm - clean
        recs[L] = (cm - ch) / max(stake, 1e-6)
        price = (3 * 256 * D + 64 * ((L + 1) * D + D)) * 16 / 1e6
        res[f'mlp{L}'] = {'rec_hybrid64': round(recs[L], 4),
                          'hyb_full_ref': HYB_REF[L], 'units1024_ref': U1024_REF[L],
                          'price_mbit': round(price, 1)}
        print(f"mlp{L}: {res[f'mlp{L}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks + shooks:
        hk.remove()

    pa = all(recs[L] >= 0.90 * HYB_REF[L] for L in TARGETS)
    pb = all(res[f'mlp{L}']['price_mbit'] <= 45 for L in TARGETS)
    pc = sum(1 for L in TARGETS if recs[L] > U1024_REF[L]) >= len(TARGETS) / 2
    out = {'res': res,
           'pred_a_retains_90': bool(pa), 'pred_b_price_45': bool(pb),
           'pred_c_beats_units_half': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
