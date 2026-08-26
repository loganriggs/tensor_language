# deep_hybrid_b: UNIT CORE + STREAM RIDGE (S1534: deep MLPs = concentrated
# ~64-256-unit core + diffuse tail; the tail should be the linear-in-stream part).
# Stand-in per target: EXACT top-256-unit sub-MLP (ranked std(h) x ||Down col||)
# PLUS a linall ridge [aL, m0..m{L-1}] fit on the SUB-MLP'S RESIDUAL (sequential
# refit). One capture pass; frozen anchors; NEV=960. Price ~ 14 Mbit core + ridge.
#
# Registered predictions:
#   pred_a hybrid recovery >= .65 at a majority of this part's targets.
#   pred_b hybrid beats BOTH parents (K=256 core alone and linall alone) by >= .08
#          at a majority.
#   pred_c the weakest target (mlp14 in part a / mlp15 in part b) >= .45.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_hybrid_b_results.json'
NFIT = 480; NEV = 960
H = m.transformer.h
TARGETS = [4, 6, 8, 10, 11, 13, 15]
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
        FITS[L] = (Wl, Wr, Wd, bb_, WA, xmA, ymA, Yf.mean(0).to(DEV), blocks)
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
                Wl, Wr, Wd, bb_, WA, xmA, ymA, gmean, blocks = FITS[L]
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
                        if mode == 'ridge_only':
                            st = (gmean + rid - rid.mean(0)).view(B, T, D)
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
    CORE_REF = {4: .4297, 5: .3394, 6: .525, 7: .4058, 8: .4793, 9: .4892,
                10: .4218, 11: .4496, 12: .4275, 13: .4092, 14: .1974, 15: .321,
                16: .78, 17: .7943}
    LINALL_REF = {4: .74, 5: .65, 6: .62, 7: .4595, 8: .4568, 9: .4819, 10: .4009,
                  11: .4505, 12: .4338, 13: .4217, 14: .3401, 15: .4383, 16: .81,
                  17: .878}
    hyb = {}
    for L in TARGETS:
        cm = ce_run(L, 'mean')
        chy = ce_run(L, 'hybrid')
        stake = cm - clean
        hyb[L] = (cm - chy) / max(stake, 1e-6)
        res[f'mlp{L}'] = {'mean': round(cm, 4), 'hybrid': round(chy, 4),
                          'rec_hybrid': round(hyb[L], 4),
                          'core256_ref': CORE_REF[L], 'linall_ref': LINALL_REF[L]}
        print(f"mlp{L}: {res[f'mlp{L}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks + shooks:
        hk.remove()

    n_65 = sum(1 for L in TARGETS if hyb[L] >= 0.65)
    n_beat = sum(1 for L in TARGETS
                 if hyb[L] >= max(CORE_REF[L], LINALL_REF[L]) + 0.08)
    weakest = min(TARGETS, key=lambda L: CORE_REF[L])
    pa = n_65 > len(TARGETS) // 2
    pb = n_beat > len(TARGETS) // 2
    pc = hyb[weakest] >= 0.45
    out = {'res': res, 'n_ge_65': n_65, 'n_beats_parents': n_beat,
           'pred_a_majority_65': bool(pa), 'pred_b_beats_parents': bool(pb),
           'pred_c_weakest_45': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
