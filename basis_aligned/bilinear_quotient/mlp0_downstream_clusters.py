# mlp0_downstream_clusters: CLUSTERS DEFINED BY DOWNSTREAM DISCRIMINATION (user
# directive 2026-08-25: don't cluster mlp0's outputs by their own geometry — cluster by
# how DOWNSTREAM computation discriminates them, from the weights; low K = minimal
# description, K=2 would be maximally interpretable). Prior art: mlp0 effective rank 24
# (S1384); activation-defined k-means (mlp0_clusters, module-ladder era): K=16 .37 /
# K=64 .56 / K=256 .65 recovery, digits in 2 clusters at K=16.
# Method: token t -> mlp0 table vector v_t -> DISCRIMINATION EMBEDDING = concat of
# block-1 weight images [c_q1 v; c_k1 v; c_v1 v; Left1 v; Right1 v], each randomly
# projected to 128 dims (fixed seed) -> k-means at K in {2, 16, 64, 256} -> cluster
# table = mean mlp0 output per cluster (fit rows).控 Control: same K, k-means on raw
# v_t (activation-defined). Extra arm: SVD rank-24 of the table (tests the remembered
# rank). All fid_opt on FROZEN sweep anchors. Bits: K x D x 16 + V x log2(K).
#
# Registered predictions:
#   pred_a downstream-defined beats activation-defined at BOTH K=16 and K=64.
#   pred_b K=64 downstream >= .60 fid_opt.
#   pred_c SVD rank-24 of the table >= .80 fid_opt (the S1384 effective rank is a
#          usable compression rank).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_downstream_clusters_results.json'
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
    CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0',)}
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
    FT = capture(FITR)
    print("fit capture done", flush=True)

    Yf = FT['m0'].reshape(-1, D)
    toksF = FITR[:, :-1].reshape(-1)
    tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
    tsum.index_add_(0, toksF, Yf); tcnt.index_add_(0, toksF, torch.ones(toksF.shape[0]))
    gmean_t = Yf.mean(0)
    TOKTAB = torch.where(tcnt.unsqueeze(1) > 0, tsum / tcnt.clamp_min(1).unsqueeze(1),
                         gmean_t.unsqueeze(0)).to(DEV)
    # discrimination embedding from downstream WEIGHTS (block 1 readers of the stream)
    at1 = H[1].attn
    maps = [at1.c_q.weight, at1.c_k.weight, at1.c_v.weight,
            H[1].mlp.Left.weight, H[1].mlp.Right.weight]
    g = torch.Generator().manual_seed(13)
    Tc = (TOKTAB - gmean_t.to(DEV))
    parts = []
    for Wm in maps:
        img = Tc @ Wm.float().to(DEV).T          # [V, out]
        P = torch.randn(img.shape[1], 128, generator=g).to(DEV) / (img.shape[1] ** 0.5)
        parts.append(img @ P)
    EMB_DOWN = torch.nn.functional.normalize(torch.cat(parts, 1), dim=1)   # [V, 640]
    EMB_ACT = torch.nn.functional.normalize(Tc, dim=1)

    def kmeans_tab(EMB, K, seed):
        gk = torch.Generator().manual_seed(seed)
        # weight k-means by token frequency (tcnt) so clusters reflect usage
        w = tcnt.to(DEV).clamp_min(0.0)
        cent = EMB[torch.randperm(EMB.shape[0], generator=gk)[:K]].clone()
        for it in range(20):
            lab = torch.cdist(EMB, cent).argmin(1)
            for k2 in range(K):
                sel = lab == k2
                if float(w[sel].sum()) > 0:
                    cent[k2] = (EMB[sel] * w[sel].unsqueeze(1)).sum(0) / w[sel].sum()
        lab = torch.cdist(EMB, cent).argmin(1)
        tabsum = torch.zeros(K, D, device=DEV); tabw = torch.zeros(K, device=DEV)
        tabsum.index_add_(0, lab, TOKTAB * w.unsqueeze(1)); tabw.index_add_(0, lab, w)
        ctab = torch.where(tabw.unsqueeze(1) > 0, tabsum / tabw.clamp_min(1e-6).unsqueeze(1),
                           gmean_t.to(DEV).unsqueeze(0))
        return ctab[lab]                          # expanded to [V, D]

    VAR = {'full': TOKTAB}
    for K in (2, 16, 64, 256):
        VAR[f'down{K}'] = kmeans_tab(EMB_DOWN, K, 100 + K)
        VAR[f'act{K}'] = kmeans_tab(EMB_ACT, K, 200 + K)
        print(f"K={K} built", flush=True)
    U_, S_, Vt_ = torch.linalg.svd(Tc, full_matrices=False)
    VAR['svd24'] = gmean_t.to(DEV) + (U_[:, :24] * S_[:24]) @ Vt_[:24]
    import math
    MBITS = {'full': 50257 * D * 16 / 1e6, 'svd24': (50257 * 24 + 24 * D) * 16 / 1e6}
    for K in (2, 16, 64, 256):
        b = (K * D * 16 + 50257 * math.log2(K)) / 1e6
        MBITS[f'down{K}'] = b; MBITS[f'act{K}'] = b

    gmean = gmean_t.to(DEV)
    print("variants built", flush=True)


    mh = H[0].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                CAP['on'] = True; CAP['store'] = {n: [] for n in ('m0',)}
                STAND['mode'] = None
                fwd(idx)
                CAP['on'] = False
                E = {n: torch.cat(v) for n, v in CAP['store'].items()}
                B = idx.shape[0]
                tokse = idx.reshape(-1).cpu()
                if mode == 'mean':
                    st = gmean.expand(B, T, D)
                else:
                    st = VAR[mode][tokse].view(B, T, D)
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
    ALLM = [None, 'mean', 'full', 'svd24'] + [f'{p}{K}' for K in (2, 16, 64, 256)
                                              for p in ('down', 'act')]
    for mode in ALLM:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp0']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    fids = {k: round(fid(res[k]), 4) for k in res if k not in ('None', 'mean')}
    pa = fids['down16'] > fids['act16'] and fids['down64'] > fids['act64']
    pb = fids['down64'] >= 0.60
    pc = fids['svd24'] >= 0.80
    out = {'ce': res, 'fid_opt': fids, 'mbits': {k: round(v, 3) for k, v in MBITS.items()},
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_down_beats_act': bool(pa), 'pred_b_down64_60': bool(pb),
           'pred_c_rank24_80': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    _j.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
