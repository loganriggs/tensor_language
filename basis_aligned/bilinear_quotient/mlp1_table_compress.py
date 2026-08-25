# mlp1_table_compress: MAKE THE TOKEN TABLE A GLASS PLANK (pricing: the full table is
# 926 Mbit vs mlp1's own 255 Mbit — it fails the plank test). Compress the table and
# put the variants on the bits-vs-fidelity frontier (fid_opt on FROZEN sweep anchors):
# arms = full table (ref .934) / SVD rank-64 / SVD rank-256 / tiered: top-2000 tokens
# exact + rank-64 tail / top-8000 exact + rank-64 tail. No residual ridge here — table
# compression isolated. Fit skip=80, EVAL skip=7000.
#
# Registered predictions:
#   pred_a SVD rank-256 (211 Mbit, passes the plank test) holds >= .90 fid_opt.
#   pred_b tiered top-2000 + rank-64 (~90 Mbit) holds >= .88.
#   pred_c the best fid-per-Mbit arm beats the full table's fid-per-Mbit by >= 5x.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_table_compress_results.json'
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
    # compressed variants (all centered on the global mean before factoring)
    Tc = TOKTAB - gmean_t.to(DEV)
    U_, S_, Vt_ = torch.linalg.svd(Tc, full_matrices=False)
    VAR = {}
    for r in (64, 256):
        VAR[f'svd{r}'] = gmean_t.to(DEV) + (U_[:, :r] * S_[:r]) @ Vt_[:r]
    freq_order = tcnt.argsort(descending=True)
    for topn in (2000, 8000):
        Tt = VAR['svd64'].clone()
        keep = freq_order[:topn]
        Tt[keep] = TOKTAB[keep]
        VAR[f'tier{topn}'] = Tt
    VAR['full'] = TOKTAB
    MBITS = {'full': 50257 * D * 16 / 1e6,
             'svd64': (50257 * 64 + 64 * D) * 16 / 1e6,
             'svd256': (50257 * 256 + 256 * D) * 16 / 1e6,
             'tier2000': (2000 * D + 50257 * 64 + 64 * D) * 16 / 1e6,
             'tier8000': (8000 * D + 50257 * 64 + 64 * D) * 16 / 1e6}

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
    for mode in [None, 'mean', 'full', 'svd64', 'svd256', 'tier2000', 'tier8000']:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    mh.remove()
    for hk in hooks:
        hk.remove()

    import json as _j
    sw = _j.load(open(PT + 'optimal_ablation_all_results.json'))['results']['mlp1']
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - res['None'], 1e-6)
    fids = {k: round(fid(res[k]), 4) for k in ('full', 'svd64', 'svd256', 'tier2000', 'tier8000')}
    fpb = {k: fids[k] / MBITS[k] for k in fids}
    pa = fids['svd256'] >= 0.90
    pb = fids['tier2000'] >= 0.88
    best = max(fpb, key=lambda k: fpb[k])
    pc = fpb[best] >= 5.0 * fpb['full']
    out = {'ce': res, 'fid_opt': fids, 'mbits': {k: round(v, 1) for k, v in MBITS.items()},
           'fid_per_mbit': {k: round(v, 5) for k, v in fpb.items()}, 'best': best,
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_svd256_90': bool(pa), 'pred_b_tier2000_88': bool(pb),
           'pred_c_5x_efficiency': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    _j.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
