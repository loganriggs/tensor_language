# mlp_ladder_depth: the MODULE LADDER as a DEPTH SWEEP — the same simplicity-curve
# instrument (mlp1_clusters / mlp0_clusters / mlp17_clusters) run on ALL 18 MLPs, so the
# per-module scoreboard becomes a curve over depth instead of three scattered points.
#
# WHAT THE THREE POINTS SO FAR SAY. Front modules are TOKEN-RESOLVED and log-linear: mlp1
# ceiling 94.4% with no elbow (§1323), mlp0 ceiling 86.3%, same shape, 8.8x smaller stake
# (§1324). The top module is the opposite on both axes: mlp17's table ceiling is only 49.7%
# and its curve ELBOWS at K=16 (k16/ceiling 0.84 — sixteen categories buy 84% of everything
# a 50k-entry token table buys, §1325). Three points, two regimes. This sweep asks where
# the transition is and whether it is a transition at all.
#
# Reduced K set (1/16/64/256 + the 50k table) to keep 18 modules inside ~20 minutes; the
# elbow lives at K=16 so nothing diagnostic is lost. Per-layer incremental save + resume,
# per LESSONS (heavy queue scripts must survive a kill).
#
# Registered predictions:
#   pred_a CEILING FALLS WITH DEPTH: mean table-ceiling over blocks 0-5 exceeds the mean
#          over blocks 12-17 by >= 0.15. (The un-tableable, contextual share grows upward.)
#   pred_b THE ELBOW IS A LATE-MODEL PROPERTY: the categorical index k16/table is >= 0.70
#          for at least 3 of blocks 12-17, AND <= 0.60 for every one of blocks 0-2.
#   pred_c THE STAKE IS A BARBELL (§713's rank profile, now priced by mean-ablation CE):
#          every block in 6-14 has stake < 0.5 nats, while at least one of mlp1/mlp2/mlp3
#          exceeds 3 nats.
# Diagnostics, not bars: per-layer digit-cluster count at K=16, and the K=16 frequency-
# spread-vs-shuffled-null ratio (mlp17 came in at 7x — is frequency organization also a
# late-model property?).
import json, os, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_ladder_depth_results.json'
NFIT = 1920; NR = 480; V = 50257
KS = (1, 16, 64, 256)
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'tab': None, 'mean': None}


def mlp_hook(mod, args, out):
    if CUR['mode'] is None:
        return out
    if CUR['mode'] == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    return CUR['tab'][CUR['toks']].to(out.dtype)


@torch.no_grad()
def fwd(idx):
    CUR['toks'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def kmeans(X, K, w, iters=25, seed=41):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx0 = torch.randperm(X.shape[0], generator=g)[:K].to(X.device)
    C = X[idx0].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for k in range(K):
            mk = a == k
            if mk.any():
                C[k] = (X[mk] * w[mk].unsqueeze(1)).sum(0) / w[mk].sum()
    return a, C


@torch.no_grad()
def one_layer(LI, FITR, EVR, enc):
    hk = H[LI].mlp.register_forward_hook(mlp_hook)
    sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    caps = {}
    def cap_hook(mod, args, out):
        caps['out'] = out.detach().float(); return out
    hk2 = H[LI].mlp.register_forward_hook(cap_hook)
    CUR['mode'] = None
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        flat_t = idx.reshape(-1)
        sums.index_add_(0, flat_t, caps['out'].reshape(-1, D))
        cnts.index_add_(0, flat_t, torch.ones_like(flat_t, dtype=torch.float))
    hk2.remove()
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    seen = (cnts > 0); tok_ids = torch.nonzero(seen).squeeze(1)
    Xs, ws = tab[seen], cnts[seen]

    assigns, ktabs = {}, {}
    for K in KS:
        if K == 1:
            assigns[K] = torch.zeros(V, dtype=torch.long, device=DEV)
            ktabs[K] = gmean.unsqueeze(0)
        else:
            a, C = kmeans(Xs, K, ws)
            fa = torch.zeros(V, dtype=torch.long, device=DEV); fa[tok_ids] = a
            assigns[K] = fa; ktabs[K] = C

    def ce_eval(mode):
        CUR['mode'] = mode
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            tot += float(F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                         tg[:, 16:].reshape(-1), reduction='sum'))
            n += idx.shape[0] * (T - 16)
        return tot / n

    CUR.update(mean=gmean, tab=tab)
    res = {'full': round(ce_eval(None), 4), 'mean': round(ce_eval('mean'), 4),
           'table50k': round(ce_eval('t1'), 4)}
    for K in KS:
        CUR['tab'] = ktabs[K][assigns[K]]
        res[f'k{K}'] = round(ce_eval('t1'), 4)
    hk.remove()
    stake = res['mean'] - res['full']
    rec = {k: round((res['mean'] - v) / max(stake, 1e-6), 4)
           for k, v in res.items() if k not in ('full', 'mean')}

    dig_ids = [t for t in range(V) if seen[t] and enc.decode([t]).strip().isdigit()
               and len(enc.decode([t]).strip()) == 1]
    dig_clusters = sorted(set(int(assigns[16][t]) for t in dig_ids))
    a16 = assigns[16][tok_ids]; lf = torch.log(cnts[tok_ids].clamp_min(1))
    def spread(assign):
        mus = [float(lf[assign == k].mean()) for k in range(16) if (assign == k).any()]
        return float(torch.tensor(mus).std()) if len(mus) > 1 else 0.0
    g = torch.Generator(device='cpu').manual_seed(7)
    nulls = [spread(a16[torch.randperm(a16.numel(), generator=g).to(a16.device)])
             for _ in range(3)]
    fs, fn = spread(a16), sum(nulls) / len(nulls)
    return {'layer': LI, 'ce': res, 'stake': round(stake, 4), 'recovery': rec,
            'k16_over_table': round(rec['k16'] / max(rec['table50k'], 1e-6), 4),
            'digit_clusters_at_k16': dig_clusters, 'n_digit_clusters': len(dig_clusters),
            'freq_spread_k16': round(fs, 4), 'freq_spread_null': round(fn, 4),
            'freq_ratio': round(fs / max(fn, 1e-6), 2)}


def score(layers):
    by = {d['layer']: d for d in layers}
    def ceil(li): return by[li]['recovery']['table50k']
    def cat(li): return by[li]['k16_over_table']
    front = [ceil(l) for l in range(0, 6) if l in by]
    back = [ceil(l) for l in range(12, 18) if l in by]
    pa = (len(front) == 6 and len(back) == 6 and
          sum(front) / 6 - sum(back) / 6 >= 0.15)
    late_elbow = sum(1 for l in range(12, 18) if l in by and cat(l) >= 0.70)
    early_flat = all(cat(l) <= 0.60 for l in range(0, 3) if l in by)
    pb = late_elbow >= 3 and early_flat and all(l in by for l in list(range(0, 3)) + list(range(12, 18)))
    mid_inert = all(by[l]['stake'] < 0.5 for l in range(6, 15) if l in by)
    front_big = any(by[l]['stake'] > 3.0 for l in (1, 2, 3) if l in by)
    pc = (mid_inert and front_big and
          all(l in by for l in list(range(6, 15)) + [1, 2, 3]))
    return {'pred_a_ceiling_falls_with_depth': bool(pa),
            'pred_b_elbow_is_late': bool(pb),
            'pred_c_stake_barbell': bool(pc),
            'mean_ceiling_front05': round(sum(front) / max(len(front), 1), 4),
            'mean_ceiling_back1217': round(sum(back) / max(len(back), 1), 4),
            'n_late_elbow_of_6': late_elbow, 'early_all_flat': bool(early_flat)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)
    FITR = ROWS[:NFIT, :T + 1].contiguous(); EVR = ROWS[NFIT:, :T + 1].contiguous()
    cl.assert_disjoint(FITR, EVR, label='mlp_ladder_depth')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')

    layers = []
    if os.path.exists(OUT):                      # resume
        try:
            layers = json.load(open(OUT)).get('layers', [])
            print(f'resuming: {len(layers)} layers already done', flush=True)
        except Exception:
            layers = []
    done = {d['layer'] for d in layers}
    for LI in range(18):
        if LI in done:
            continue
        try:
            d = one_layer(LI, FITR, EVR, enc)
        except Exception as e:
            print(f'mlp{LI} FAILED: {type(e).__name__} {e}', flush=True)
            continue
        layers.append(d); layers.sort(key=lambda z: z['layer'])
        print(f"mlp{LI}: stake {d['stake']:.3f} | ceiling {d['recovery']['table50k']:.3f} "
              f"| k16/tab {d['k16_over_table']:.2f} | digits {d['n_digit_clusters']} "
              f"| freq {d['freq_ratio']}x", flush=True)
        json.dump({'n_fit': NFIT, 'n_eval': NR, 'KS': list(KS), 'layers': layers,
                   'partial': True}, open(OUT, 'w'), indent=1)

    out = {'n_fit': NFIT, 'n_eval': NR, 'KS': list(KS), 'layers': layers,
           'partial': len(layers) < 18, 'runtime_s': round(time.time() - t0, 1)}
    out.update(score(layers))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\npred_a ceiling-falls {out['pred_a_ceiling_falls_with_depth']} "
          f"(front05 {out['mean_ceiling_front05']} vs back1217 {out['mean_ceiling_back1217']})")
    print(f"pred_b elbow-late {out['pred_b_elbow_is_late']} "
          f"({out['n_late_elbow_of_6']}/6 late elbows; early all flat {out['early_all_flat']})")
    print(f"pred_c stake-barbell {out['pred_c_stake_barbell']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
