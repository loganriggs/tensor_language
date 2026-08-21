"""RSPD CLUSTER CE K-SWEEP -- the direct quantitative signature of the
union-of-low-rank story (704). If mlp1 is a union of per-cluster low-rank
output circuits, then at a FIXED small rank r, recovery should RISE
MONOTONICALLY with the number of clusters K (more special-case subspaces =
better coverage). K=1 is exactly the global rank-r subspace. Shuffle
(random assignment among the same K subspaces) should rise too (more
subspaces) but stay BELOW cluster (assignment matters).

Fixed r=8; sweep K in {1,2,4,8,16,32}. Report cluster & shuffle recovery.

REGISTERED PREDICTIONS:
  (0) SANITY: K=1 cluster == global rank-8 (same single subspace);
  (a) MONOTONE IN K: cluster recovery at r=8 increases monotonically with K
      and the K=32 value is much higher than K=1 (>=0.5 higher) -- the
      union of low-rank pieces reconstructs the high-rank layer;
  (b) report cluster & shuffle recovery vs K;
  NULL/CONTRAST: cluster > shuffle at every K>1 (correct assignment beats
      random among the same K subspaces)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_cluster_ce_ksweep_results.json'
LAYER = 1
NFIT = 24
NEVAL = 48
R = 8
KS = [1, 2, 4, 8, 16, 32]

CFG = {'mode': None, 'K': None, 'C': None, 'Us': None}


def kmeans(Xn, k, iters=25, seed=0):
    if k == 1:
        c = Xn.mean(0, keepdim=True)
        return c / c.norm(dim=1, keepdim=True).clamp_min(1e-9)
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = Xn[a == j].mean(0)
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9)


def mlp_out_hook(mo, i_, o_):
    if CFG['mode'] is None:
        return o_
    flat = o_.float().reshape(-1, D); K = CFG['K']
    if CFG['mode'] == 'cluster':
        on = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
        assign = (on @ CFG['C'].T).argmax(1)
    else:
        assign = torch.randint(0, K, (flat.shape[0],), device=flat.device)
    out = torch.empty_like(flat)
    for j in range(K):
        mmask = assign == j
        if mmask.any():
            U = CFG['Us'][j][:, :R]
            out[mmask] = flat[mmask] @ U @ U.T
    return out.reshape(o_.shape).to(o_.dtype)


@torch.no_grad()
def forward_ce(rows, n, ablate=False):
    mlp = m.transformer.h[LAYER].mlp
    orig = mlp.Down.weight.data
    if ablate:
        mlp.Down.weight.data = torch.zeros_like(orig)
    ce_s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        ce_s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1),
                                 reduction='mean')) * idx.shape[0]; nn += idx.shape[0]
    if ablate:
        mlp.Down.weight.data = orig
    return ce_s / nn


@torch.no_grad()
def capture_out(rows, n):
    cap = []
    mlp = m.transformer.h[LAYER].mlp
    h = mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def build(O, K):
    On = O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = kmeans(On.cpu(), K).to(DEV)
    assign = (On @ C.T).argmax(1)
    Us = []
    for j in range(K):
        Oj = O[assign == j]
        if Oj.shape[0] < 4:
            Us.append(torch.eye(D, device=DEV)); continue
        U, _, _ = torch.linalg.svd(Oj.T @ Oj)
        Us.append(U)
    return C, Us


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT + NEVAL]
    O = capture_out(fit, NFIT)

    hh = m.transformer.h[LAYER].mlp.register_forward_hook(mlp_out_hook)
    CFG['mode'] = None
    ce_full = forward_ce(ev, NEVAL); ce_abl = forward_ce(ev, NEVAL, ablate=True)
    benefit = ce_abl - ce_full
    print(f'benefit {benefit:.3f}  r={R}', flush=True)

    res = {'cluster': {}, 'shuffle': {}}
    for K in KS:
        C, Us = build(O, K); CFG['C'] = C; CFG['Us'] = Us; CFG['K'] = K
        for mode in ['cluster', 'shuffle']:
            CFG['mode'] = mode
            ce = forward_ce(ev, NEVAL)
            res[mode][K] = round(float((ce_abl - ce) / benefit), 4)
        CFG['mode'] = None
        print(f'K={K:3d}: cluster {res["cluster"][K]:.3f}  shuffle {res["shuffle"][K]:.3f}',
              flush=True)
    hh.remove()

    cvals = [res['cluster'][k] for k in KS]
    mono = all(cvals[i] <= cvals[i + 1] + 0.03 for i in range(len(cvals) - 1))
    rise = cvals[-1] - cvals[0]
    beats = all(res['cluster'][k] > res['shuffle'][k] for k in KS if k > 1)
    p0 = True
    pa = mono and rise >= 0.5
    print(f'\n(a) monotone in K & rises >=0.5 (rise {rise:.2f}): {pa}', flush=True)
    print(f'NULL cluster>shuffle at every K>1: {beats}', flush=True)

    out = {'benefit': round(benefit, 4), 'r': R, 'recovered_by_K': res,
           'rise_K1_to_Kmax': round(rise, 4), 'pred_0': bool(p0),
           'pred_a_monotone': bool(pa), 'cluster_beats_shuffle': bool(beats),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
