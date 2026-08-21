"""RSPD CLUSTER CE -- the CE-honest test of the user's cluster-then-low-rank
idea, in FUNCTIONAL space (702 did it in energy space, which over-weights
loss-irrelevant directions). Question: can a UNION of per-cluster low-rank
OUTPUT subspaces reproduce a high-rank layer (mlp1) better than a single
global low-rank subspace at the same rank r?

Method: fit clusters on mlp1's Down-OUTPUT direction over a fit set; per
cluster, take the top-r left-singular directions of that cluster's outputs
(its own subspace). At eval, hook mlp1's output; for each token assign it
to the nearest cluster centroid and REPLACE its output with the projection
onto that cluster's rank-r subspace. Price by real CE. Compare to a GLOBAL
rank-r output projection (one subspace for all) and to a SHUFFLED-centroid
control (random assignment).

recovered(r) = (CE_ablate - CE_r) / (CE_ablate - CE_full), where CE_ablate
zeroes mlp1's output (keeps bias). If clustered > global at small r, the
high-rank layer IS a union of per-cluster low-rank functional pieces.

REGISTERED PREDICTIONS:
  (0) SANITY: at full rank, both global and clustered projection reproduce
      baseline CE (projection onto a full basis is identity);
  (a) HYPOTHESIS: clustered rank-r recovers MORE than global rank-r at the
      same r, for small r (e.g. r in {4,8,16}) -- per-cluster subspaces
      serve their own tokens better;
  (b) report recovered(r) for global / clustered / shuffled across r;
  NULL: SHUFFLED-centroid assignment (random clusters) does NOT beat global
      (any gain must come from real token->subspace structure)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_cluster_ce_results.json'
LAYER = 1
NFIT = 24
NEVAL = 48
K = 8
RANKS = [4, 8, 16, 32, 64, 128]

CFG = {'mode': None, 'r': None, 'C': None, 'Us': None, 'Ug': None, 'assign_rand': None}


def kmeans(Xn, k, iters=25, seed=0):
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
    flat = o_.float().reshape(-1, D); r = CFG['r']
    if CFG['mode'] == 'global':
        U = CFG['Ug'][:, :r]
        flat = flat @ U @ U.T
    else:
        if CFG['mode'] == 'cluster':
            on = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
            assign = (on @ CFG['C'].T).argmax(1)
        else:   # shuffle: random per-batch assignment (the null)
            assign = torch.randint(0, K, (flat.shape[0],), device=flat.device)
        out = torch.empty_like(flat)
        for j in range(K):
            mmask = assign == j
            if mmask.any():
                U = CFG['Us'][j][:, :r]
                out[mmask] = flat[mmask] @ U @ U.T
        flat = out
    return flat.reshape(o_.shape).to(o_.dtype)


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
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT + NEVAL]

    O = capture_out(fit, NFIT)                       # (Nfit, D) mlp1 outputs
    On = O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = kmeans(On.cpu(), K).to(DEV)
    assign = (On @ C.T).argmax(1)
    Us = []
    for j in range(K):
        Oj = O[assign == j]
        if Oj.shape[0] < 4:
            Us.append(torch.eye(D, device=DEV)); continue
        U, S, Vh = torch.linalg.svd(Oj.T @ Oj)       # left basis of cluster outputs
        Us.append(U)
    Ug, _, _ = torch.linalg.svd(O.T @ O)
    CFG['C'] = C; CFG['Us'] = Us; CFG['Ug'] = Ug

    hh = m.transformer.h[LAYER].mlp.register_forward_hook(mlp_out_hook)

    CFG['mode'] = None
    ce_full = forward_ce(ev, NEVAL)
    ce_abl = forward_ce(ev, NEVAL, ablate=True)
    benefit = ce_abl - ce_full
    print(f'benefit {benefit:.3f}  (CE_full {ce_full:.3f} CE_ablate {ce_abl:.3f})', flush=True)

    res = {'global': {}, 'cluster': {}, 'shuffle': {}}
    for r in RANKS:
        CFG['r'] = r
        for mode in ['global', 'cluster', 'shuffle']:
            CFG['mode'] = mode
            ce = forward_ce(ev, NEVAL)
            res[mode][r] = round(float((ce_abl - ce) / benefit), 4)
        CFG['mode'] = None
        print(f"r={r:4d}: global {res['global'][r]:.3f}  cluster {res['cluster'][r]:.3f}  "
              f"shuffle {res['shuffle'][r]:.3f}", flush=True)
    hh.remove()

    gains = [res['cluster'][r] - res['global'][r] for r in [4, 8, 16] if r in res['cluster']]
    pa = all(gv > 0.02 for gv in gains)
    shuf_gain = max(res['shuffle'][r] - res['global'][r] for r in RANKS)
    null_ok = shuf_gain < 0.02
    print(f'\n(a) cluster beats global at small r (gains {[round(x,3) for x in gains]}): {pa}',
          flush=True)
    print(f'NULL shuffle does not beat global (max gain {shuf_gain:.3f}<0.02): {null_ok}',
          flush=True)

    out = {'benefit': round(benefit, 4), 'recovered': res,
           'cluster_gain_small_r': [round(x, 4) for x in gains],
           'shuffle_max_gain': round(float(shuf_gain), 4),
           'pred_a_cluster_beats_global': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
