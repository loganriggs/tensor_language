"""Answer two user questions:
 A) 703 RANDOM BASELINE + data: block0.attn rank-2 A-SVD recovered vs a
    RANDOM rank-2 projection, at N=3k and N=12k tokens.
 B) 704 DATA-ROBUSTNESS: mlp1 cluster/global/shuffle recovered at r=8, at
    fit N=24 rows (~6k) and N=96 rows (~24k) -- does more data change the
    cluster>global conclusion?
No new predictions; robustness/baseline reporting (controls are the point)."""
import sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'


def asvd_fast(W, X, eps=1e-3):
    U, S, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    G = X.T @ X; G.diagonal().add_(eps)
    return U * S, torch.linalg.solve(G, (Vh @ X).T).T


@torch.no_grad()
def forward_ce(rows, n, mod=None, W=None):
    orig = None
    if mod is not None:
        orig = mod.weight.data
        mod.weight.data = (torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype)) if W is not None else orig
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
        lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if orig is not None: mod.weight.data = orig
    return s/nn


@torch.no_grad()
def capture(mod, rows, n, in_dim):
    cap = []
    h = mod.register_forward_pre_hook(lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def partA(rows, ev, nev):
    print('\n=== A) block0.attn rank-2: A-SVD vs RANDOM baseline (703) ===', flush=True)
    mod = m.transformer.h[0].attn.c_proj; W = mod.weight.data.float().to(DEV)
    ce_full = forward_ce(ev, nev); ce_abl = forward_ce(ev, nev, mod, 'ablate')
    ben = ce_abl - ce_full
    out = {}
    for nr in [12, 48]:
        X = capture(mod, rows[:nr], nr, 1152)
        A, B = asvd_fast(W, X)
        rec_asvd = (ce_abl - forward_ce(ev, nev, mod, A[:, :2] @ B[:2, :])) / ben
        g = torch.Generator().manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g)); Q = Q.to(DEV)
        recs = []
        for seed in range(3):
            gg = torch.Generator().manual_seed(seed)
            Qq, _ = torch.linalg.qr(torch.randn(D, D, generator=gg)); Qr = Qq[:, :2].to(DEV)
            recs.append(float((ce_abl - forward_ce(ev, nev, mod, Qr @ (Qr.T @ W))) / ben))
        out[nr*256] = {'asvd_r2': round(float(rec_asvd), 4),
                       'random_r2_mean': round(float(np.mean(recs)), 4),
                       'random_r2_all': [round(r, 4) for r in recs]}
        print(f'N={nr*256:5d}: A-SVD rank-2 recovered {rec_asvd:.3f} | '
              f'random rank-2 {np.mean(recs):+.3f} (3 seeds {[round(r,2) for r in recs]})', flush=True)
    return {'benefit': round(ben, 4), 'by_N': out}


CFG = {'mode': None, 'r': 8, 'C': None, 'Us': None, 'K': 8}
def hook(mo, i_, o_):
    if CFG['mode'] is None: return o_
    flat = o_.float().reshape(-1, D); K = CFG['K']
    if CFG['mode'] == 'global':
        U = CFG['Us'][0][:, :CFG['r']]; flat = flat @ U @ U.T
    else:
        if CFG['mode'] == 'cluster':
            on = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
            a = (on @ CFG['C'].T).argmax(1)
        else:
            a = torch.randint(0, K, (flat.shape[0],), device=flat.device)
        outp = torch.empty_like(flat)
        for j in range(K):
            mm = a == j
            if mm.any(): U = CFG['Us'][j][:, :CFG['r']]; outp[mm] = flat[mm] @ U @ U.T
        flat = outp
    return flat.reshape(o_.shape).to(o_.dtype)


def kmeans(Xn, k, seed=0):
    g = torch.Generator().manual_seed(seed); C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(25):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9), a


@torch.no_grad()
def partB(rows, ev, nev):
    print('\n=== B) mlp1 cluster/global/shuffle r=8: data-robustness (704) ===', flush=True)
    mlp = m.transformer.h[1].mlp
    def capout(rr, n):
        cap = []
        h = mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
        for i in range(0, n, 4):
            bb = rr[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        h.remove(); return torch.cat(cap, 0)
    hh = mlp.register_forward_hook(hook)
    CFG['mode'] = None
    ce_full = forward_ce(ev, nev); ce_abl = forward_ce(ev, nev, mlp.Down, 'ablate'); ben = ce_abl - ce_full
    out = {}
    for nfit in [24, 96]:
        O = capout(rows[:nfit], nfit); On = (O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu()
        C, assign = kmeans(On, 8); CFG['C'] = C.to(DEV)
        Us = []
        for j in range(8):
            Oj = O[assign.numpy() == j]
            U, _, _ = torch.linalg.svd(Oj.T @ Oj) if Oj.shape[0] >= 4 else (torch.eye(D, device=DEV),)*3
            Us.append(U if Oj.shape[0] >= 4 else torch.eye(D, device=DEV))
        # global = single subspace over all
        Ug, _, _ = torch.linalg.svd(O.T @ O)
        r = {}
        for mode, US in [('cluster', Us), ('global', [Ug]), ('shuffle', Us)]:
            CFG['mode'] = mode; CFG['Us'] = US
            r[mode] = round(float((ce_abl - forward_ce(ev, nev)) / ben), 4); CFG['mode'] = None
        out[nfit*256] = r
        print(f'fit N={nfit*256:5d}: cluster {r["cluster"]:+.3f}  global {r["global"]:+.3f}  '
              f'shuffle {r["shuffle"]:+.3f}', flush=True)
    hh.remove()
    return {'benefit': round(ben, 4), 'by_fitN': out}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(96 + 48)
    ev = rows[96:96+48]; nev = 48
    res = {'A_703_random_baseline': partA(rows, ev, nev),
           'B_704_data_robustness': partB(rows, ev, nev), 'runtime_s': time.time()-t0}
    json.dump(res, open(PT + 'robustness_and_baseline_results.json', 'w'), indent=1)
    print(f'\nwrote results ({res["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
