"""CLUSTER RIGOR VALIDATION (my offer to the user): confirm the key result
-- per-cluster rank-r output subspaces beat a single global rank-r subspace
in CE (704) -- is NOT an artifact of the simple single-init k-means.
Compare THREE clusterings of mlp1's output (32k tokens):
  (1) simple k-means (random-datapoint init, seed 0) -- the original;
  (2) k-means++ init with 5 restarts (best inertia) -- more principled;
  (3) shuffled assignment -- the null.
Measure the 704 metric: recovered(r=8) = (CE_ablate - CE_r)/(CE_ablate -
CE_full) with mlp1's output replaced per-token by its cluster's rank-8
subspace. Also report partition STABILITY: adjusted Rand index (ARI)
between the 5 k-means++ restarts.

REGISTERED PREDICTIONS:
  (0) SANITY: CE_full/ablate reproduce; simple k-means recovery ~ 704;
  (a) NOT AN ARTIFACT: k-means++ (best of 5) recovers cluster>>global at
      r=8 just like simple k-means (within 0.1), and both >> shuffle -- the
      finding survives a better clusterer;
  (b) report recovery for all three + ARI stability of k-means++ restarts;
  NULL: shuffle recovery << cluster for both clusterers."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1; NFIT = 128; NEVAL = 96; K = 8; R = 8
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_rigor_validation_results.json'
CFG = {'mode': None, 'C': None, 'Us': None}


def kmeans_simple(Xn, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9), a


def kmeanspp(Xn, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = Xn.shape[0]
    # D^2 seeding
    first = int(torch.randint(0, n, (1,), generator=g))
    cen = [first]
    d2 = torch.cdist(Xn, Xn[first:first+1]).squeeze(1) ** 2
    for _ in range(k - 1):
        probs = d2 / d2.sum().clamp_min(1e-9)
        nxt = int(torch.multinomial(probs, 1, generator=g))
        cen.append(nxt)
        nd = torch.cdist(Xn, Xn[nxt:nxt+1]).squeeze(1) ** 2
        d2 = torch.minimum(d2, nd)
    C = Xn[cen].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    inertia = float((torch.cdist(Xn, C).min(1).values ** 2).sum())
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9), a, inertia


def ari(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    from collections import Counter
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    cont = {}
    for i in range(n): cont[(a[i], b[i])] = cont.get((a[i], b[i]), 0) + 1
    sum_c = sum(v * (v - 1) / 2 for v in cont.values())
    sa = sum(v * (v - 1) / 2 for v in ca.values())
    sb = sum(v * (v - 1) / 2 for v in cb.values())
    tot = n * (n - 1) / 2
    exp = sa * sb / tot
    return (sum_c - exp) / (0.5 * (sa + sb) - exp + 1e-12)


def hook(mo, i_, o_):
    if CFG['mode'] is None: return o_
    flat = o_.float().reshape(-1, D)
    if CFG['mode'] == 'global':
        U = CFG['Us'][0][:, :R]; flat = flat @ U @ U.T
    else:
        if CFG['mode'] == 'cluster':
            on = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
            a = (on @ CFG['C'].T).argmax(1)
        else:
            a = torch.randint(0, K, (flat.shape[0],), device=flat.device)
        out = torch.empty_like(flat)
        for j in range(K):
            mm = a == j
            if mm.any(): out[mm] = flat[mm] @ CFG['Us'][j][:, :R] @ CFG['Us'][j][:, :R].T
        flat = out
    return flat.reshape(o_.shape).to(o_.dtype)


@torch.no_grad()
def forward_ce(rows, n, ablate=False):
    mlp = m.transformer.h[LAYER].mlp; orig = mlp.Down.weight.data
    if ablate: mlp.Down.weight.data = torch.zeros_like(orig)
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if ablate: mlp.Down.weight.data = orig
    return s/nn


@torch.no_grad()
def capture(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def subspaces(O, assign):
    Us = []
    for j in range(K):
        Oj = O[assign == j]
        U, _, _ = torch.linalg.svd(Oj.T @ Oj) if Oj.shape[0] >= R+2 else (torch.eye(D, device=DEV),)*3
        Us.append(U if Oj.shape[0] >= R+2 else torch.eye(D, device=DEV))
    return Us


@torch.no_grad()
def measure(O, C, assign):
    Us = subspaces(O, assign); Ug, _, _ = torch.linalg.svd(O.T @ O)
    CFG['C'] = C.to(DEV)
    res = {}
    for mode, US in [('cluster', Us), ('global', [Ug]), ('shuffle', Us)]:
        CFG['mode'] = mode; CFG['Us'] = US
        res[mode] = round(float((ce_abl - forward_ce(ev_g, NEVAL))/ben), 4); CFG['mode'] = None
    return res


@torch.no_grad()
def main():
    global ev_g, ce_abl, ben
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev_g = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    O = capture(fit, NFIT); On = (O / O.norm(dim=1, keepdim=True).clamp_min(1e-9))
    Onc = On.cpu()
    hh = m.transformer.h[LAYER].mlp.register_forward_hook(hook)
    CFG['mode'] = None
    ce_full = forward_ce(ev_g, NEVAL); ce_abl = forward_ce(ev_g, NEVAL, ablate=True); ben = ce_abl - ce_full
    print(f'benefit {ben:.3f}  r={R}', flush=True)

    # (1) simple
    Cs, asg_s = kmeans_simple(Onc, K, seed=0)
    r_simple = measure(O, Cs, asg_s.numpy())
    print(f'simple k-means:  {r_simple}', flush=True)

    # (2) k-means++ x5 restarts
    best = None; assigns = []
    for s in range(5):
        Cpp, asg, inert = kmeanspp(Onc, K, seed=s)
        assigns.append(asg.numpy())
        if best is None or inert < best[2]: best = (Cpp, asg, inert)
    r_pp = measure(O, best[0], best[1].numpy())
    aris = [ari(assigns[i], assigns[j]) for i in range(5) for j in range(i+1, 5)]
    mean_ari = float(np.mean(aris))
    print(f'k-means++ (best/5): {r_pp}  restart-ARI {mean_ari:.2f}', flush=True)
    hh.remove()

    not_artifact = (abs(r_pp['cluster'] - r_simple['cluster']) < 0.1 and
                    r_pp['cluster'] > r_pp['global'] + 0.3)
    null_ok = r_simple['cluster'] > r_simple['shuffle'] and r_pp['cluster'] > r_pp['shuffle']
    print(f'\n(a) not an artifact (kmeans++ ~ simple, cluster>>global): {not_artifact}', flush=True)
    print(f'NULL cluster>shuffle both: {null_ok}; restart stability ARI {mean_ari:.2f}', flush=True)

    out = {'benefit': round(ben, 4), 'simple': r_simple, 'kmeanspp': r_pp,
           'restart_ari_mean': round(mean_ari, 3), 'restart_aris': [round(a, 3) for a in aris],
           'not_artifact': bool(not_artifact), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
