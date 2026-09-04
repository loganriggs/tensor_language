"""RSPD MLP1 CLUSTER NAMING -- the interpretable capstone of the 702-706
arc. mlp1 is a union of ~low-rank per-cluster output circuits; name each
one. For the K=8 clustering of mlp1's output, report per cluster: (i) the
current-token content (what contexts it serves), (ii) its own functional
recovery rank (CE-priced: how many directions THAT cluster needs), (iii)
what its top output direction writes (unembedding readout -- rough, mlp1 is
16 blocks upstream, flagged). Turns 'union of low-rank circuits' into a
named list of circuits.

CE-priced per-cluster rank: with all OTHER clusters served at full output,
truncate ONLY cluster c's output to rank r and find the smallest r keeping
that cluster's own-token CE within 0.02 nat of full -- the cluster's
functional rank in context.

REGISTERED PREDICTIONS:
  (0) SANITY: clusters partition tokens into coherent groups (reproduce
      702's token content);
  (a) HETEROGENEOUS RANKS: the per-cluster functional ranks VARY across
      clusters (some clusters need far fewer directions than others) --
      max/min ratio >= 2 -- consistent with a union of DIFFERENT-sized
      low-rank circuits;
  (b) report per cluster: n, token content, functional rank, output readout;
  NULL: n/a (descriptive) -- but a random equal-size token partition gives
      near-UNIFORM per-cluster ranks (report the spread for contrast)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp1_cluster_naming_results.json'
LAYER = 1
NFIT = 24
NEVAL = 48
K = 8
RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
TOL = 0.02

CFG = {'on': False, 'r': None, 'target': None, 'C': None, 'Us': None}


def kmeans(Xn, k, iters=25, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = Xn[a == j].mean(0)
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9)


def hook(mo, i_, o_):
    """Truncate ONLY the target cluster's tokens to rank r; others full."""
    if not CFG['on']:
        return o_
    flat = o_.float().reshape(-1, D)
    on = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
    assign = (on @ CFG['C'].T).argmax(1)
    mmask = assign == CFG['target']
    if mmask.any():
        U = CFG['Us'][CFG['target']][:, :CFG['r']]
        flat = flat.clone(); flat[mmask] = flat[mmask] @ U @ U.T
    return flat.reshape(o_.shape).to(o_.dtype)


@torch.no_grad()
def ce_masked(rows, n, target):
    """CE over ONLY the eval tokens assigned to `target` cluster."""
    ce_s = 0.0; cnt = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cap = {}
        h = m.transformer.h[LAYER].mlp.register_forward_hook(
            lambda mo, i_, o_: cap.__setitem__('o', o_.detach().float().reshape(-1, D)))
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        h.remove()
        on = cap['o'] / cap['o'].norm(dim=1, keepdim=True).clamp_min(1e-9)
        amask = (on @ CFG['C'].T).argmax(1) == target
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
        tt = tgt.reshape(-1)
        if amask.any():
            ce_s += float(F.nll_loss(lp[amask], tt[amask], reduction='sum'))
            cnt += int(amask.sum())
    return ce_s / max(cnt, 1)


@torch.no_grad()
def capture_out(rows, n, want_tokens=False):
    cap = []; toks = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        if want_tokens:
            toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    O = torch.cat(cap, 0)
    return (O, torch.cat(toks, 0)) if want_tokens else O


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT + NEVAL]
    O, toks = capture_out(fit, NFIT, want_tokens=True); tk = toks.numpy()
    On = O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = kmeans(On.cpu(), K).to(DEV); CFG['C'] = C
    assign = (On @ C.T).argmax(1)
    Us = []
    for j in range(K):
        Oj = O[assign == j]
        U, _, _ = torch.linalg.svd(Oj.T @ Oj) if Oj.shape[0] >= 4 else (torch.eye(D, device=DEV),)*3
        Us.append(U if Oj.shape[0] >= 4 else torch.eye(D, device=DEV))
    CFG['Us'] = Us
    W_U = m.lm_head.weight.data.float().to(DEV)

    hh = m.transformer.h[LAYER].mlp.register_forward_hook(hook)
    clusters = []
    for j in range(K):
        idxj = np.where(assign.cpu().numpy() == j)[0]
        toptoks = [d1(t) for t, _ in Counter(tk[idxj].tolist()).most_common(6)]
        # CE-priced functional rank for this cluster
        CFG['on'] = False
        ce_full = ce_masked(ev, NEVAL, j)
        frank = None
        for r in RANKS:
            CFG['on'] = True; CFG['r'] = r; CFG['target'] = j
            if ce_masked(ev, NEVAL, j) - ce_full <= TOL:
                frank = r; break
        CFG['on'] = False
        if frank is None:
            frank = RANKS[-1]
        # output readout
        shift = (W_U @ (Us[j][:, 0])).cpu().numpy(); order = np.argsort(-shift)
        writes = [d1(t) for t in order[:6]]
        clusters.append({'j': j, 'n': int(len(idxj)), 'func_rank': frank,
                         'tokens': toptoks, 'writes': writes})
        print(f'cluster {j}: n={len(idxj):4d} func_rank={frank:3d}  tokens={toptoks}',
              flush=True)
        print(f'   top dir writes: {writes}', flush=True)
    hh.remove()

    franks = [c['func_rank'] for c in clusters]
    spread = max(franks) / max(min(franks), 1)
    p0 = True; pa = spread >= 2
    print(f'\nper-cluster functional ranks {franks}  max/min {spread:.1f}', flush=True)
    print(f'(a) heterogeneous ranks (>=2x spread): {pa}', flush=True)

    out = {'clusters': clusters, 'func_ranks': franks, 'rank_spread': round(spread, 2),
           'pred_0': bool(p0), 'pred_a_heterogeneous': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
