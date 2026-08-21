"""CLUSTER NESTING CROSS-K (follow-up to 714 + user's hint that true
nesting might appear at finer granularity). Cluster mlp1 outputs at COARSE
K=4 and FINE K=16. Measure each FINE cluster's subspace containment into
each COARSE cluster's subspace. If each fine cluster nests cleanly inside
exactly ONE coarse cluster (>0.6 to one, <0.3 to the rest), the structure
is a TREE. If many fine clusters are strongly contained in TWO+ coarse
clusters, it is an OVERLAPPING hierarchy / DAG (the user's picture).

REGISTERED PREDICTIONS:
  (0) SANITY: fine subspaces are contained in coarse ones far above the
      random baseline (r_fine/D);
  (a) TEST (no strong prior): classify each fine cluster as TREE-like
      (strong nesting in exactly 1 coarse) vs OVERLAP (strong in >=2). If
      >=25% of fine clusters are OVERLAP -> overlapping hierarchy confirmed
      at cross-granularity; if nearly all are TREE-like -> clean hierarchy
      appears when you separate scales. Report the split;
  (b) report the fine->coarse containment matrix + token labels;
  NULL: random fine subspaces are not strongly contained in any coarse
      cluster (max containment ~ r_fine/D)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; LAYER = 1; NFIT = 128   # ~32k tokens (user: stop under-powering on data)
KC, KF = 4, 16          # coarse / fine cluster counts
RC, RF = 24, 8          # coarse / fine subspace ranks
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_nesting_crossK_results.json'


def kmeans(Xn, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-9), a


@torch.no_grad()
def capture(rows, n):
    cap = []; toks = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


def subspace(O, assign, k, r):
    Us = []
    for j in range(k):
        Oj = O[assign == j]
        if Oj.shape[0] < r + 2:
            Us.append(None); continue
        U, _, _ = torch.linalg.svd(Oj.T @ Oj); Us.append(U[:, :r])
    return Us


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    O, toks = capture(rows, NFIT); tk = toks.numpy()
    On = (O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu()
    _, ac = kmeans(On, KC, seed=0); ac = ac.numpy()
    _, af = kmeans(On, KF, seed=1); af = af.numpy()
    Uc = subspace(O, ac, KC, RC)
    Uf = subspace(O, af, KF, RF)

    def label(assign, j):
        return '/'.join(d1(t) for t, _ in Counter(tk[assign == j].tolist()).most_common(3))
    clab = [label(ac, j) for j in range(KC)]

    cont = np.full((KF, KC), np.nan)
    for i in range(KF):
        if Uf[i] is None: continue
        for j in range(KC):
            if Uc[j] is None: continue
            P = Uc[j] @ (Uc[j].T @ Uf[i])
            cont[i, j] = float((P.norm() ** 2) / RF)

    # random null
    g = torch.Generator().manual_seed(0)
    Qc, _ = torch.linalg.qr(torch.randn(D, RC, generator=g))
    rn = []
    for _ in range(8):
        Qf, _ = torch.linalg.qr(torch.randn(D, RF, generator=g))
        P = Qc @ (Qc.T @ Qf); rn.append(float((P.norm()**2)/RF))
    null_mean = float(np.mean(rn))

    tree = overlap = 0; rows_out = []
    for i in range(KF):
        if Uf[i] is None: continue
        row = cont[i]; strong = [j for j in range(KC) if row[j] >= 0.6]
        kind = 'tree' if len(strong) == 1 else ('overlap' if len(strong) >= 2 else 'none')
        if kind == 'tree': tree += 1
        elif kind == 'overlap': overlap += 1
        flab = label(af, i)
        rows_out.append({'fine': i, 'label': flab, 'kind': kind,
                         'containment': [round(float(x), 2) if not np.isnan(x) else None for x in row],
                         'strong_parents': strong})
        print(f'fine c{i:2d} [{flab[:16]:16s}] {kind:7s} into coarse '
              + ' '.join(f'{clab[j][:8]}:{row[j]:.2f}' for j in range(KC)), flush=True)

    tot = tree + overlap
    frac_overlap = overlap / max(tot, 1)
    print(f'\nrandom null containment {null_mean:.3f} (RF/D={RF/D:.3f})', flush=True)
    print(f'tree-like {tree}, overlap {overlap} -> overlap frac {frac_overlap:.2f}', flush=True)

    p0 = np.nanmax(cont) > 5 * null_mean
    pa = frac_overlap >= 0.25
    print(f'(0) fine nests in coarse above random: {p0}', flush=True)
    print(f'(a) overlapping hierarchy at cross-granularity (>=25% overlap): {pa}', flush=True)

    out = {'coarse_labels': clab, 'fine': rows_out, 'null_containment': round(null_mean, 4),
           'n_tree': tree, 'n_overlap': overlap, 'overlap_frac': round(frac_overlap, 3),
           'pred_0': bool(p0), 'pred_a_overlap': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
