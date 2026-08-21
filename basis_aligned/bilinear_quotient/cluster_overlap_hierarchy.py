"""CLUSTER OVERLAP HIERARCHY (user observation): the mlp1 clusters are not
a clean tree -- they OVERLAP, so a group can be a sub-part of two parents
(an overlapping hierarchy / DAG). Measure directional SUBSPACE CONTAINMENT
between the K cluster output-subspaces: containment(i->j) = average fraction
of cluster i's top-r directions that lie inside cluster j's top-r subspace
(||U_j U_j^T U_i||_F^2 / r). High containment(i->j) means cluster i is a
sub-part of j. A cluster contained in >=2 others = overlapping hierarchy.
Also token-level soft overlap (ambiguous membership).

REGISTERED PREDICTIONS:
  (0) SANITY: diagonal containment = 1; random-subspace containment ~ r/D
      (=16/1152=0.014), so any large off-diagonal is real structure;
  (a) OVERLAPPING HIERARCHY (user's obs): >=1 cluster is strongly contained
      (>=0.6) in TWO OR MORE others -> the structure is a DAG, not a tree;
      report the containment matrix and the multi-parent clusters;
  (b) report token-level soft overlap: fraction of tokens whose 2nd-best
      centroid cosine is within 15% of the best (ambiguous membership);
  NULL: random unit subspaces of rank r have mean pairwise containment
      << the observed (near r/D)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE, BLUES
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; LAYER = 1; NFIT = 16; K = 8; R = 16
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_overlap_hierarchy_results.json'


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


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    O, toks = capture(rows, NFIT); tk = toks.numpy()
    On = (O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu()
    C, assign = kmeans(On, K); assign = assign.numpy()

    # per-cluster orthonormal top-R subspace of outputs
    Us = []
    labels = []
    for j in range(K):
        Oj = O[assign == j]
        U, S, Vh = torch.linalg.svd(Oj.T @ Oj)
        Us.append(U[:, :R])                       # (D, R) orthonormal
        labels.append('/'.join(d1(t) for t, _ in Counter(tk[assign == j].tolist()).most_common(3)))

    # directional containment C[i,j] = ||U_j U_j^T U_i||_F^2 / R
    cont = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            P = Us[j] @ (Us[j].T @ Us[i])         # project U_i onto U_j subspace
            cont[i, j] = float((P.norm() ** 2) / R)

    # random-subspace null
    g = torch.Generator().manual_seed(0)
    rn = []
    for _ in range(8):
        Q, _ = torch.linalg.qr(torch.randn(D, R, generator=g)); Q2, _ = torch.linalg.qr(torch.randn(D, R, generator=g))
        P = Q2 @ (Q2.T @ Q); rn.append(float((P.norm()**2)/R))
    null_mean = float(np.mean(rn))

    # multi-parent clusters: contained (>=0.6) in >=2 OTHERS
    multi = []
    for i in range(K):
        parents = [j for j in range(K) if j != i and cont[i, j] >= 0.6]
        if len(parents) >= 2:
            multi.append({'cluster': i, 'label': labels[i], 'parents': parents,
                          'parent_labels': [labels[j] for j in parents]})

    # token soft overlap: 2nd-best centroid cosine within 15% of best
    sims = (On @ C.T).numpy()
    srt = np.sort(sims, 1)[:, ::-1]
    ambiguous = float(np.mean(srt[:, 1] >= 0.85 * srt[:, 0]))

    print('containment matrix (row i contained in col j):', flush=True)
    for i in range(K):
        print(f'  c{i} [{labels[i][:18]:18s}] ' + ' '.join(f'{cont[i,j]:.2f}' for j in range(K)), flush=True)
    print(f'\nrandom-subspace null containment {null_mean:.3f} (r/D={R/D:.3f})', flush=True)
    print(f'multi-parent (contained>=0.6 in >=2) clusters: {len(multi)}', flush=True)
    for mm in multi:
        print(f'  c{mm["cluster"]} [{mm["label"]}] < parents {mm["parents"]} '
              f'{mm["parent_labels"]}', flush=True)
    print(f'token ambiguous-membership fraction: {ambiguous:.2f}', flush=True)

    # heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.2)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    im = ax.imshow(cont, cmap=BLUES, vmin=0, vmax=1, origin='upper')
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f'c{j}' for j in range(K)], fontsize=9)
    ax.set_yticklabels([f'c{i} {labels[i][:16]}' for i in range(K)], fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{cont[i,j]:.2f}', ha='center', va='center', fontsize=7.5,
                    color=INK if cont[i,j] < 0.6 else SURFACE)
    ax.set_xlabel('contained IN cluster j (parent) ->', fontsize=10)
    ax.set_ylabel('<- cluster i (child)', fontsize=10)
    ax.set_title('mlp1 cluster subspace CONTAINMENT (directional)\n'
                 'row i contained in col j; >0.6 = i is a sub-part of j (overlapping hierarchy)',
                 fontsize=10.5, color=INK, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label('containment fraction', fontsize=9)
    fig.tight_layout()
    fig.savefig(PT + 'cluster_overlap_hierarchy.png', dpi=150, facecolor=SURFACE)
    print('wrote cluster_overlap_hierarchy.png', flush=True)

    pa = len(multi) >= 1
    null_ok = null_mean < 0.1
    out = {'containment': cont.round(3).tolist(), 'labels': labels,
           'null_containment': round(null_mean, 4), 'multi_parent': multi,
           'ambiguous_frac': round(ambiguous, 3), 'R': R, 'K': K,
           'pred_a_overlapping': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) overlapping hierarchy ({len(multi)} multi-parent): {pa}; NULL: {null_ok}')
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
