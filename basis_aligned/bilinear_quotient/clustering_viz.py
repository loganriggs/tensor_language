"""Visualize the mlp1 output clustering: (1) the token-token cosine-
similarity matrix ordered by cluster (block-diagonal = the clusters), with
a cluster-color sidebar and a colorbar; (2) the KxK cluster-centroid
similarity. Also print cluster sizes + token content. Descriptive."""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE, DIVERGING
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; LAYER = 1; NFIT = 12; K = 8; PERC = 60
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'


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
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    O, toks = capture(rows, NFIT); tk = toks.numpy()
    On = (O / O.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu()
    C, assign = kmeans(On, K); assign = assign.numpy()

    # subsample PERC tokens per cluster, order by cluster
    order = []; bounds = [0]; sizes = []
    rng = np.random.default_rng(0)
    for j in range(K):
        ij = np.where(assign == j)[0]
        sizes.append(len(ij))
        pick = rng.choice(ij, size=min(PERC, len(ij)), replace=False)
        order.extend(pick.tolist()); bounds.append(len(order))
    order = np.array(order)
    S = (On[order] @ On[order].T).numpy()          # cosine-sim, ordered by cluster
    Kc = (C @ C.T).numpy()                          # KxK centroid similarity

    # colors per cluster
    cmapK = plt.cm.tab10(np.linspace(0, 1, 10))
    rowcol = np.array([cmapK[assign[i] % 10] for i in order])

    fig = plt.figure(figsize=(11, 5.6)); fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(1, 3, width_ratios=[0.05, 1, 0.62], wspace=0.28,
                  left=0.06, right=0.97, top=0.9, bottom=0.1)
    # cluster sidebar ("c bar")
    axc = fig.add_subplot(gs[0, 0]); axc.set_facecolor(SURFACE)
    axc.imshow(rowcol[:, None, :], aspect='auto', origin='upper')
    axc.set_xticks([]); axc.set_yticks([]); axc.set_ylabel('token (by cluster)', fontsize=9)
    for s in axc.spines.values(): s.set_visible(False)
    # similarity heatmap
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(SURFACE)
    im = ax.imshow(S, cmap=DIVERGING, vmin=-1, vmax=1, aspect='auto', origin='upper')
    for b in bounds[1:-1]:
        ax.axhline(b-0.5, color=INK, lw=0.6); ax.axvline(b-0.5, color=INK, lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('mlp1 output cosine-similarity, tokens ordered by cluster\n'
                 '(block-diagonal blocks = the 8 clusters)', fontsize=11, color=INK, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label('cosine similarity', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    # KxK centroid similarity
    ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(SURFACE)
    im2 = ax2.imshow(Kc, cmap=DIVERGING, vmin=-1, vmax=1, origin='upper')
    ax2.set_xticks(range(K)); ax2.set_yticks(range(K))
    ax2.set_xticklabels(range(K), fontsize=8); ax2.set_yticklabels(range(K), fontsize=8)
    ax2.set_title('cluster-centroid similarity (KxK)', fontsize=10, color=INK, loc='left')
    for i in range(K):
        for j in range(K):
            ax2.text(j, i, f'{Kc[i,j]:.1f}', ha='center', va='center', fontsize=6.5,
                     color=INK if abs(Kc[i,j]) < 0.6 else SURFACE)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02); cb2.ax.tick_params(labelsize=8)

    out = PT + 'clustering_viz.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE); print('wrote', out)

    print(f'\nK={K} clusters of mlp1 output (N={len(tk)} tokens):')
    det = []
    for j in range(K):
        ij = np.where(assign == j)[0]
        toptoks = [d1(t) for t, _ in Counter(tk[ij].tolist()).most_common(6)]
        offdiag = np.mean([Kc[j, l] for l in range(K) if l != j])
        det.append({'j': j, 'n': int(len(ij)), 'mean_offdiag_sim': round(float(offdiag), 3),
                    'tokens': toptoks})
        print(f'  cluster {j}: n={len(ij):4d}  mean sim to other clusters {offdiag:+.2f}  {toptoks}')
    json.dump({'clusters': det, 'centroid_sim': Kc.round(3).tolist()},
              open(PT + 'clustering_viz_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
