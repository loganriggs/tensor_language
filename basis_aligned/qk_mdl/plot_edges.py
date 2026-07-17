"""Harvest of edge_heatmap.json: four lower-triangle dCE heatmaps (zero/mean/
pca1/pca4), the weights-only importance map, its rank-correlation with the
causal maps, and summary stats for results/15."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
d = json.load(open(f'{QK}/edge_heatmap.json'))
NL = 18
SRC = ['emb'] + [f'{t}{l}' for l in range(NL) for t in ('attn', 'mlp')]
DEST = [str(l) for l in range(1, NL)] + ['unembed']
METHODS = ['zero', 'mean', 'pca1', 'pca4']

mats = {me: np.full((len(SRC), len(DEST)), np.nan) for me in METHODS}
for key, v in d['edges'].items():
    edge, me = key.split('|')
    s, t = edge.split('->')
    mats[me][SRC.index(s), DEST.index(t)] = v

fig, axes = plt.subplots(1, 4, figsize=(22, 8.5), sharey=True)
norm = SymLogNorm(linthresh=0.01, vmin=-0.3, vmax=3.0)
for ax, me in zip(axes, METHODS):
    im = ax.imshow(mats[me], aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_title({'zero': 'zero-ablate', 'mean': 'global-mean', 'pca1': 'PCA-1',
                  'pca4': 'PCA-4'}[me])
    ax.set_xticks(range(len(DEST)))
    ax.set_xticklabels(DEST, fontsize=6.5, rotation=90)
    ax.set_xlabel('destination (layer reads)')
axes[0].set_yticks(range(len(SRC)))
axes[0].set_yticklabels(SRC, fontsize=6)
axes[0].set_ylabel('source stream')
fig.colorbar(im, ax=axes, fraction=0.015, label='ΔCE (symlog)')
fig.suptitle('Edge ablations: source stream ablated ONLY in the destination\'s reads (bilin18, T=512)', y=0.98)
fig.savefig(f'{QK}/results/fig_edge_heatmaps.png', dpi=130, bbox_inches='tight')
print('heatmaps saved')

# ---- weights-only importance ----
from tier2_model import load_elriggs
m, cfg = load_elriggs('bilin18', device='cpu')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
def read_mat(L):
    if L == 'unembed':
        return m.lm_head.weight.detach().float()
    blk = m.transformer.h[int(L)]
    a = blk.attn
    parts = [a.c_q.weight, a.c_k.weight, a.c_q2.weight, a.c_k2.weight, a.c_v.weight,
             blk.mlp.Left.weight, blk.mlp.Right.weight]
    return torch.cat([p.detach().float() for p in parts], 0)
def src_mat(nm):
    if nm == 'emb':
        return None
    l = int(nm[4:] if nm.startswith('attn') else nm[3:])
    blk = m.transformer.h[l]
    return (blk.attn.c_proj.weight if nm.startswith('attn') else blk.mlp.Down.weight).detach().float()

wmat = np.full((len(SRC), len(DEST)), np.nan)
for j, t in enumerate(DEST):
    R = read_mat(t)
    rn = R.norm()
    for i, s in enumerate(SRC):
        if np.isnan(mats['zero'][i, j]):
            continue
        Ws = src_mat(s)
        if Ws is None:
            continue
        wmat[i, j] = float((R @ Ws).norm() / (rn * Ws.norm()))
fig2, ax = plt.subplots(figsize=(6.5, 8.5))
im = ax.imshow(wmat, aspect='auto', cmap='Blues')
ax.set_xticks(range(len(DEST))); ax.set_xticklabels(DEST, fontsize=6.5, rotation=90)
ax.set_yticks(range(len(SRC))); ax.set_yticklabels(SRC, fontsize=6)
ax.set_title('weights-only: ||R_dest · W_src||_F (normalized)')
fig2.colorbar(im, fraction=0.03)
fig2.savefig(f'{QK}/results/fig_edge_weights.png', dpi=130, bbox_inches='tight')

# rank correlation weights vs |zero dCE|
from scipy.stats import spearmanr
mask = ~np.isnan(wmat) & ~np.isnan(mats['zero'])
rho, p = spearmanr(wmat[mask], np.abs(mats['zero'][mask]))
print(f'weights-vs-|zero| Spearman rho={rho:.3f} (n={mask.sum()}, p={p:.1e})')

# ---- summaries ----
z = mats['zero']
def top(mat, k=12, absval=False):
    v = np.abs(mat) if absval else mat
    idx = np.dstack(np.unravel_index(np.argsort(v, axis=None)[::-1], v.shape))[0]
    out = []
    for i, j in idx:
        if np.isnan(mat[i, j]): continue
        out.append((SRC[i], DEST[j], round(float(mat[i, j]), 4)))
        if len(out) >= k: break
    return out
print('\nTOP zero-ablation edges:')
for s, t, v in top(z): print(f'  {s} -> {t}: {v:+.4f}')
print('\nMOST NEGATIVE (helpful-to-ablate) edges, zero:')
neg = top(-np.where(np.isnan(z), -9, z), 10)
for s, t, v in neg: print(f'  {s} -> {t}: {-v:+.4f}')
# method ladder aggregate over big edges
big = (~np.isnan(z)) & (np.abs(z) > 0.02)
for me in METHODS:
    print(f'{me}: mean dCE over |zero|>0.02 edges = {np.nanmean(mats[me][big]):+.4f} '
          f'(n={big.sum()})')
# hub row
i5 = SRC.index('attn5')
row = [(DEST[j], round(float(z[i5, j]), 4)) for j in range(len(DEST)) if not np.isnan(z[i5, j])]
print('\nattn5 row (zero):', row)
json.dump({'spearman_weights_vs_zero': rho,
           'top_zero': top(z), 'most_negative': [(s, t, -v) for s, t, v in neg],
           'attn5_row': row},
          open(f'{QK}/edge_summary.json', 'w'), indent=1)
print('done')
