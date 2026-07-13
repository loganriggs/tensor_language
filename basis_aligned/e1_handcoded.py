"""e1: hand-coded block-sparse bilinear net; insert a rotation -> function and
folded weights unchanged, unfolded weights fully dense.

Task: 8 inputs in 4 pairs, y_c = x_{2c} * x_{2c+1}, one block active at a time.
Hand-coded solution: E=I, hidden unit k reads the pair (L[k]=e_{2k}, R[k]=e_{2k+1}),
D routes hidden k to model dim k, U reads class c off model dim c.
"""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from palette import DIVERGING, INK, SECONDARY
from common import forward, fold, interaction, hoyer, near_zero_frac, \
    block_score, block_data, random_orthogonal

torch.manual_seed(0)
DEV = 'cpu'
N_BLOCKS, D_IN, D_MODEL, D_H = 4, 8, 8, 4
PAIRS = [(2 * c, 2 * c + 1) for c in range(N_BLOCKS)]


def handcoded():
    p = {'E': torch.eye(D_MODEL, D_IN),
         'L': torch.zeros(D_H, D_MODEL), 'R': torch.zeros(D_H, D_MODEL),
         'D': torch.zeros(D_MODEL, D_H), 'U': torch.zeros(N_BLOCKS, D_MODEL)}
    for k in range(N_BLOCKS):
        p['L'][k, 2 * k] = 1.0
        p['R'][k, 2 * k + 1] = 1.0
        p['D'][k, k] = 1.0
        p['U'][k, k] = 1.0
    return p


def rotated(p, seed=1):
    """Insert Q at the embedding interface, Q2 at the unembedding interface."""
    Q = random_orthogonal(D_MODEL, DEV, seed=seed)
    Q2 = random_orthogonal(D_MODEL, DEV, seed=seed + 100)
    return {'E': Q @ p['E'], 'L': p['L'] @ Q.T, 'R': p['R'] @ Q.T,
            'D': Q2 @ p['D'], 'U': p['U'] @ Q2.T}


p_id = handcoded()
p_rot = rotated(p_id)

# --- checks: same function, same folded weights, same interaction form
x, y = block_data(4096, DEV)
err_task_id = (forward(p_id, x) - y).abs().max().item()
err_task_rot = (forward(p_rot, x) - y).abs().max().item()
xr = torch.randn(4096, D_IN)  # also off-distribution: full-support inputs
err_fn = (forward(p_id, xr) - forward(p_rot, xr)).abs().max().item()
f_id, f_rot = fold(p_id), fold(p_rot)
err_fold = max((f_id[k] - f_rot[k]).abs().max().item() for k in f_id)
B_id, B_rot = interaction(p_id), interaction(p_rot)

results = {
    'max_err_task_identity': err_task_id,
    'max_err_task_rotated': err_task_rot,
    'max_fn_diff_full_support': err_fn,
    'max_folded_weight_diff': err_fold,
    'block_score_identity': block_score(B_id, PAIRS),
    'block_score_rotated': block_score(B_rot, PAIRS),
    'sparsity': {}
}
for name, p in [('identity', p_id), ('rotated', p_rot)]:
    rep = {}
    for k, w in {**p, **fold(p)}.items():
        rep[k] = {'hoyer': round(hoyer(w), 4), 'zero_frac': round(near_zero_frac(w), 4)}
    results['sparsity'][name] = rep

print(json.dumps(results, indent=2))
with open('/workspace/tensor_language/basis_aligned/e1_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)

# --- figure: unfolded vs folded heatmaps, identity vs rotated
mats = ['E', 'L', 'R', 'D', 'U', 'Lf', 'Rf', 'Df']
fig, axes = plt.subplots(2, 8, figsize=(15, 4.6))
for row, (name, p) in enumerate([('hand-coded', p_id), ('+ inserted rotation', p_rot)]):
    all_w = {**p, **fold(p)}
    for col, k in enumerate(mats):
        ax = axes[row, col]
        w = all_w[k].numpy()
        v = max(abs(w).max(), 1e-9)
        ax.imshow(w, cmap=DIVERGING, vmin=-v, vmax=v, aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#e1e0d9')
        if row == 0:
            ax.set_title({'Lf': 'L̃ = LE', 'Rf': 'R̃ = RE', 'Df': 'D̃ = UD'}.get(k, k),
                         fontsize=11, color=INK)
        ax.set_xlabel(f'zeros {near_zero_frac(all_w[k]):.0%}', fontsize=8,
                      color=SECONDARY, labelpad=2)
    axes[row, 0].set_ylabel(name, fontsize=11, color=INK)
# separator between unfolded (basis-dependent) and folded (basis-independent) groups
for row in range(2):
    axes[row, 4].annotate('', xy=(1.12, 0.5), xycoords='axes fraction')
fig.suptitle('Same function, two parameterizations: unfolded weights (E,L,R,D,U) are '
             'basis-dependent; folded weights (L̃,R̃,D̃) are not', fontsize=12, color=INK)
fig.text(0.638, 0.02, '◀ unfolded (rotation destroys sparsity)   |   folded (sparsity survives) ▶',
         ha='center', fontsize=9, color=SECONDARY)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
line_x = (axes[0, 4].get_position().x1 + axes[0, 5].get_position().x0) / 2
fig.add_artist(plt.Line2D([line_x, line_x], [0.06, 0.90], color='#898781',
                          lw=1, ls=':', transform=fig.transFigure))
fig.savefig('/workspace/tensor_language/basis_aligned/figures/e1_handcoded.png', dpi=160)
print('figure saved')
