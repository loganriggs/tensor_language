"""e7 figure: Pareto frontier over (n_atoms, L0) with CE-finetuned points."""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID

BASE = '/workspace/tensor_language/basis_aligned'
r7 = json.load(open(f'{BASE}/e7_results.json'))
r6 = json.load(open(f'{BASE}/e6_results.json'))

# validated palette: k=1 blue, k=4 red, k=16 green, k=64 purple
KC = {1: '#3987e5', 4: '#e34948', 16: '#2f9e63', 64: '#9a6ae1'}

fig, ax = plt.subplots(figsize=(8.5, 5.6))

# stage A grid: one curve per k over n_atoms
grid = [x for x in r7['rows'] if x['method'] == 'topk_dict']
for k in sorted({x['k'] for x in grid}):
    rs = sorted([x for x in grid if x['k'] == k], key=lambda x: x['n_atoms'])
    ax.plot([x['n_atoms'] for x in rs], [x['dce'] for x in rs], 'o-',
            color=KC[k], lw=2, ms=6, label=f'top-k dict, L0={k}')

# CE-finetuned points (stars) + arrows from their stage-A ancestors
for x in [x for x in r7['rows'] if x['method'] == 'topk_dict_ceft']:
    anc = [a for a in grid if a['n_atoms'] == x['n_atoms'] and a['k'] == x['k']][0]
    ax.annotate('', xy=(x['n_atoms'], x['dce_after']),
                xytext=(anc['n_atoms'], anc['dce']),
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    ax.plot(x['n_atoms'], x['dce_after'], '*', color=KC[x['k']], ms=16,
            mec='#fcfcfb', mew=1, zorder=6)
km = [x for x in r7['rows'] if x['method'] == 'kmeans_ceft']
if km:
    x = km[0]
    ax.plot(x['n_atoms'], x['dce_before'], 'o', color=INK, ms=7, mfc='none', mew=1.6)
    ax.plot(x['n_atoms'], x['dce_after'], '*', color=INK, ms=16, mec='#fcfcfb',
            mew=1, zorder=6)
    ax.annotate('', xy=(x['n_atoms'], x['dce_after']),
                xytext=(x['n_atoms'], x['dce_before']),
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    ax.annotate('k-means corner\n(unlearned → CE-tuned)',
                (x['n_atoms'], x['dce_before']), textcoords='offset points',
                xytext=(8, 2), fontsize=8, color=SECONDARY)

# e6 corners for context
for row in r6['rows']:
    if row['method'] == 'svd' and row['label'] in ('r=100', 'r=512'):
        n = int(row['label'][2:])
        ax.plot(n, row['dce'], 's', color=INK, ms=7, mfc='none', mew=1.6)
        ax.annotate(f'SVD {row["label"]} (dense codes)', (n, row['dce']),
                    textcoords='offset points', xytext=(8, -2), fontsize=8,
                    color=SECONDARY)
ax.plot(50304, 0, 'D', color=INK, ms=7, mfc='none', mew=1.6)
ax.annotate('original E\n(n=V, L0=1)', (50304, 0), textcoords='offset points',
            xytext=(2, -26), fontsize=8, color=SECONDARY, ha='center')

ax.axhline(0, color=GRID, lw=1)
ax.set_xscale('log', base=2)
ax.set_xlabel('number of dictionary atoms n (log2)', color=INK)
ax.set_ylabel('ΔCE (nats) on pile-10k', color=INK)
ax.set_title('Pareto frontier: objects (n) × per-token sparsity (L0) → behavior\n'
             'stars = CE-finetuned through the frozen model', fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc='upper right')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY, labelsize=9)
ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e7_pareto.png', dpi=160)
print('saved')
