"""e6 figures: budget curves (FVU, dCE) per representation class + dCE-vs-FVU."""

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID

BASE = '/workspace/tensor_language/basis_aligned'
r = json.load(open(f'{BASE}/e6_results.json'))
rows = r['rows']

C = {'svd': '#3987e5', 'kmeans': '#e34948', 'rq': '#2f9e63', 'tree': '#9a6ae1'}
LABEL = {'svd': 'SVD (rank prior)', 'kmeans': 'k-means (fewer objects)',
         'rq': 'residual VQ (sum of h objects)', 'tree': 'tree code (hierarchy prior)'}
CONTROLS = {'svd_random': 'random basis', 'kmeans_shuffled': 'shuffled assign'}

by = defaultdict(list)
for row in rows:
    by[row['method']].append(row)
for k in by:
    by[k].sort(key=lambda x: x['budget'])


def style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

for ax, key, ylab in [(axes[0], 'fvu', 'FVU (weight audit)'),
                      (axes[1], 'dce', 'ΔCE nats (behavior audit)')]:
    for m in ['svd', 'kmeans', 'rq', 'tree']:
        rs = by[m]
        ax.plot([x['budget'] for x in rs], [x[key] for x in rs], 'o-',
                color=C[m], lw=2, ms=6, label=LABEL[m])
    for m, lab in CONTROLS.items():
        for x in by.get(m, []):
            ax.plot(x['budget'], x[key], 'x', color=INK, ms=8, mew=2)
            ax.annotate(lab, (x['budget'], x[key]), textcoords='offset points',
                        xytext=(6, 4), fontsize=7.5, color=SECONDARY)
    ax.set_xlabel('parameter budget (fraction of V·d floats)', color=INK)
    ax.set_ylabel(ylab, color=INK)
    style(ax)
axes[0].legend(frameon=False, fontsize=8.5, loc='lower left')
axes[0].set_title('weights barely compress', fontsize=11, color=INK)
axes[1].set_title(f"behavior collapses (baseline CE {r['baseline_ce']:.2f})",
                  fontsize=11, color=INK)

ax = axes[2]
for m in ['svd', 'kmeans', 'rq', 'tree']:
    rs = by[m]
    ax.plot([x['fvu'] for x in rs], [x['dce'] for x in rs], 'o-', color=C[m],
            lw=1.5, ms=6, label=LABEL[m])
ns = by['noise']
ax.plot([x['fvu'] for x in ns], [x['dce'] for x in ns], 'x--', color=INK,
        lw=1.5, ms=8, mew=2, label='additive gaussian noise')
ax.annotate('noise at FVU 0.75:\n+0.43 nats', (0.75, 0.43), textcoords='offset points',
            xytext=(8, -18), fontsize=8, color=SECONDARY)
ax.annotate('deletion at FVU 0.75:\n+4.3 nats', (0.75, 4.33), textcoords='offset points',
            xytext=(-70, 8), fontsize=8, color=SECONDARY)
for m, lab in CONTROLS.items():
    for x in by.get(m, []):
        ax.plot(x['fvu'], x['dce'], '+', color=INK, ms=9, mew=2)
        ax.annotate(lab, (x['fvu'], x['dce']), textcoords='offset points',
                    xytext=(-6, 5), fontsize=7.5, color=SECONDARY, ha='right')
ax.set_xlabel('FVU', color=INK)
ax.set_ylabel('ΔCE (nats)', color=INK)
ax.legend(frameon=False, fontsize=8)
ax.set_title('FVU does not predict damage:\nsubtraction ≫ addition at equal FVU',
             fontsize=10.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e6_embedding.png', dpi=160)
print('saved')
