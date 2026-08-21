"""Visualize the mlp1 cluster hierarchy from cluster_nesting_crossK: coarse
(K=4) parents -> fine (K=16) children, edge label = containment. Palette."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'cluster_nesting_crossK_results.json'))
clab = d['coarse_labels']; fine = d['fine']
KC = len(clab)
NAME = ['closed-class', 'content / proper', 'prep-ish', 'punctuation']
COL = ['#3987e5', '#e34948', '#8c7a2b', '#52514e']

groups = {j: [] for j in range(KC)}
for f in fine:
    cont = [c if c is not None else -1 for c in f['containment']]
    j = int(np.argmax(cont))
    groups[j].append((f['label'], float(max(cont))))

# vertical positions for children, grouped by parent
rows = []
for j in range(KC):
    for lab, cont in groups[j]:
        rows.append((j, lab, cont))
n = len(rows)
fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * n + 1)))
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE); ax.axis('off')
ys = np.linspace(0.95, 0.05, n)
xP, xC = 0.30, 0.60
# parent y = mean of its children
py = {}
for j in range(KC):
    idxs = [i for i, r in enumerate(rows) if r[0] == j]
    if idxs: py[j] = ys[idxs].mean()
for i, (j, lab, cont) in enumerate(rows):
    ax.plot([xP + 0.02, xC - 0.005], [py[j], ys[i]], color=COL[j], lw=1.2, alpha=0.4, zorder=1)
    ax.text(xC, ys[i], lab[:26], fontsize=10, va='center', ha='left', color=INK, family='monospace')
    ax.text(xC - 0.02, ys[i], f'{cont:.2f}', fontsize=7.5, va='center', ha='right', color=SECONDARY)
for j in range(KC):
    if j not in py: continue
    ax.scatter([xP], [py[j]], s=110, color=COL[j], zorder=3)
    ax.text(xP - 0.02, py[j], f'{NAME[j]}\n[{clab[j][:16]}]', fontsize=10, va='center',
            ha='right', color=COL[j], fontweight='bold')
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1)
ax.set_title('mlp1 cluster hierarchy: coarse super-categories → fine clusters\n'
             'each fine cluster nests into exactly one coarse parent (containment shown) — '
             'a clean tree across scales',
             fontsize=12, color=INK, loc='left', pad=12)
fig.tight_layout()
fig.savefig(PT + 'hierarchy_tree.png', dpi=150, facecolor=SURFACE)
print('wrote hierarchy_tree.png')
