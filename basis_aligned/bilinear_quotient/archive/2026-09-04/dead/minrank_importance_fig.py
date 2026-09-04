"""Component x cluster CE-importance heatmap from cluster_global_minrank:
which global A-SVD components each cluster relies on. Shows the shared
(most clusters need most components) vs the few specialized cases."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE, BLUES

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'cluster_global_minrank_results.json'))
imp = np.array(d['importance'])              # (M, K) CE increase from ablating component k for cluster c
M, K = imp.shape
mr = d['minrank']
# normalize each cluster column to its max for visual comparability
impn = imp / (imp.max(0, keepdims=True) + 1e-9)
impn = np.clip(impn, 0, 1)

fig, ax = plt.subplots(figsize=(7.5, 9)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
im = ax.imshow(impn, cmap=BLUES, aspect='auto', vmin=0, vmax=1, origin='upper')
ax.set_xticks(range(K))
ax.set_xticklabels([f'c{c}\nr90={mr[str(c)]}' for c in range(K)], fontsize=8)
ax.set_ylabel('global A-SVD component (0 = largest singular value)', fontsize=10)
ax.set_xlabel('CE-ablation-covariance cluster', fontsize=10)
ax.set_title('Which shared components each mlp1 cluster relies on\n'
             '(CE-importance of ablating each global component, normalized per cluster)\n'
             'most clusters use most components — the vocabulary is shared, not split',
             fontsize=11, color=INK, loc='left', pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label('normalized CE importance', fontsize=9)
fig.tight_layout()
out = PT + 'minrank_importance.png'
fig.savefig(out, dpi=150, facecolor=SURFACE); print('wrote', out)
