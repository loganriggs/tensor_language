"""Understanding-benchmark PROGRESSION: earlier (same-data, inflated) vs now (held-out, certified), and how
the method improved the honest number."""
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
labels = ['smooth-map\n(§900,\nin-sample)', 'lookup table\n(§901,\nheld-out)', 'token linear-map\n(§939,\nheld-out)']
vals   = [0.81, 0.315, 0.422]
kind   = ['same', 'held', 'held']   # same-data vs held-out
col    = {'same': '#b0aea7', 'held': '#104281'}
cols   = [col[k] for k in kind]

fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(labels))
ax.bar(x, vals, width=0.6, color=cols, edgecolor=SURFACE, linewidth=1.3, zorder=3)
for xi, v in zip(x, vals): ax.text(xi, v+0.02, f'{v:.2f}', ha='center', va='bottom', color=INK, fontsize=11)
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2); ax.text(len(labels)-0.5, 1.02, 'full model', ha='right', va='bottom', color=INK, fontsize=9)
# divider between same-data and held-out
ax.axvline(0.5, color=MUTED, lw=1.0, ls=':', zorder=1)
ax.text(0.0, 1.06, 'same-data\n(inflated)', ha='center', color=SECONDARY, fontsize=9.5)
ax.text(1.5, 1.08, 'held-out (certified)', ha='center', color='#104281', fontsize=9.5, fontweight='bold')
# arrow showing method improvement within held-out
ax.annotate('', xy=(2, 0.422), xytext=(1, 0.315), arrowprops=dict(arrowstyle='->', color='#8c2b2b', lw=2))
ax.text(1.5, 0.30, 'same variables,\nsmooth per-token map: +0.11', ha='center', color='#8c2b2b', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, color=INK)
ax.set_ylabel('fraction of the model understood (0 = mean-ablate, 1 = full)', fontsize=10, color=INK)
ax.set_title('How the understanding benchmark changed\nearlier ~0.8 was in-sample (and rank-confounded, §836); certified held-out is ~0.4',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(0, 1.14)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
plt.tight_layout(); out = PT + 'benchmark_progression.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
