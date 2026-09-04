"""Correction figure: whole-model named-variable understanding, in-sample vs held-out (§900 vs §901).
Shows the overfitting gap and that prev-token adds nothing out-of-sample."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
sd = json.load(open(PT + 'whole_model_understanding_results.json'))['levels']
fr = json.load(open(PT + 'whole_model_understanding_fresh_results.json'))['levels']
levels = ['token', 'token+topic', 'token+topic+prev']
labels = ['token\n(grammar)', '+ continuous\ntopic', '+ prev-token\n(local)']
insample = [sd[l]['understanding_frac'] for l in levels]
held = [fr[l]['understanding_frac'] for l in levels]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(levels)); w = 0.36
ax.bar(x - w/2, insample, w, color='#898781', edgecolor=SURFACE, linewidth=1.2, zorder=3, label='in-sample (fit & scored on same rows)')
ax.bar(x + w/2, held, w, color='#104281', edgecolor=SURFACE, linewidth=1.2, zorder=3, label='held-out (honest)')
for xi, v in zip(x - w/2, insample): ax.text(xi, v+0.015, f'{v:.2f}', ha='center', va='bottom', color=SECONDARY, fontsize=9.5)
for xi, v in zip(x + w/2, held): ax.text(xi, v+0.015, f'{v:.2f}', ha='center', va='bottom', color=INK, fontsize=9.5)
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2); ax.text(len(levels)-0.5, 1.02, 'full model', ha='right', va='bottom', color=INK, fontsize=9)
ax.axhline(0.0, color=SECONDARY, lw=1.0, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, color=INK)
ax.set_ylabel('fraction of the WHOLE model recovered (all 36 components at once)', fontsize=10, color=INK)
ax.set_title('How much of the model our named variables really explain\nin-sample looks like 0.81 — held-out it is 0.29 (and prev-token adds nothing)',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(-0.05, 1.12)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.legend(loc='upper left', frameon=False, fontsize=9.5)
plt.tight_layout(); out = PT + 'fresh_vs_insample.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
