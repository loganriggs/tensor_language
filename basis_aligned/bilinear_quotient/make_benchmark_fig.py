"""Updated understanding benchmark (HELD-OUT). Per-component: best generalizing stand-in (rank-512 linear map
of named variables + content, §910). Whole-model aggregate annotated. 0 = mean-ablate, 1 = full model."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'learned_map_rank512_results.json'))['components']
order = ['mlp0', 'attn5', 'mlp5', 'mlp8', 'mlp11', 'mlp16']
labels = ['mlp0\n(front)', 'attn5\n(mid)', 'mlp5\n(mid)', 'mlp8\n(mid)', 'mlp11\n(mid)', 'mlp16\n(readout)']
vals = [d[c]['upstream_map_fresh'] for c in order]
band = ['#104281', '#898781', '#898781', '#898781', '#898781', '#3987e5']

fig, ax = plt.subplots(figsize=(10, 5.4), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(order))
ax.bar(x, vals, width=0.62, color=band, edgecolor=SURFACE, linewidth=1.2, zorder=3)
for xi, v in zip(x, vals): ax.text(xi, v+0.02, f'{v:.2f}', ha='center', va='bottom', color=INK, fontsize=10)
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2); ax.text(len(order)-0.5, 1.02, 'full model', ha='right', va='bottom', color=INK, fontsize=9)
ax.axhline(0.0, color=SECONDARY, lw=1.0, zorder=2)
# whole-model benchmark line
wm = 0.406
ax.axhline(wm, color='#8c2b2b', lw=1.6, ls=(0, (6, 3)), zorder=4)
ax.text(len(order)-0.5, wm+0.02, f'WHOLE MODEL benchmark = {wm:.2f}', ha='right', va='bottom', color='#8c2b2b', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=INK)
ax.set_ylabel('fraction understood, held-out (0 = mean-ablate, 1 = full)', fontsize=10, color=INK)
ax.set_title('Understanding benchmark — updated & held-out\nbest generalizing stand-in (named variables + content); grammar solved, content the frontier',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(0, 1.12)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color='#104281', label='front (grammar)'), mp.Patch(color='#898781', label='middle (content)'),
                   mp.Patch(color='#3987e5', label='readout')], loc='upper center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.16))
plt.tight_layout(); out = PT + 'benchmark.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
