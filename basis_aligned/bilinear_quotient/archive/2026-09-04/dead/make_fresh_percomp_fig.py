"""Corrected per-component understanding: same-data (§893) vs held-out (§902). Front generalizes; middle/
readout token-tables overfit (some go negative = worse than mean-ablation out-of-sample)."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'layer_understanding_fresh_results.json'))['components']
order = ['mlp0', 'attn0', 'mlp1', 'mlp2', 'attn5', 'mlp5', 'attn8', 'mlp8', 'attn11', 'mlp11', 'mlp15', 'mlp16', 'mlp17']
same = [d[c]['samedata_893'] for c in order]
fresh = [max(d[c]['fresh_understanding'], -1.1) for c in order]  # clamp for display

fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, same, w, color='#898781', edgecolor=SURFACE, linewidth=1.1, zorder=3, label='same-data (§893, in-sample)')
ax.bar(x + w/2, fresh, w, color=['#104281' if f >= 0 else '#8c2b2b' for f in fresh], edgecolor=SURFACE, linewidth=1.1, zorder=3, label='held-out (honest)')
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2); ax.text(len(order)-0.5, 1.02, 'full model', ha='right', va='bottom', color=INK, fontsize=9)
ax.axhline(0.0, color=INK, lw=1.2, zorder=2)
ax.text(0.1, -0.16, 'below 0 = the stand-in is worse than mean-ablation out-of-sample (not a token function)', color='#8c2b2b', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha='right', fontsize=10, color=INK)
ax.set_ylabel('understanding fraction (0 = mean-ablate, 1 = full)', fontsize=10, color=INK)
ax.set_title('Which layers we genuinely understand as token tables — in-sample vs held-out\nfront (grammar) generalizes; middle & readout token-tables were overfit',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(-1.15, 1.15)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.legend(loc='lower left', frameon=False, fontsize=9.5)
plt.tight_layout(); out = PT + 'fresh_percomp.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
