"""Stacked bars: how much of each component our named variables recover, built up token → +topic → +prev
(§896). 0 = mean-ablate, 1 = full. Shows which variable fills each layer's gap."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'layer_understanding_v3_results.json'))['components']
order = ['mlp0', 'attn0', 'attn5', 'mlp5', 'mlp8', 'mlp11', 'attn11', 'mlp16']
tok = [d[c]['token']['frac'] for c in order]
top = [max(d[c]['token+topic']['frac'] - d[c]['token']['frac'], 0) for c in order]
prev = [max(d[c]['token+topic+prev']['frac'] - d[c]['token+topic']['frac'], 0) for c in order]

fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xs = range(len(order))
c_tok, c_top, c_prev = '#104281', '#3987e5', '#cde2fb'
ax.bar(xs, tok, width=0.7, color=c_tok, edgecolor=SURFACE, linewidth=1.2, zorder=3, label='token (grammar / context-free)')
ax.bar(xs, top, width=0.7, bottom=tok, color=c_top, edgecolor=SURFACE, linewidth=1.2, zorder=3, label='+ topic (continuous gist)')
ax.bar(xs, prev, width=0.7, bottom=[tok[i]+top[i] for i in xs], color=c_prev, edgecolor=SURFACE, linewidth=1.2, zorder=3, label='+ prev-token (local / bigram)')
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.axhline(0.0, color=SECONDARY, lw=1.0, zorder=2)
ax.text(len(order)-0.5, 1.02, 'full model', ha='right', va='bottom', color=INK, fontsize=9)
for i in xs:
    tot = tok[i]+top[i]+prev[i]
    ax.text(i, tot+0.02, f'{tot:.2f}', ha='center', va='bottom', color=INK, fontsize=9)
ax.set_xticks(list(xs)); ax.set_xticklabels(order, rotation=30, ha='right', fontsize=10, color=INK)
ax.set_ylabel('fraction of the component recovered by named-variable tables', fontsize=10, color=INK)
ax.set_title('Filling the understanding gap with our named variables\n(0 = mean-ablate · 1 = full model · stacked: token → +topic → +prev-token)',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(0, 1.18)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.legend(loc='upper center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
out = PT + 'gapfill.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
