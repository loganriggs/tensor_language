"""Bar chart of the per-layer understanding metric (§893). 0 = mean-ablate (know nothing),
1 = full model. Bar = token-table understanding fraction; tick = shuffled-token null (genuine
understanding = bar above tick). Colored by band: front(grammar)/middle(content)/readout."""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'layer_understanding_results.json'))
comps = d['components']
order = ['mlp0', 'attn0', 'mlp1', 'mlp2', 'attn5', 'mlp5', 'attn8', 'mlp8', 'attn11', 'mlp11', 'mlp15', 'mlp16', 'mlp17']
band = {'mlp0': 'front', 'attn0': 'front', 'mlp1': 'front', 'mlp2': 'front',
        'attn5': 'middle', 'mlp5': 'middle', 'attn8': 'middle', 'mlp8': 'middle', 'attn11': 'middle', 'mlp11': 'middle',
        'mlp15': 'readout', 'mlp16': 'readout', 'mlp17': 'readout'}
col = {'front': '#104281', 'middle': '#898781', 'readout': '#3987e5'}
# noisy = tiny ablation-cost denominator (unreliable fraction)
noisy = {c for c in order if (comps[c]['ce_meanablate'] - d['ce_full']) < 0.05}

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=140)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xs = range(len(order))
vals = [comps[c]['understanding_frac'] for c in order]
nulls = [comps[c]['shuffled_null_frac'] for c in order]
bars = ax.bar(xs, vals, width=0.72, color=[col[band[c]] for c in order],
              edgecolor=SURFACE, linewidth=1.5, zorder=3)
for c in order:
    if c in noisy:
        i = order.index(c); bars[i].set_alpha(0.35); bars[i].set_hatch('//')
# shuffled-null tick per bar
for i, c in enumerate(order):
    ax.plot([i-0.36, i+0.36], [nulls[i], nulls[i]], color=INK, lw=1.4, zorder=4)
ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.axhline(0.0, color=SECONDARY, lw=1.0, zorder=2)
ax.text(len(order)-0.5, 1.02, 'full model (understand everything)', ha='right', va='bottom', color=INK, fontsize=9)
ax.text(len(order)-0.5, 0.02, 'mean-ablate (understand nothing)', ha='right', va='bottom', color=SECONDARY, fontsize=9)
ax.set_xticks(list(xs)); ax.set_xticklabels(order, rotation=35, ha='right', fontsize=10, color=INK)
ax.set_ylabel('fraction of the component captured by a token→output table', fontsize=10, color=INK)
ax.set_title('How much of each layer we understand as a token table\n(0 = mean-ablate · 1 = full model · black tick = shuffled-token null)',
             fontsize=12.5, color=INK, pad=12)
ax.set_ylim(-0.15, 1.25)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
# band labels
import matplotlib.patches as mp
handles = [mp.Patch(color=col[b], label=lab) for b, lab in
           [('front', 'front — grammar'), ('middle', 'middle — content'), ('readout', 'readout')]]
handles.append(mp.Patch(facecolor=MUTED, alpha=0.35, hatch='//', label='near-free to ablate (noisy)'))
ax.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.22))
plt.tight_layout()
out = PT + 'layer_understanding.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE)
print('wrote', out)
