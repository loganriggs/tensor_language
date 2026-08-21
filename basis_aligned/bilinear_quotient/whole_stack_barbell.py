"""Figure: bilin18 whole-stack barbell (§812). Per-layer loss-benefit (how much each layer
matters) as bars, with the class+position understood-share as a line, across all 18 layers.
Shows the barbell: heavy class+position computation at the front, a smaller class+position
read-out at the back, and a nearly-inert middle. From existing §808 data; local plot, no GPU."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
FIG = PT + 'whole_stack_barbell.png'
BLUE, ACCENT = '#3987e5', '#b5852a'

d = json.load(open(PT + 'bilin18_scoreboard_mp_results.json'))
pc = d['per_component']
L = np.arange(18)
# per-layer: sum attn+mlp benefit; benefit-weighted class+position keep
ben = np.array([pc[f'attn{i}']['benefit'] + pc[f'mlp{i}']['benefit'] for i in L])
kw = []
for i in L:
    tb = tk = 0.0
    for w in ('attn', 'mlp'):
        c = pc[f'{w}{i}']
        if c['benefit'] > 0:
            tb += c['benefit']; tk += c['benefit'] * max(min(c['keep_mean'], 1), 0)
    kw.append(tk / tb if tb > 0 else np.nan)
kw = np.array(kw)

fig, ax = plt.subplots(figsize=(12, 5.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.bar(L, ben, color=BLUE, width=0.7, zorder=3, label='loss-benefit (how much the layer matters)')
ax.set_ylabel('loss-benefit  (nats)', fontsize=10.5, color=INK)
ax.set_xlabel('layer', fontsize=10.5, color=INK)
ax.set_xticks(L); ax.set_xticklabels(L, fontsize=9, color=INK)
ax.set_title('bilin18 is a barbell: front computes, back reads, middle is quiet', fontsize=13, color=INK, pad=12)

ax2 = ax.twinx()
ax2.plot(L, kw*100, color=ACCENT, lw=2, marker='o', ms=5, zorder=4, label='class+position share (right axis)')
ax2.set_ylabel('class + position share  (%)', fontsize=10.5, color=ACCENT)
ax2.set_ylim(0, 105); ax2.tick_params(colors=ACCENT)
ax2.spines['right'].set_color(ACCENT)

# band annotations
for x0, x1, txt in [(-0.5, 5.5, 'FRONT: compute\nclass+position\n(81% of all benefit)'),
                    (5.5, 11.5, 'MIDDLE:\nnearly inert'),
                    (11.5, 17.5, 'BACK: read out\nclass+position')]:
    ax.axvspan(x0, x1, color=MUTED, alpha=0.05, zorder=0)
    ax.text((x0+x1)/2, ax.get_ylim()[1]*0.82, txt, ha='center', va='top', fontsize=9, color=SECONDARY)

ax.grid(axis='y', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top',): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY)
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, frameon=False, fontsize=9.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.10))
fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote', FIG)
print(f'front benefit {ben[:6].sum():.2f} ({ben[:6].sum()/ben.sum()*100:.0f}%) | mid {ben[6:12].sum():.2f} | back {ben[12:].sum():.2f}')
