"""Figure: how much of each early bilin18 component do we understand (corrected,
mean-preserving metric from §808 bilin18_scoreboard_mp). Bar height = the component's
loss-benefit (how much ablating it costs = how much it matters); each bar split into the
class+position share (understood) and the remainder. Answers the user's question 'how
much of the early layers do we 100% understand?' with the corrected numbers. Local plot,
no GPU."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
FIG = PT + 'early_understood_corrected.png'
BLUE, GREEN, REM = '#3987e5', '#4a9e6f', '#c9c7bf'   # class, position-ish accent, remainder (muted)

d = json.load(open(PT + 'bilin18_scoreboard_mp_results.json'))
pc = d['per_component']
NL = 6
names = [f'{w}{L}' for L in range(NL) for w in ('attn', 'mlp')]
ben = np.array([pc[n]['benefit'] for n in names])
keep = np.array([max(min(pc[n]['keep_mean'], 1), 0) for n in names])
understood = ben * keep
remainder = ben * (1 - keep)

fig, ax = plt.subplots(figsize=(12, 5.2)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xs = np.arange(len(names))
ax.bar(xs, understood, color=BLUE, label='class + position (understood)', width=0.72, zorder=3)
ax.bar(xs, remainder, bottom=understood, color=REM, label='remainder (diffuse content)', width=0.72, zorder=3)

# per-bar understood % label
for i, n in enumerate(names):
    if ben[i] > 0.05:
        ax.text(i, ben[i] + 0.03, f'{keep[i]*100:.0f}%', ha='center', va='bottom',
                fontsize=8.5, color=SECONDARY)

ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9, color=INK)
ax.set_ylabel('loss-benefit  (nats; how much the component matters)', fontsize=10.5, color=INK)
ax.set_title('bilin18 early layers: how much of each component is class + position',
             fontsize=13, color=INK, pad=12)
early_frac = understood[ben > 0].sum() / ben[ben > 0].sum()
ax.text(0.985, 0.93, f'layers 0–5 overall: {early_frac*100:.0f}% class+position\nwhole model: {d["nw_meanpreserve"]*100:.0f}%',
        transform=ax.transAxes, ha='right', va='top', fontsize=10.5, color=INK,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=GRID))
ax.grid(axis='y', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY)
ax.legend(frameon=False, fontsize=10, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.09))
fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote', FIG)
print(f'early (0-5) class+position: {early_frac:.3f}; whole model: {d["nw_meanpreserve"]}')
print('remainder concentrated in MLPs: ' + ', '.join(f'{n} {(1-keep[i])*100:.0f}%' for i, n in enumerate(names) if (1-keep[i]) > 0.15 and ben[i] > 0.1))
