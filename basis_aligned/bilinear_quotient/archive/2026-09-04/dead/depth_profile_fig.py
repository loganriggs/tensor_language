"""Depth profile of the 18 MLP layers: benefit (bars) + functional rank r80
(line), from rspd_depth_rank_map_results.json. The barbell. Matches palette."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'rspd_depth_rank_map_results.json'))
NL = 18
ben = [d['profile'][str(i)]['benefit'] for i in range(NL)]
r80 = [d['profile'][str(i)]['r80'] for i in range(NL)]
# mark deep-middle (tiny benefit) where r80 is not meaningful
inert = [b < 0.06 for b in ben]

fig, ax = plt.subplots(figsize=(10.5, 5.2)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xs = np.arange(NL)
BLUE, RED = '#3987e5', '#e34948'
cols = [MUTED if inert[i] else BLUE for i in range(NL)]
ax.bar(xs, ben, color=cols, width=0.68, zorder=3)
ax.set_ylabel('benefit: CE nats lost if MLP ablated', fontsize=11, color=BLUE)
ax.tick_params(axis='y', colors=BLUE, labelsize=9)
ax.set_xticks(xs); ax.set_xticklabels([f'mlp{i}' for i in range(NL)], fontsize=8, rotation=45)
ax.set_ylim(0, 2.55)

ax2 = ax.twinx()
r80_plot = [r if not inert[i] else np.nan for i, r in enumerate(r80)]
ax2.plot(xs, r80_plot, 'o-', color=RED, lw=1.8, ms=6, zorder=5)
ax2.set_yscale('log', base=2); ax2.set_ylim(0.8, 640)
ax2.set_yticks([1, 4, 16, 64, 256]); ax2.set_yticklabels(['1', '4', '16', '64', '256'], fontsize=9)
ax2.set_ylabel('functional rank r80 (log)', fontsize=11, color=RED)
ax2.tick_params(axis='y', colors=RED, labelsize=9)

# annotations
ax.annotate('low-rank,\nhigh-benefit\nEDGES', xy=(0, ben[0]), xytext=(2.1, 2.15),
            fontsize=9, color=INK, ha='center',
            arrowprops=dict(arrowstyle='->', color=SECONDARY, lw=1))
ax.annotate('', xy=(16, ben[16]), xytext=(14.4, 2.0),
            arrowprops=dict(arrowstyle='->', color=SECONDARY, lw=1))
ax.text(10, 0.35, 'deep-middle mlp6–14: nearly inert\n(~0.03 nats; r80 not meaningful)',
        fontsize=8.5, color=MUTED, ha='center', style='italic')
ax.text(2.4, 1.15, 'early-middle:\nhigh-rank', fontsize=8.5, color=RED, ha='left')

ax.set_title('bilin18 MLP work across depth — a barbell\n'
             'bars = contribution (grey = near-inert); red line = functional rank of the decoder',
             fontsize=12.5, color=INK, loc='left', pad=12)
ax.grid(True, axis='y', color=GRID, lw=0.6, zorder=0)
for s in ['top']: ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
for s in ['left', 'bottom', 'right']:
    ax.spines[s].set_color(SECONDARY)
fig.tight_layout()
out = PT + 'depth_profile.png'
fig.savefig(out, dpi=150, facecolor=SURFACE); print('wrote', out)
