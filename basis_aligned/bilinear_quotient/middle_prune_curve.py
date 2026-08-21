"""Figure: prune curves (§816). CE cost vs number of layers dropped (cheapest-first), for the
middle band (concave/redundant — half compressible) vs the front band (convex/costly). Local
plot from existing middle_prune data, no GPU."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
FIG = PT + 'middle_prune_curve.png'
BLUE, RED = '#3987e5', '#c0562b'

d = json.load(open(PT + 'middle_prune_results.json'))
k = np.arange(7)
mid = np.array(d['middle']['cost_vs_k']); front = np.array(d['front']['cost_vs_k'])

fig, ax = plt.subplots(figsize=(9.5, 5.6)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(k, front, color=RED, lw=2.2, marker='s', ms=6, zorder=4, label='FRONT (layers 0–5): convex — each layer irreplaceable')
ax.plot(k, mid, color=BLUE, lw=2.2, marker='o', ms=6, zorder=4, label='MIDDLE (layers 6–11): concave — half is spare')
ax.axhline(0, color=MUTED, lw=0.8)

# annotate the "half the middle is free" point
ax.annotate('drop 3 of 6 middle\nlayers for 0.27 nats', xy=(3, mid[3]), xytext=(3.3, 1.5),
            fontsize=9.5, color=BLUE, arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

ax.set_xlabel('number of layers deleted (cheapest first)', fontsize=10.5, color=INK)
ax.set_ylabel('cross-entropy cost  (nats)', fontsize=10.5, color=INK)
ax.set_title('The maintenance middle is half redundant; the front is not', fontsize=13, color=INK, pad=12)
ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=SECONDARY)
ax.legend(frameon=False, fontsize=10, loc='upper left')
fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote', FIG)
