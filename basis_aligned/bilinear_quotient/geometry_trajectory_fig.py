"""Figure: whole-stack token-geometry trajectory (§857). Two panels: (top) effective dimension of the
token-mean representation per stage (embedding, after each of 18 layers) — the collapse/expand story;
(bottom) consecutive RSA (how much each layer re-clusters the relative similarity structure). From
layer_geometry_full_results.json; local plot, no GPU."""
import json, numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'layer_geometry_full_results.json'))
ed = d['effective_dim']; stages = ['emb'] + [f'L{i}' for i in range(18)]
edv = [ed[s] for s in stages]
rsa = d['consecutive_rsa']; rk = list(rsa.keys()); rv = [rsa[k] for k in rk]
BLUE, RED = '#3987e5', '#c0562b'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False); fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2): ax.set_facecolor(SURFACE)
x = np.arange(len(stages))
ax1.plot(x, edv, color=BLUE, lw=2, marker='o', ms=5, zorder=3)
ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=8, color=INK)
ax1.set_ylabel('effective dimension\n(token geometry)', fontsize=10.5, color=INK)
ax1.set_title('bilin18 token geometry across depth: two expansions, three collapses', fontsize=13, color=INK, pad=10)
for xi, lab in [(1,'L0 collapse\n→class'),(2,'L1\nre-expand'),(6,'L5\ncollapse'),(10,'L9 peak\n(middle re-inflates)'),(18,'L17\n→3-dim\nprediction')]:
    ax1.annotate(lab, xy=(xi, edv[xi]), xytext=(xi, edv[xi]+ (14 if edv[xi]<45 else -20)), fontsize=8, color=SECONDARY, ha='center')
ax1.grid(axis='y', color=GRID, lw=0.8); ax1.set_axisbelow(True)
for s in ('top','right'): ax1.spines[s].set_visible(False)
ax2.plot(np.arange(len(rv)), rv, color=RED, lw=2, marker='s', ms=5, zorder=3)
ax2.axhline(1.0, color=MUTED, lw=0.8, ls='--')
ax2.set_xticks(np.arange(len(rk))); ax2.set_xticklabels([k.replace('->','→') for k in rk], fontsize=7, rotation=45, ha='right', color=INK)
ax2.set_ylabel('consecutive RSA\n(1=unchanged, low=re-cluster)', fontsize=10.5, color=INK); ax2.set_ylim(0.4, 1.02)
ax2.annotate('front re-clusters', xy=(0.5, 0.53), fontsize=9, color=SECONDARY)
ax2.annotate('middle near-frozen', xy=(8, 0.99), fontsize=9, color=SECONDARY, ha='center')
ax2.annotate('readout\nre-clusters', xy=(15, 0.83), fontsize=9, color=SECONDARY, ha='center')
ax2.grid(axis='y', color=GRID, lw=0.8); ax2.set_axisbelow(True)
for s in ('top','right'): ax2.spines[s].set_visible(False)
for ax in (ax1, ax2): ax.tick_params(colors=SECONDARY)
fig.tight_layout(); fig.savefig(PT+'geometry_trajectory.png', dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote geometry_trajectory.png')
