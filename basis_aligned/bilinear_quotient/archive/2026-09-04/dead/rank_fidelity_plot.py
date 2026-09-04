"""Plot rank vs fidelity (recovered loss-benefit) for the front components,
from rspd_front_layers_scaled_results.json. MLPs emphasized (solid),
attention as context (dashed). Matches repo palette.py."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
d = json.load(open(PT + 'rspd_front_layers_scaled_results.json'))
fm = d['front_map']

# color = block, linestyle = component type
BLOCKCOL = {'0': '#3987e5', '1': '#e34948', '2': INK}
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

fig, ax = plt.subplots(figsize=(8.2, 5.4))
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

for key, v in fm.items():
    blk = key[5]                      # 'block0.mlp' -> '0'
    kind = key.split('.')[1]
    rec = v['recovered']
    xs = [r for r in RANKS if str(r) in rec]
    ys = [rec[str(r)] for r in xs]
    is_mlp = (kind == 'mlp')
    ax.plot(xs, ys, marker='o' if is_mlp else '^',
            ms=6 if is_mlp else 4.5, lw=2.2 if is_mlp else 1.3,
            color=BLOCKCOL[blk], ls='-' if is_mlp else '--',
            alpha=1.0 if is_mlp else 0.55,
            label=f"{key}  (benefit {v['benefit']:.2f} nats, r80={v['r80']})",
            zorder=5 if is_mlp else 3)
    # mark r80 for MLPs
    if is_mlp and v['r80'] is not None and str(v['r80']) in rec:
        ax.scatter([v['r80']], [rec[str(v['r80'])]], s=120, facecolor='none',
                   edgecolor=BLOCKCOL[blk], lw=2.0, zorder=6)

ax.axhline(0.80, color=SECONDARY, lw=1.2, ls=':', zorder=2)
ax.text(1.05, 0.815, '80% (r80 threshold)', color=SECONDARY, fontsize=9, va='bottom')
ax.axhline(1.0, color=MUTED, lw=1.0, ls='-', zorder=1)
ax.axhline(0.0, color=MUTED, lw=1.0, ls='-', zorder=1)
ax.text(300, 0.02, 'ablation (layer removed)', color=MUTED, fontsize=8, va='bottom', ha='right')

ax.set_xscale('log', base=2)
ax.set_xticks(RANKS); ax.set_xticklabels([str(r) for r in RANKS])
ax.set_xlim(0.9, 560); ax.set_ylim(-5.6, 1.15)
ax.set_xlabel('rank r of the data-conditioned A-SVD surrogate (log scale)', fontsize=11)
ax.set_ylabel('fidelity = fraction of the layer\'s loss-benefit recovered', fontsize=11)
ax.set_title('bilin18 front layers: how many decoder directions reproduce each layer',
             fontsize=12.5, color=INK, loc='left', pad=12)
ax.grid(True, which='major', color=GRID, lw=0.7, zorder=0)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color(SECONDARY)
ax.tick_params(colors=SECONDARY, labelsize=9)
leg = ax.legend(loc='lower right', fontsize=8.3, frameon=True, framealpha=0.95,
                edgecolor=GRID)
leg.get_frame().set_facecolor(SURFACE)
ax.annotate('MLPs 1 & 2: low-rank surrogates are\nWORSE than deleting the layer',
            xy=(4, -4.1), xytext=(9, -5.2), fontsize=8.6, color='#8c2b2b',
            arrowprops=dict(arrowstyle='->', color='#8c2b2b', lw=1.0))

fig.tight_layout()
out = PT + 'rank_fidelity_front.png'
fig.savefig(out, dpi=150, facecolor=SURFACE)
print('wrote', out)
