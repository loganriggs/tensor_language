"""Capstone bottom-up map: the three registers of bilin18 read straight off the layers.
Top: per-layer linear-recoverability of each MLP (R^2, §942) — the linear->multiplicative->linear arc.
Bottom: logit-lens CE by late layer (§944) — where the prediction is actually formed."""
import sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
r2 = json.load(open(PT+'middle_nonlinearity_crossmodel_results.json'))['models']['bilin18']['r2_by_layer']
lens = json.load(open(PT+'readout_role_results.json'))['logit_lens_ce']
ce_full = json.load(open(PT+'readout_role_results.json'))['ce_full']
L = np.arange(18)

FRONT='#1b7837'; MID='#8c2b2b'; READ='#104281'
def reg(i): return FRONT if i<=5 else (MID if i<=15 else READ)
cols=[reg(i) for i in L]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 7.6), dpi=140, gridspec_kw={'height_ratios':[1.15,1]})
fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2): ax.set_facecolor(SURFACE)

# top: linear R2 arc
ax1.bar(L, r2, width=0.72, color=cols, edgecolor=SURFACE, linewidth=1.2, zorder=3)
ax1.plot(L, r2, color=INK, lw=1.0, alpha=0.5, zorder=4, marker='o', ms=3)
ax1.set_xticks(L); ax1.set_xticklabels([str(i) for i in L], fontsize=8, color=SECONDARY)
ax1.set_ylabel('MLP output linearly\nrecoverable (R²)', fontsize=10, color=INK)
ax1.set_ylim(0, 1.0)
ax1.set_title('bilin18, read off the layers: linear front → multiplicative middle → linear readout',
              fontsize=12.5, color=INK, pad=10)
for i in [0,1,15,17]: ax1.text(i, r2[i]+0.02, f'{r2[i]:.2f}', ha='center', va='bottom', color=INK, fontsize=8)
ax1.text(2.5, 0.9, 'FRONT\ngrammar (linear)', ha='center', color=FRONT, fontsize=9, fontweight='bold')
ax1.text(10.5, 0.62, 'MIDDLE\ncontent (multiplicative)', ha='center', color=MID, fontsize=9, fontweight='bold')
ax1.text(16.5, 0.55, 'READOUT\n(linear)', ha='center', color=READ, fontsize=9, fontweight='bold')

# bottom: logit lens
lx=[int(k) for k in lens]; lv=[lens[k] for k in lens]
ax2.plot(lx, lv, color=READ, lw=2.2, marker='o', ms=6, zorder=3)
ax2.axhline(ce_full, color=INK, ls=(0,(4,3)), lw=1.0, zorder=2)
ax2.text(12.1, ce_full+0.12, f'final CE {ce_full:.2f}', color=INK, fontsize=9)
for k in lens: ax2.text(int(k), lens[k]+0.12, f'{lens[k]:.1f}', ha='center', color=READ, fontsize=8.5)
ax2.annotate('', xy=(17,3.35), xytext=(15,5.7), arrowprops=dict(arrowstyle='->', color=MID, lw=2))
ax2.text(15.6, 5.0, 'last 2 blocks close\na 2.6-nat gap\n(near-linear, §945)', color=MID, fontsize=8.5)
ax2.set_xticks(lx); ax2.set_xticklabels([str(i) for i in lx], fontsize=9, color=SECONDARY)
ax2.set_xlabel('layer', fontsize=10, color=INK)
ax2.set_ylabel('logit-lens CE\n(read output off layer)', fontsize=10, color=INK)
ax2.set_title('The answer is assembled by L15 but not output-readable; the readout rotates it into tokens',
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=SECONDARY); ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
plt.tight_layout(); out=PT+'stack_map.png'
plt.savefig(out, bbox_inches='tight', facecolor=SURFACE); print('wrote', out)
