"""Consolidated map of what blocks 0-2 do: benefit (bars), functional rank
(r80), plain-language role, and category lean. Matches palette.py."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
# (label, benefit nats, r80, role, lean)  -- from 699/701/703/711
rows = [
    ('block0.attn', 1.28, '2',   'suffix/morphology + boundary→continuation', 'syntactic'),
    ('block0.mlp',  2.08, '8',   'broad class / open-vocab decision',              'open-vocab'),
    ('block1.attn', 2.06, '1',   'boundary detector → continuation',          'subword'),
    ('block1.mlp',  1.02, '128', 'open-vocab (genuinely high-rank)',               'subword'),
    ('block2.attn', 0.32, '8',   'minor',                                          'flat'),
    ('block2.mlp',  0.15, '256', 'negligible',                                     'flat'),
]
LEANCOL = {'syntactic': '#3987e5', 'open-vocab': '#e34948', 'subword': '#8c2b2b',
           'flat': MUTED}

fig, ax = plt.subplots(figsize=(10, 4.8)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ys = range(len(rows))[::-1]
for y, (lab, ben, r80, role, lean) in zip(ys, rows):
    ax.barh(y, ben, height=0.62, color=LEANCOL[lean], edgecolor=SURFACE, lw=1.2, zorder=3)
    ax.text(-0.06, y, lab, ha='right', va='center', fontsize=10.5, color=INK, family='monospace')
    ax.text(ben + 0.04, y, f'r80={r80}', ha='left', va='center', fontsize=9,
            color=SECONDARY, family='monospace')
    ax.text(ben + 0.42, y, role, ha='left', va='center', fontsize=9.2, color=INK)

ax.set_xlim(0, 4.7); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks([])
ax.set_xlabel('benefit: cross-entropy nats lost if the component is ablated', fontsize=10.5)
ax.set_title('What the first three blocks of bilin18 do\n'
             'bar = contribution, r80 = functional rank of its decoder map, color = category lean',
             fontsize=12.5, color=INK, loc='left', pad=12)
ax.grid(True, axis='x', color=GRID, lw=0.7, zorder=0)
for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(SECONDARY); ax.tick_params(colors=SECONDARY, labelsize=9)
# legend for lean
from matplotlib.patches import Patch
leg = ax.legend(handles=[Patch(color=LEANCOL[k], label=k) for k in
                         ['syntactic', 'open-vocab', 'subword', 'flat']],
                loc='lower right', fontsize=9, frameon=True, framealpha=0.95, edgecolor=GRID,
                title='category lean')
leg.get_frame().set_facecolor(SURFACE)
ax.text(4.55, -0.35, 'low r80 = few decoder directions; high r80 = genuinely high-rank',
        ha='right', va='bottom', fontsize=8, color=MUTED, style='italic')

fig.tight_layout()
out = PT + 'front_layers_map.png'
fig.savefig(out, dpi=150, facecolor=SURFACE); print('wrote', out)
