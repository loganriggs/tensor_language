import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
G = '/workspace/tensor_language/basis_aligned/tn_gauge'
ov_toy = json.load(open(f'{G}/toy_regime1_rotation.json'))
qk_toy = json.load(open(f'{G}/toy_qk_torus_floor.json'))
ov_fl = json.load(open(f'{G}/bilin18_regime1.json'))
ov_toy_drop = np.mean([v['drop_pct'] for v in ov_toy['OV_floors'].values()])
qk_toy_drop = qk_toy['mean_drop_pct']
ov_fl_drop = np.mean([r['l1_drop_pct'] for r in ov_fl['per_head_floor']])

fig, ax = plt.subplots(figsize=(7.5, 4.3))
labels = ['toy OV\n(full O(32))', 'toy QK\n(16-angle torus)', 'flagship OV\n(shared across depth)']
vals = [ov_toy_drop, qk_toy_drop, ov_fl_drop]
cols = ['#3b7dd8', '#5aa0e8', '#c0392b']
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('L1 sparsity gained by the EXACT rotation gauge (%)')
ax.set_title('Regime 1: the square-rotation baseline is nearly empty\n(all gauges verified ΔCE≈0; sparsity must come from overcompleteness)')
ax.set_ylim(0, 9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{G}/fig_regime1.png', dpi=120)
print('saved fig_regime1.png', ov_toy_drop, qk_toy_drop, ov_fl_drop)
