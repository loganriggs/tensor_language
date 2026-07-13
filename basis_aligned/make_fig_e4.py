"""e4 figure: the metric decides superposition — MSE vs #eps-computed features."""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID

BASE = '/workspace/tensor_language/basis_aligned'
r = json.load(open(f'{BASE}/e4_results.json'))
arms, bl = r['arms'], r['baselines']

STYLE = {  # validated categorical palette; hand-coded = squares, trained = circles
    'dedicated_handcoded': ('#3987e5', 's', 'dedicated (hand-coded)'),
    'superposition_handcoded': ('#e34948', 's', 'superposition (hand-coded)'),
    'scratch_mse': ('#3987e5', 'o', 'trained, MSE loss'),
    'scratch_L8': ('#e34948', 'o', 'trained, L8 loss'),
    'superpos_then_mse_ft': ('#9a6ae1', 'o', 'superposition → MSE finetune'),
    'superpos_then_L8_ft': ('#2f9e63', 'o', 'superposition → L8 finetune'),
}
OFFSET = {  # label offsets (x-mult, y-add)
    'dedicated_handcoded': (1.05, 3), 'superposition_handcoded': (1.0, -22),
    'scratch_mse': (1.05, 2), 'scratch_L8': (0.94, 4),
    'superpos_then_mse_ft': (1.05, -11), 'superpos_then_L8_ft': (1.06, -12),
}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, (c, m, label) in STYLE.items():
    a = arms[name]
    x, y = a['mse_1active'], a['n_computed']
    ax.plot(x, y, m, color=c, ms=10, mec='#fcfcfb', mew=1.2, zorder=5)
    fx, fy = OFFSET[name]
    ax.text(x * fx, y + fy, label, fontsize=9, color=INK,
            ha='left' if fx > 1 else 'right')
# finetune arrows from the hand-coded superposition point
sp = arms['superposition_handcoded']
for tgt in ['superpos_then_mse_ft', 'superpos_then_L8_ft']:
    a = arms[tgt]
    ax.annotate('', xy=(a['mse_1active'], a['n_computed']),
                xytext=(sp['mse_1active'], sp['n_computed']),
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2, ls='--'))
ax.axvline(bl['oneactive_dedicated'], color=INK, lw=1.2, ls='--')
ax.text(bl['oneactive_dedicated'] * 0.95, 70, 'dedicated-baseline MSE', rotation=90,
        fontsize=8.5, color=SECONDARY, ha='right', va='center')
ax.axhline(r['d_h'], color=MUTED, lw=1.2, ls=':')
ax.text(6e-3, r['d_h'] + 3, f"d_h = {r['d_h']} (dedicated ceiling)", fontsize=8.5,
        color=SECONDARY, ha='right')
ax.axhline(r['m'], color=MUTED, lw=1.2, ls=':')
ax.text(8.5e-4, r['m'] - 9, f"all m = {r['m']}", fontsize=8.5, color=SECONDARY, ha='left')
ax.set_xscale('log')
ax.set_xlim(8e-4, 7e-3)
ax.set_ylim(-6, 140)
ax.set_xlabel('MSE on 1-active inputs (log) — the metric of e3', color=INK)
ax.set_ylabel(f"# features computed to ε = {r['eps_tol']} — the metric of the CiS post", color=INK)
ax.set_title('Same architecture, same data (m=128 squares, d_h=32):\n'
             'the loss alone decides whether superposition exists', fontsize=11, color=INK)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY, labelsize=9)
ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e4_metric.png', dpi=160)
print('saved')
