"""e5 figure: CE sits at the eps end; a ReLU readout breaks the linear rank bound."""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID

BASE = '/workspace/tensor_language/basis_aligned'
r = json.load(open(f'{BASE}/e5_results.json'))
arms = r['arms']
M, D_H = r['m'], r['d_h']
BOUND = r['oneactive_dedicated_mse']

# validated palette: blue = MSE-linear family, red = eps-family, green = nonlinear readout
ORDER = [
    ('dedicated_handcoded', 'dedicated\n(hand-coded)', '#3987e5'),
    ('scratch_mse', 'trained\nMSE', '#3987e5'),
    ('mse_relu', 'trained MSE\n+ ReLU readout', '#2f9e63'),
    ('superposition_handcoded', 'superposition\n(hand-coded)', '#e34948'),
    ('scratch_L8', 'trained\nL8', '#e34948'),
    ('ce', 'trained\nCE', '#e34948'),
    ('ce_smooth09', 'trained CE\nsmoothing 0.9', '#e34948'),
]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})

ax = axes[0]
xs = range(len(ORDER))
for i, (k, label, c) in enumerate(ORDER):
    n = arms[k]['n_cls']
    ax.bar(i, n, color=c, width=0.62)
    ax.text(i, n + 3, str(n), ha='center', fontsize=9.5, color=INK)
ax.axhline(M, color=MUTED, lw=1.2, ls=':')
ax.text(-0.4, M + 3, f'all m = {M}', fontsize=8.5, color=SECONDARY, ha='left')
ax.axhline(D_H, color=MUTED, lw=1.2, ls='--')
ax.text(6.4, D_H + 3, f'd_h = {D_H}', fontsize=8.5, color=SECONDARY, ha='right')
ax.set_xticks(list(xs))
ax.set_xticklabels([label for _, label, _ in ORDER], fontsize=7.8, color=INK)
ax.set_ylabel('# features computed (classification audit)', color=INK)
ax.set_ylim(0, 145)
ax.set_title('CE lands at the ε end (all m, even smoothed);\nReLU readout doubles MSE capacity',
             fontsize=11, color=INK)

ax = axes[1]
PTS = {'dedicated_handcoded': ('#3987e5', 's', 'dedicated (hand-coded)', (1.05, 3)),
       'scratch_mse': ('#3987e5', 'o', 'trained MSE', (1.05, -9)),
       'scratch_L8': ('#e34948', 'o', 'trained L8', (0.94, 3)),
       'superposition_handcoded': ('#e34948', 's', 'superposition (hand-coded)', (0.94, -10)),
       'mse_relu': ('#2f9e63', 'o', 'MSE + ReLU readout', (1.0, 8))}
for k, (c, m, label, (fx, fy)) in PTS.items():
    a = arms[k]
    ax.plot(a['mse_1active'], a['n_eps'], m, color=c, ms=10, mec='#fcfcfb', mew=1.2, zorder=5)
    ax.text(a['mse_1active'] * fx, a['n_eps'] + fy, label, fontsize=8.5, color=INK,
            ha='left' if fx > 1 else 'right')
ax.axvline(BOUND, color=INK, lw=1.4, ls='--')
ax.text(BOUND * 0.95, 95, 'rank bound (LINEAR readout only)', rotation=90, fontsize=8.5,
        color=SECONDARY, ha='right', va='center')
ax.set_xscale('log')
ax.set_xlim(5e-4, 7e-3)
ax.set_ylim(-6, 145)
ax.set_xlabel('MSE on 1-active inputs (log)', color=INK)
ax.set_ylabel(f'# features computed to ε = {r["eps_tol"]}', color=INK)
ax.set_title('the ReLU readout sits LEFT of the linear rank bound\nwith 2× the features',
             fontsize=11, color=INK)

for ax in axes:
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, axis='y')
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e5_ce_relu.png', dpi=160)
print('saved')
