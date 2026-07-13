"""Figures for e2 (trajectories + folded weights per arm) and e3 (superposition)."""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from palette import DIVERGING, BLUES, INK, SECONDARY, MUTED, GRID
from common import fold, interaction, near_zero_frac

BASE = '/workspace/tensor_language/basis_aligned'
# validated categorical palette (see session log): blue red green purple
C = {'mid_LRD': '#3987e5', 'all_ELRDU': '#e34948', 'rot_handcoded': '#9a6ae1',
     'trained': '#3987e5', 'sparse': '#e34948', 'extra': '#2f9e63'}
ARM_LABEL = {'none': 'no sparsification', 'mid_LRD': 'sparsify L,R,D only',
             'all_ELRDU': 'sparsify all incl. E,U', 'rot_handcoded': 'rotated hand-coded init'}


def style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


# ================================================================ e2
res = json.load(open(f'{BASE}/e2_results.json'))

fig, ax = plt.subplots(figsize=(7, 4.6))
for arm in ['mid_LRD', 'all_ELRDU', 'rot_handcoded']:
    runs = [r for r in res if r['arm'] == arm]
    for i, r in enumerate(runs):
        h = [(s['frac_remaining'], max(s['fvu'], 1e-9)) for s in r['hist'] if s['phase'] == 'l1']
        fin = [(s['frac_remaining'], max(s['fvu'], 1e-9)) for s in r['hist'] if 'finetune' in s['phase']]
        fr, fv = zip(*h)
        ax.plot(fr, fv, color=C[arm], lw=2, alpha=0.85,
                label=ARM_LABEL[arm] if i == 0 else None)
        if fin:
            ax.plot(fin[0][0], fin[0][1], 'o', color=C[arm], ms=8,
                    mec='#fcfcfb', mew=1.5, zorder=5)
ax.axhline(0.01, color=MUTED, lw=1.2, ls='--')
ax.text(0.985, 0.0115, 'degradation threshold (FVU 0.01)', ha='right', fontsize=8.5,
        color=SECONDARY)
ax.set_yscale('log')
ax.set_xlim(1.02, 0)  # reversed: pruning moves rightward
ax.set_xlabel('fraction of sparsified weights remaining', color=INK)
ax.set_ylabel('val FVU (log)', color=INK)
ax.set_title('Iterated L1 + prune: error vs sparsity (dot = post-revert finetune)',
             fontsize=11, color=INK)
style(ax)
ax.legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e2_trajectory.png', dpi=160)

# folded weights + interaction form per arm, seed 0
arms = ['none', 'mid_LRD', 'all_ELRDU', 'rot_handcoded']
fig, axes = plt.subplots(4, 4, figsize=(10.5, 11))
for row, arm in enumerate(arms):
    st = torch.load(f'{BASE}/e2_state_{arm}_s0.pt')
    p = st['p']
    f = fold(p)
    B = interaction(p)
    Babs = B.abs().sum(0)
    for col, (k, w) in enumerate([('L̃ = LE', f['Lf']), ('R̃ = RE', f['Rf']),
                                  ('D̃ = UD', f['Df']), ('Σ_c |B_c|', Babs)]):
        ax = axes[row, col]
        w = w.numpy()
        if col < 3:
            v = max(abs(w).max(), 1e-9)
            ax.imshow(w, cmap=DIVERGING, vmin=-v, vmax=v)
        else:
            ax.imshow(w, cmap=BLUES, vmin=0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        if row == 0:
            ax.set_title(k, fontsize=11, color=INK)
        if col < 3:
            ax.set_xlabel(f'zeros {near_zero_frac(torch.tensor(w)):.0%}',
                          fontsize=8, color=SECONDARY, labelpad=2)
    r0 = [r for r in res if r['arm'] == arm and r['seed'] == 0][0]
    axes[row, 0].set_ylabel(
        f"{ARM_LABEL[arm]}\nblock {r0['block_score']:.2f} · "
        f"junk {r0['B_mass']['cross_block_unprobed']:.2f}",
        fontsize=9.5, color=INK)
fig.suptitle('Folded weights and invariant interaction form Σ_c|B_c| by arm (seed 0)\n'
             'junk = |B| mass on never-probed cross-block entries', fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{BASE}/figures/e2_folded.png', dpi=160)

# ================================================================ e3
r3 = json.load(open(f'{BASE}/e3_results.json'))
sw = r3['sweep']
dh = [r['d_h'] for r in sw]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
sw_pos = [r for r in sw if r['rank_bound'] > 0]  # d_h=64 has bound 0: off log scale
ax.plot(dh, [r['mse'] for r in sw], 'o-', color=C['trained'], lw=2, ms=7, label='trained bilinear')
ax.plot([r['d_h'] for r in sw_pos], [r['dedicated'] for r in sw_pos], 's--', color=INK,
        lw=1.5, ms=5, label='dedicated baseline')
ax.plot([r['d_h'] for r in sw_pos], [r['rank_bound'] for r in sw_pos], '-', color=C['extra'],
        lw=1.5, label='rank bound (no net can beat this)')
ax.axhline(r3['base_zero'], color=MUTED, lw=1.2, ls=':')
ax.text(dh[-1], r3['base_zero'] * 1.15, 'predict zero', ha='right', fontsize=8.5, color=SECONDARY)
ax.set_yscale('log')
ax.set_xlabel('hidden dims d_h', color=INK)
ax.set_ylabel('MSE (log)', color=INK)
ax.set_title(f'm={r3["m"]} squares, p={r3["p_active"]}', fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9)
style(ax)

ax = axes[1]
ax.plot(dh, [r['n_computed'] for r in sw], 'o-', color=C['trained'], lw=2, ms=7,
        label='features computed (|c_i−1|<0.25)')
ax.plot(dh, dh, '--', color=INK, lw=1.5, label='d_h (dedicated ceiling)')
ax.axhline(r3['m'], color=MUTED, lw=1.2, ls=':')
ax.text(dh[0], r3['m'] - 4, f'all {r3["m"]}', fontsize=8.5, color=SECONDARY)
ax.set_xlabel('hidden dims d_h', color=INK)
ax.set_ylabel('# features computed', color=INK)
ax.set_title('no superposition: computed ≈ d_h', fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc='lower right')
style(ax)

ax = axes[2]
sp = r3['sparsify_16']
h = [(s['frac_remaining'], s['fvu']) for s in sp['hist'] if s['phase'] == 'l1']
fin = [(s['frac_remaining'], s['fvu']) for s in sp['hist'] if 'finetune' in s['phase']]
fr, fv = zip(*h)
ax.plot(fr, fv, color=C['sparse'], lw=2, label='L1 + prune trajectory')
if fin:
    ax.plot(fin[0][0], fin[0][1], 'o', color=C['sparse'], ms=8, mec='#fcfcfb', mew=1.5,
            zorder=5, label='post-revert finetune')
ax.axhline(sp['final']['dedicated_16'], color=INK, lw=1.5, ls='--')
ax.text(0.02, sp['final']['dedicated_16'] * 1.02, 'dedicated(16) baseline', fontsize=8.5,
        color=SECONDARY, ha='left', va='bottom')
if 'degrade_threshold' in sp['final']:
    ax.axhline(sp['final']['degrade_threshold'], color=MUTED, lw=1.2, ls='--')
    ax.text(0.02, sp['final']['degrade_threshold'] * 1.02, 'degradation threshold',
            fontsize=8.5, color=SECONDARY, ha='left', va='bottom')
ax.axvline(sp['final']['dedicated_frac_weights'], color=MUTED, lw=1.2, ls=':')
ax.text(sp['final']['dedicated_frac_weights'] * 1.1, 7.42e-3, 'dedicated\nweight count',
        fontsize=8, color=SECONDARY, ha='left')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(1.2, 0.008)  # reversed: pruning moves rightward
ax.set_xlabel('fraction of weights remaining (log)', color=INK)
ax.set_ylabel('MSE (log)', color=INK)
ax.set_title(f"sparsify d_h=16: computed {sp['final']['n_computed']} at end", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc='upper left')
style(ax)

fig.tight_layout()
fig.savefig(f'{BASE}/figures/e3_superposition.png', dpi=160)
print('figures saved')
