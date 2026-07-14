"""The one communicative graph: ΔCE vs description length (MDL) for every way
we've represented pythia-410m's embedding.

MDL convention: fp16 floats (2 bytes each) + discrete indices at log2(n) bits.
  svd r:            (V + d) r floats
  kmeans n:         n d floats + V log2(n) bits
  rq c,h:           h c d floats + V h log2(c) bits
  tree b,h:         nodes d floats + V h log2(b) bits (nodes from params field)
  topk dict n,k:    (n d + V k + d) floats + V k log2(n) bits
  original:         V d floats
"""

import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID

import os

BASE = '/workspace/tensor_language/basis_aligned'
r6 = json.load(open(f'{BASE}/e6_results.json'))
r7 = json.load(open(f'{BASE}/e7_results.json'))
r8 = json.load(open(f'{BASE}/e8_results.json')) if os.path.exists(f'{BASE}/e8_results.json') else None
V, D = r6['V'], r6['d']
MiB = 1024 * 1024


def mdl_bytes(floats, index_bits=0):
    return 2 * floats + index_bits / 8


pts = []  # (mdl, dce, series, label)
for row in r6['rows']:
    m, lab = row['method'], row['label']
    if m == 'svd':
        b = mdl_bytes(row['params'])
    elif m == 'kmeans':
        n = row['params'] // D
        b = mdl_bytes(row['params'], V * math.log2(n))
    elif m == 'rq':
        h = int(lab.split('h=')[1])
        b = mdl_bytes(row['params'], V * h * math.log2(1024))
    elif m == 'tree':
        bb, h = int(lab.split(',')[0][2:]), int(lab.split('h=')[1])
        b = mdl_bytes(row['params'], V * h * math.log2(bb))
    else:
        continue  # controls/noise excluded
    pts.append((b, row['dce'], 'unlearned', f"{m} {lab}"))

for row in r7['rows']:
    if row['method'] == 'topk_dict':
        n, k = row['n_atoms'], row['k']
        b = mdl_bytes(n * D + V * k + D, V * k * math.log2(n))
        pts.append((b, row['dce'], 'mse_dict', f'n={n} L0={k}'))
    elif row['method'] == 'topk_dict_ceft':
        n, k = row['n_atoms'], row['k']
        b = mdl_bytes(n * D + V * k + D, V * k * math.log2(n))
        pts.append((b, row['dce_after'], 'ce_dict', f'n={n} L0={k}'))
    elif row['method'] == 'kmeans_ceft':
        b = mdl_bytes(row['n_atoms'] * D, V * math.log2(row['n_atoms']))
        pts.append((b, row['dce_after'], 'ce_dict', 'k-means 25.6k CE-tuned'))

if r8 is not None:
    digit_bits = V * 4 * 4  # 4 digits x 4 bits per token
    for row in r8['rows']:
        if row['ordering'] == 'semantic' and 'dce' in row:
            pts.append((mdl_bytes(row['params'], digit_bits), row['dce'],
                        'tt', f"TT semantic r{row['rmax']}"))
    ft = r8['tt_ce_finetune']
    pts.append((mdl_bytes(ft['params'], digit_bits), ft['dce_after'],
                'tt_ce', 'TT semantic CE-trained'))

SERIES = {'unlearned': ('#2f9e63', 'o', 'unlearned (e6: SVD / k-means / RQ / tree)', 'none'),
          'mse_dict': ('#3987e5', 'o', 'learned top-k dict (MSE fit)', 'full'),
          'ce_dict': ('#e34948', '*', 'learned dict + CE-trained atoms', 'full'),
          'tt': ('#9a6ae1', 's', 'tensor-train (semantic order, TT-SVD)', 'none'),
          'tt_ce': ('#9a6ae1', '*', 'tensor-train + CE-trained cores', 'full')}

fig, ax = plt.subplots(figsize=(8.8, 5.8))
YMIN = 0.04
for s, (c, m, lab, fill) in SERIES.items():
    xs = [(b, d) for b, d, ss, _ in pts if ss == s]
    ax.plot([x / MiB for x, _ in xs], [max(d, YMIN) for _, d in xs], m,
            color=c, ms=13 if m == '*' else 6.5, ls='none', label=lab,
            mfc='none' if fill == 'none' else c, mew=1.5,
            mec=c if fill == 'none' else '#fcfcfb')

# pareto lower envelope across everything
env, best = [], float('inf')
for b, d, _, _ in sorted(pts):
    if d < best:
        best = d
        env.append((b, max(d, YMIN)))
ax.step([b / MiB for b, _ in env], [d for _, d in env], where='post',
        color=INK, lw=1.2, ls='--', alpha=0.6, zorder=1)

# annotate the headline points
for b, d, s, lab in pts:
    if s == 'ce_dict' and 'n=1024' in lab:
        ax.annotate(f'1024 atoms, L0=64:\n+{d:.2f} nats @ {b / MiB:.0f} MiB',
                    (b / MiB, d), textcoords='offset points', xytext=(10, 6),
                    fontsize=9, color=INK)
    if s == 'unlearned' and 'kmeans n=25k' in lab:
        ax.annotate('k-means, half the vocab', (b / MiB, d),
                    textcoords='offset points', xytext=(6, 4), fontsize=8,
                    color=SECONDARY)
    if s == 'unlearned' and lab == 'svd r=512':
        ax.annotate('SVD r=512', (b / MiB, d), textcoords='offset points',
                    xytext=(6, 4), fontsize=8, color=SECONDARY)

orig = mdl_bytes(V * D) / MiB
ax.axvline(orig, color=MUTED, lw=1.4, ls=':')
ax.text(orig * 0.97, 3.5, f'original E: {orig:.0f} MiB, ΔCE = 0', rotation=90,
        fontsize=8.5, color=SECONDARY, ha='right', va='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(YMIN, 9)
ax.set_yticks([0.05, 0.1, 0.3, 1, 3])
ax.set_yticklabels(['0.05', '0.1', '0.3', '1', '3'])
ax.set_xlabel('description length of the embedding representation (MiB; fp16 floats + index bits)',
              color=INK)
ax.set_ylabel('ΔCE (nats, log) — pythia-410m on pile-10k', color=INK)
ax.set_title('Behavior vs description length: every representation of the embedding\n'
             '(dashed = Pareto envelope; below-left is better)', fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc='lower left')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY, labelsize=9)
ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/e7_mdl.png', dpi=160)
print('saved')
