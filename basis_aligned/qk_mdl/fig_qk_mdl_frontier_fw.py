"""Figure v2 (tick 155): layer-0 QK MDL frontier with TRAINING-DISTRIBUTION cost.

Post-tick-154 convention: held-out delta-CE on FineWeb (307,200 predictions) is the headline cost
axis (Pile is off-distribution and shows a coarsening-helps confound). Panels:
  A. FineWeb delta-CE vs description length      B. factor FVU vs description length
  C. factor FVU vs FineWeb delta-CE
Reads qk_audit_big.json + qk_fw_fill.json + qk_sae_dict.json. Writes fig_qk_mdl_frontier_fw.png.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
BIG = json.load(open(f'{QK}/qk_audit_big.json'))['arms']
FILL = json.load(open(f'{QK}/qk_fw_fill.json'))['arms']
P2 = json.load(open(f'{QK}/qk_sae_dict.json'))['arms']

INK, SUB, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e4', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
V, ROW, NHB = 50304, 256, 18
MB = {'svd': lambda r: 32 * r * (V + ROW + 1) * NHB / 1e6,
      'dict1024': NHB * (32 * (1024 * ROW + ROW) + V * 8 * (32 + 10)) / 1e6,
      'merge_p': lambda K: (32 * NHB * K * ROW + NHB * V * 11) / 1e6}
RAW = 7417.6

# (Mbits, dce_fw, fvu) per family
svd = sorted([(FILL['svd rank 8']['Mbits'], FILL['svd rank 8']['dce_fw'], FILL['svd rank 8']['fvu'])] +
             [(MB['svd'](r), BIG[f'svd rank {r}']['dce_fw'], BIG[f'svd rank {r}']['fvu'])
              for r in (16, 32, 64, 128)])
merge_p = sorted([(FILL[f'merge K={K} per-head-branch']['Mbits'],
                   FILL[f'merge K={K} per-head-branch']['dce_fw'],
                   FILL[f'merge K={K} per-head-branch']['fvu']) for K in (256, 8192)] +
                 [(312.0, BIG['merge K=2048 per-head-branch']['dce_fw'], None)])
merge_g = [(FILL['merge K=2048 GLOBAL']['Mbits'], FILL['merge K=2048 GLOBAL']['dce_fw'],
            FILL['merge K=2048 GLOBAL']['fvu'])]
DICTS = [('token-linear', 'o', MB['dict1024'], BIG['dict n=1024 k=8 token-linear']['dce_fw'],
          BIG['dict n=1024 k=8 token-linear']['fvu']),
         ('token-OMP/LS', 's', MB['dict1024'], BIG['dict n=1024 k=8 token-OMP/LS']['dce_fw'],
          BIG['dict n=1024 k=8 token-OMP/LS']['fvu']),
         ('batch-topk', '^', FILL['dict n=1024 k=8 batch-topk']['Mbits'],
          FILL['dict n=1024 k=8 batch-topk']['dce_fw'], FILL['dict n=1024 k=8 batch-topk']['fvu']),
         ('matryoshka', 'D', FILL['dict n=1024 k=8 matryoshka']['Mbits'],
          FILL['dict n=1024 k=8 matryoshka']['dce_fw'], FILL['dict n=1024 k=8 matryoshka']['fvu']),
         ('OMP/LS n=4096', 'P', FILL['dict n=4096 k=8 token-OMP/LS']['Mbits'],
          FILL['dict n=4096 k=8 token-OMP/LS']['dce_fw'], FILL['dict n=4096 k=8 token-OMP/LS']['fvu'])]
two = (97.7, BIG['two-stage merge2048 -> OMP dict n=512 k=8']['dce_fw'],
       P2['two-stage merge2048 -> dict n=512 k=8 OMP/LS']['fvu'])
try:
    DC = json.load(open(f'{QK}/qk_dict_collapse.json'))          # dict + heads-2,5 collapse
except FileNotFoundError:
    DC = None

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURF, constrained_layout=True)
for ax in axes:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d3')
    ax.tick_params(colors=SUB, labelsize=9)

# A — FineWeb cost vs bits
ax = axes[0]
ax.axhline(0, color='#b9b8b3', linewidth=1, linestyle='--', zorder=1)
ax.plot([m for m, d, f in svd], [d for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
ax.plot([m for m, d, f in merge_p], [d for m, d, f in merge_p], '-s', color=ORANGE, linewidth=2,
        markersize=6, label='merge, per-head-branch', zorder=3)
ax.scatter([merge_g[0][0]], [merge_g[0][1]], marker='o', s=45, facecolors=SURF,
           edgecolors=ORANGE, linewidths=2, label='merge, global (K=2048)', zorder=3)
for arm, mk, mbits, d, f in DICTS:
    ax.scatter([mbits], [d], marker=mk, s=55, color=AQUA, edgecolors=SURF, linewidths=1.2,
               zorder=4, label=f'dict ({arm})')
ax.scatter([two[0]], [two[1]], marker='*', s=240, color=INK, zorder=5, label='two-stage')
if DC:
    ax.scatter([DC['Mbits']], [DC['dce_fw']], marker='X', s=90, color=INK, zorder=5,
               label='dict + heads 2,5 position-only')
    ax.annotate('dict + content-free heads\ncollapsed (4.8% bits)', (DC['Mbits'], DC['dce_fw']),
                textcoords='offset points', xytext=(-8, 10), ha='right', fontsize=8.5, color=SUB)
ax.scatter([RAW], [0.0], marker='o', s=40, color=INK, zorder=5)
ax.annotate('raw factors (exact)', (RAW, 0.0), textcoords='offset points', xytext=(-8, 10),
            ha='right', fontsize=8.5, color=SUB)
ax.annotate('dictionaries: +0.006 at 6% bits\n(= SVD r64 quality at ¼ the bits)',
            (MB['dict1024'], 0.0059), textcoords='offset points', xytext=(14, 16),
            fontsize=8.5, color=INK)
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('held-out ΔCE, FineWeb (nats)', color=SUB, fontsize=10)
ax.set_title('A — training-distribution cost vs description length', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=7, framealpha=0, labelcolor=SUB, loc='upper right')

# B — FVU vs bits
ax = axes[1]
ax.plot([m for m, d, f in svd], [f for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
mp = [(m, f) for m, d, f in merge_p if f is not None]
ax.plot([m for m, f in mp], [f for m, f in mp], '-s', color=ORANGE, linewidth=2, markersize=6,
        label='merge, per-head-branch', zorder=3)
for arm, mk, mbits, d, f in DICTS:
    ax.scatter([mbits], [f], marker=mk, s=55, color=AQUA, edgecolors=SURF, linewidths=1.2, zorder=4)
ax.scatter([two[0]], [two[2]], marker='*', s=240, color=INK, zorder=5)
ax.annotate('two-stage', (two[0], two[2]), textcoords='offset points', xytext=(10, 4),
            fontsize=8.5, color=INK)
ax.set_xscale('log')
ax.set_ylim(0, 1)
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('fraction of variance unexplained', color=SUB, fontsize=10)
ax.set_title('B — structural error vs description length', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=7.5, framealpha=0, labelcolor=SUB, loc='upper right')

# C — FVU vs FineWeb cost
ax = axes[2]
ax.axvline(0, color='#b9b8b3', linewidth=1, linestyle='--', zorder=1)
ax.plot([d for m, d, f in svd], [f for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
for arm, mk, mbits, d, f in DICTS:
    ax.scatter([d], [f], marker=mk, s=55, color=AQUA, edgecolors=SURF, linewidths=1.2, zorder=4)
mpd = [(d, f) for m, d, f in merge_p if f is not None]
ax.scatter([d for d, f in mpd], [f for d, f in mpd], marker='s', s=55, color=ORANGE,
           edgecolors=SURF, linewidths=1.2, zorder=4)
ax.scatter([two[1]], [two[2]], marker='*', s=240, color=INK, zorder=5)
ax.set_ylim(0, 1)
ax.set_xlabel('held-out ΔCE, FineWeb (nats)', color=SUB, fontsize=10)
ax.set_ylabel('fraction of variance unexplained', color=SUB, fontsize=10)
ax.set_title('C — on-distribution, FVU and cost mostly re-couple', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=7.5, framealpha=0, labelcolor=SUB, loc='upper left')
ax.text(0.03, 0.06, 'blue=SVD  orange=merge  teal=dictionaries  ★=two-stage',
        transform=ax.transAxes, fontsize=8.5, color=SUB)

fig.suptitle('Layer-0 query/key circuit: MDL frontier on the training distribution '
             '(FineWeb, 307k held-out predictions)', color=INK, fontsize=12.5, x=0.01, ha='left')
fig.text(0.01, -0.03,
         'Exact weight-only fold; baseline CE 3.0763. Pile audits (off-distribution) show a '
         'coarsening-helps confound and are no longer the headline axis (tick 154).',
         fontsize=8, color=SUB, ha='left')
fig.savefig(f'{QK}/fig_qk_mdl_frontier_fw.png', dpi=160, bbox_inches='tight', facecolor=SURF)
print('wrote fig_qk_mdl_frontier_fw.png')
