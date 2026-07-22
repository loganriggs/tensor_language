"""Figure: layer-0 QK MDL frontier (tick 152) — three panels from the result JSONs.

  A. held-out delta-CE vs description length (all arms: SVD frontier, stage-1 merges,
     sparse dictionaries, two-stage)
  B. fraction of variance unexplained (FVU) vs description length (arms that record FVU)
  C. FVU vs delta-CE (the structural-vs-behavioral decoupling)

Reads qk_merge_stage1_l0.json + qk_sae_dict.json; writes fig_qk_mdl_frontier.png.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
S1 = json.load(open(f'{QK}/qk_merge_stage1_l0.json'))['arms']
S2 = json.load(open(f'{QK}/qk_sae_dict.json'))['arms']

INK, SUB, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e4', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # svd / merges / dictionaries

# --- collect series ---------------------------------------------------------
svd = [(v['Mbits'], v['dce'], v['fvu']) for k, v in S2.items() if k.startswith('svd rank')]
svd.sort()

merge_g = {}                                              # global partition: best dce per K
for k, v in S1.items():
    if k.startswith('stage1 GLOBAL merge'):
        K = v['K']
        if K not in merge_g or v['dce'] < merge_g[K][1]:
            merge_g[K] = (v['Mbits'], v['dce'])
merge_g = [merge_g[K] for K in sorted(merge_g)]
merge_p = sorted((v['Mbits'], v['dce']) for k, v in S1.items() if k.startswith('per-head-branch'))

DICT_MARK = {'token-linear': 'o', 'token-OMP/LS': 's', 'batch-topk': '^', 'matryoshka': 'D'}
dicts = []                                                # (Mbits, dce, fvu, arm)
for k, v in S2.items():
    if k.startswith('dict') and '(no renorm)' not in k:
        arm = next(a for a in DICT_MARK if k.endswith(a))
        dicts.append((v['Mbits'], v['dce'], v['fvu'], arm))
two = next(v for k, v in S2.items() if k.startswith('two-stage'))
RAW = 7417.6

# --- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURF, constrained_layout=True)
for ax in axes:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d3')
    ax.tick_params(colors=SUB, labelsize=9)

def dict_points(ax, ycol):
    for arm, mk in DICT_MARK.items():
        pts = [(m, d, f) for (m, d, f, a) in dicts if a == arm]
        ys = [p[ycol] for p in pts]
        ax.scatter([p[0] for p in pts], ys, marker=mk, s=55, color=AQUA,
                   edgecolors=SURF, linewidths=1.2, zorder=4,
                   label=f'dictionary ({arm})')

# Panel A: delta-CE vs MDL --------------------------------------------------
ax = axes[0]
ax.axhline(0, color='#b9b8b3', linewidth=1, linestyle='--', zorder=1)
ax.plot([m for m, d, f in svd], [d for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
ax.plot([m for m, d in merge_g], [d for m, d in merge_g], '--o', color=ORANGE, linewidth=2,
        markersize=6, markerfacecolor=SURF, label='merge, global partition', zorder=3)
ax.plot([m for m, d in merge_p], [d for m, d in merge_p], '-s', color=ORANGE, linewidth=2,
        markersize=6, label='merge, per-head-branch', zorder=3)
dict_points(ax, 1)
ax.scatter([two['Mbits']], [two['dce']], marker='*', s=260, color=INK, zorder=5,
           label='two-stage: merge 2048 → dict')
ax.scatter([RAW], [0.0], marker='o', s=40, color=INK, zorder=5)
ax.annotate('raw factors\n(exact, 7418 Mbit)', (RAW, 0.0), textcoords='offset points',
            xytext=(-8, 10), ha='right', fontsize=8.5, color=SUB)
ax.annotate('two-stage (98 Mbit, 76×)', (two['Mbits'], two['dce']),
            textcoords='offset points', xytext=(-4, -18), ha='right', fontsize=8.5, color=INK)
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('held-out ΔCE (nats)', color=SUB, fontsize=10)
ax.set_title('A — cost vs description length', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=7.5, framealpha=0, labelcolor=SUB, loc='upper right', ncols=1)

# Panel B: FVU vs MDL -------------------------------------------------------
ax = axes[1]
ax.plot([m for m, d, f in svd], [f for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
dict_points(ax, 2)
ax.scatter([two['Mbits']], [two['fvu']], marker='*', s=260, color=INK, zorder=5,
           label='two-stage')
ax.annotate('two-stage', (two['Mbits'], two['fvu']), textcoords='offset points',
            xytext=(10, 4), fontsize=8.5, color=INK)
ax.set_xscale('log')
ax.set_ylim(0, 1)
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('fraction of variance unexplained (factor tables)', color=SUB, fontsize=10)
ax.set_title('B — structural error vs description length', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=7.5, framealpha=0, labelcolor=SUB, loc='upper right')

# Panel C: FVU vs delta-CE --------------------------------------------------
ax = axes[2]
ax.axvline(0, color='#b9b8b3', linewidth=1, linestyle='--', zorder=1)
ax.plot([d for m, d, f in svd], [f for m, d, f in svd], '-o', color=BLUE, linewidth=2,
        markersize=6, label='SVD rank 8…128', zorder=3)
for arm, mk in DICT_MARK.items():
    pts = [(d, f) for (m, d, f, a) in dicts if a == arm]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=55, color=AQUA,
               edgecolors=SURF, linewidths=1.2, zorder=4, label=f'dictionary ({arm})')
ax.scatter([two['dce']], [two['fvu']], marker='*', s=260, color=INK, zorder=5, label='two-stage')
ax.annotate('SVD r=8: low bits,\nhigh cost', (svd[0][1], svd[0][2]), textcoords='offset points',
            xytext=(-10, 8), ha='right', fontsize=8.5, color=SUB)
ax.text(0.03, 0.06, 'dictionaries: worse FVU than SVD r=128,\nyet ΔCE ≤ 0 — variance ≠ behavior',
        transform=ax.transAxes, fontsize=8.5, color=SUB)
ax.set_ylim(0, 1)
ax.set_xlabel('held-out ΔCE (nats)', color=SUB, fontsize=10)
ax.set_ylabel('fraction of variance unexplained (factor tables)', color=SUB, fontsize=10)
ax.set_title('C — structural error does not predict behavioral cost', color=INK,
             fontsize=11, loc='left')
ax.legend(fontsize=7.5, framealpha=0, labelcolor=SUB, loc='upper left')

fig.suptitle('Layer-0 query/key circuit: MDL decomposition frontier (exact weight-only fold, bilin18)',
             color=INK, fontsize=12.5, x=0.01, ha='left')
fig.text(0.01, -0.04,
         'Audit: 16 held-out Pile sequences at T=512 (8,192 predictions), baseline CE 3.2341. '
         'Merges lack FVU (panels B/C show SVD, dictionary, and two-stage arms). '
         'Wide-audit check (65,536 predictions, running): all arms shift +0.01…+0.02 except the '
         'dictionary/two-stage arms, which stay ≈0 — negative values here are audit noise, not real improvement.',
         fontsize=8, color=SUB, ha='left')
fig.savefig(f'{QK}/fig_qk_mdl_frontier.png', dpi=160, bbox_inches='tight', facecolor=SURF)
print('wrote fig_qk_mdl_frontier.png')
