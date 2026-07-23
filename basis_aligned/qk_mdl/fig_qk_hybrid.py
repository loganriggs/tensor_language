"""Figure (tick 166): the redrawn frontier — incoherent-rotary dictionaries + exact anchor
rows dominate every previously measured arm at every budget.
  A. Delta-CE vs description length: old arms (SVD, merges, MSE linear/OMP, plain-ctx) vs
     the new incoherent-rotary base curve and the anchor-hybrid curve (seed range at 493).
  B. Improvement factor over the old frontier (lower envelope of all old arms, log-bits
     interpolated) at each hybrid point.
Reads qk_pareto_sweep.json, qk_solutions.json, qk_hybrid_frontier.json. Writes fig_qk_hybrid.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
SW = json.load(open(f'{QK}/qk_pareto_sweep.json'))['jobs']
SOL = json.load(open(f'{QK}/qk_solutions.json'))['jobs']
HYB = json.load(open(f'{QK}/qk_hybrid_frontier.json'))['jobs']

INK, SUB, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e4', '#fcfcfb'
BLUE, ORANGE, AQUA, PURPLE, RED = '#2a78d6', '#eb6834', '#1baf7a', '#8657c7', '#c92f2f'

cfgs = [(256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8), (4096, 8), (8192, 8), (4096, 16)]
mb = [SW[f'n{n}_k{k}_s0']['Mbits'] for n, k in cfgs]
omp = [SW[f'n{n}_k{k}_s0']['dce_omp'] for n, k in cfgs]
ctx = [SW[f'n{n}_k{k}_s0']['dce_ctx'] for n, k in cfgs]

base_pts = sorted([(SOL['base']['Mbits'], SOL['base']['dce'])] +
                  [(HYB[j]['Mbits'], HYB[j]['dce']) for j in
                   ('b512', 'b1024', 'b4096k8', 'b4096k16')])
hyb_pts = sorted([(SOL[j]['Mbits'], SOL[j]['dce']) for j in ('s1_b64', 's1_b256', 's1_b1024')] +
                 [(HYB[j]['Mbits'], HYB[j]['dce']) for j in
                  ('h512_b256', 'h1024_b256', 'h1024_b1024', 'h4096k8_b1024', 'h4096k16_b1024')])
seeds493 = [HYB[f'h1024_b256{s}']['dce'] for s in ('', '_s1', '_s2')]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF,
                               constrained_layout=True, width_ratios=[1.6, 1])
for ax in (axA, axB):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d3')
    ax.tick_params(colors=SUB, labelsize=9)

ax = axA
ax.plot([233, 466, 932, 1864, 3728], [.0451, .0353, .0170, .0062, .0016], '-o', color=BLUE,
        linewidth=1.4, markersize=4, alpha=0.45, label='SVD rank 8…128', zorder=1)
ax.plot([45, 312, 1220], [.0423, .0196, .0080], '-s', color=ORANGE, linewidth=1.4,
        markersize=4, alpha=0.45, label='merge per-head-branch', zorder=1)
ax.plot(mb, omp, '-s', color=AQUA, linewidth=1.6, markersize=5, alpha=0.7,
        label='dict, MSE + OMP (tick 160)', zorder=2)
ax.plot(mb, ctx, '-^', color='#7a7a76', linewidth=1.6, markersize=5, alpha=0.8,
        label='dict, plain OV-context (tick 160)', zorder=2)
ax.plot([p[0] for p in base_pts], [p[1] for p in base_pts], '-D', color=PURPLE, linewidth=1.9,
        markersize=6, label='dict, INCOHERENT-rotary ctx (tick 163)', zorder=3)
ax.plot([p[0] for p in hyb_pts], [p[1] for p in hyb_pts], '-o', color=RED, linewidth=2.3,
        markersize=7, label='hybrid: incoh-rotary + exact anchors (tick 165/166)', zorder=4)
mmean = sum(seeds493) / 3
ax.errorbar([493.1], [mmean], yerr=[[mmean - min(seeds493)], [max(seeds493) - mmean]],
            color=RED, capsize=3, linewidth=1.2, zorder=5, fmt='none')
ax.annotate('+0.0011 @ 1074 Mbit —\nbelow the old frontier\nbest-anywhere (+0.0018 @ 1242)',
            (1073.9, .0011), textcoords='offset points', xytext=(-150, -6), fontsize=8.5,
            color=RED)
ax.annotate('old frontier at 493 Mbit: +0.0054\nhybrid: +0.0024 (3 seeds)',
            (493.1, .0024), textcoords='offset points', xytext=(10, 22), fontsize=8.5, color=SUB)
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('held-out ΔCE, FineWeb (nats)', color=SUB, fontsize=10)
ax.set_title('A — the redrawn frontier: error-analysis-driven fixes dominate everywhere',
             color=INK, fontsize=11, loc='left')
ax.legend(fontsize=8, framealpha=0, labelcolor=SUB, loc='upper right')

ax = axB
old_b = np.array(sorted(mb + [233, 466, 932, 1864, 3728, 45, 312, 1220]))
env = []
allold = list(zip(mb, [min(a, b) for a, b in zip(omp, ctx)])) + \
    list(zip([233, 466, 932, 1864, 3728], [.0451, .0353, .0170, .0062, .0016])) + \
    list(zip([45, 312, 1220], [.0423, .0196, .0080]))
allold.sort()
xs = np.array([p[0] for p in allold])
ys = np.array([p[1] for p in allold])
for bmb, bdce in hyb_pts:
    yin = np.interp(np.log(bmb), np.log(xs), ys)
    env.append((bmb, yin / bdce))
ax.axhline(1, color='#b9b8b3', linewidth=1, linestyle='--')
ax.plot([e[0] for e in env], [e[1] for e in env], '-o', color=RED, linewidth=2, markersize=6)
for bmb, r in env:
    ax.annotate(f'{r:.1f}×', (bmb, r), textcoords='offset points', xytext=(0, 8),
                fontsize=8.5, color=INK, ha='center')
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('ΔCE improvement factor over old frontier (matched bits)', color=SUB, fontsize=10)
ax.set_title('B — improvement over the tick-160 frontier envelope', color=INK,
             fontsize=11, loc='left')

fig.suptitle('Layer-0 QK: the error-analysis arc (ticks 161–166) — exploration → diagnosis → '
             'solutions. FineWeb 307k held-out predictions, seed bars at 493 Mbit.',
             color=INK, fontsize=12, x=0.01, ha='left')
fig.text(0.01, -0.05,
         'Hybrid = per-head-branch dictionaries trained on the incoherent-rotary OV-context '
         'objective (systematic term T²·E_Δ‖μ_Δ‖², all rotary bands preserved) + exact factor '
         'rows for the top-B tokens by delivered-error attribution (B=64–1024; bits charged). '
         'Nulls along the way: reader LoRA (161), co-occurrence weights (162), head '
         'reallocation & tail weighting (165), blend once incoh-rotary is in (165).',
         fontsize=8, color=SUB, ha='left')
fig.savefig(f'{QK}/fig_qk_hybrid.png', dpi=160, bbox_inches='tight', facecolor=SURF)
print('wrote fig_qk_hybrid.png')
