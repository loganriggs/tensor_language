"""Figure (tick 160): the in-depth dictionary Pareto sweep — objectives x budgets x seeds.

  A. FineWeb delta-CE vs description length: MSE-linear, MSE-OMP, OV-context-trained dictionary
     curves over 8 budgets (2.5%-21% raw), seed min-max bars at the three anchor budgets;
     SVD and merge baselines for context.
  B. Paired per-seed improvement of the OV-context objective over each MSE encoder, vs budget —
     shows where OV-training wins (low bits) and the crossover (high bits).
Reads qk_pareto_sweep.json (+ qk_fw_fill/qk_audit_big for baselines). Writes fig_qk_pareto.png.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
SW = json.load(open(f'{QK}/qk_pareto_sweep.json'))['jobs']

INK, SUB, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e4', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

CONFIGS = [(256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8), (4096, 8), (8192, 8), (4096, 16)]
CONFIGS = sorted(CONFIGS, key=lambda nk: SW[f'n{nk[0]}_k{nk[1]}_s0']['Mbits'])
ANCHORS = [(512, 4), (1024, 8), (4096, 8)]


def s0(n, k):
    return SW[f'n{n}_k{k}_s0']


def seeds(n, k, field):
    return [SW[f'n{n}_k{k}_s{s}'][field] for s in (0, 1, 2)]


MB = [s0(n, k)['Mbits'] for (n, k) in CONFIGS]
LIN = [s0(n, k)['dce_lin'] for (n, k) in CONFIGS]
OMP = [s0(n, k)['dce_omp'] for (n, k) in CONFIGS]
CTX = [s0(n, k)['dce_ctx'] for (n, k) in CONFIGS]
DEGEN = [(n, k) == (8192, 8) for (n, k) in CONFIGS]        # linear encoder degenerated (fvu 1.18)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF,
                               constrained_layout=True, width_ratios=[1.5, 1])
for ax in (axA, axB):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d3')
    ax.tick_params(colors=SUB, labelsize=9)

# A — the frontier
ax = axA
ax.plot([233, 466, 932, 1864, 3728], [.0451, .0353, .0170, .0062, .0016], '-o', color=BLUE,
        linewidth=1.6, markersize=5, alpha=0.65, label='SVD rank 8…128', zorder=2)
ax.plot([45, 312, 1220], [.0423, .0196, .0080], '-s', color=ORANGE, linewidth=1.6,
        markersize=5, alpha=0.65, label='merge per-head-branch', zorder=2)
lin_ok = [(m_, d) for m_, d, dg in zip(MB, LIN, DEGEN) if not dg]
ax.plot([m_ for m_, d in lin_ok], [d for m_, d in lin_ok], '--o', color=AQUA, linewidth=1.8,
        markersize=6, markerfacecolor=SURF, label='dict, MSE (linear encoder)', zorder=3)
dg = [(m_, d) for m_, d, g_ in zip(MB, LIN, DEGEN) if g_]
ax.scatter([m_ for m_, d in dg], [d for m_, d in dg], marker='o', s=42, facecolors=SURF,
           edgecolors=AQUA, linewidths=1.4, zorder=3)
ax.annotate('linear encoder degenerates\n(FVU 1.18; atoms fine — see OMP)',
            dg[0], textcoords='offset points', xytext=(8, 6), fontsize=7.5, color=SUB)
ax.plot(MB, OMP, '-s', color=AQUA, linewidth=2, markersize=6,
        label='dict, MSE (OMP/LS encoder)', zorder=4)
ax.plot(MB, CTX, '-^', color=INK, linewidth=2, markersize=7,
        label='dict, OV-context-TRAINED', zorder=5)
for (n, k) in ANCHORS:
    mb = s0(n, k)['Mbits']
    for field, col in (('dce_lin', AQUA), ('dce_ctx', INK)):
        vals = seeds(n, k, field)
        ax.errorbar([mb], [sum(vals) / 3], yerr=[[sum(vals) / 3 - min(vals)],
                    [max(vals) - sum(vals) / 3]], color=col, capsize=3,
                    linewidth=1.2, zorder=6, fmt='none')
ax.annotate('OV-trained at 2.5% bits ≈\nMSE-linear at 6% bits', (MB[0], CTX[0]),
            textcoords='offset points', xytext=(-4, 16), ha='left', fontsize=8.5, color=INK)
ax.annotate('crossover: rich budgets\nfavor MSE + OMP', (MB[-1], OMP[-1]),
            textcoords='offset points', xytext=(6, 14), ha='left', fontsize=8.5, color=SUB)
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('held-out ΔCE, FineWeb (nats)', color=SUB, fontsize=10)
ax.set_title('A — dictionary Pareto sweep: objective × budget (seed min–max bars at anchors)',
             color=INK, fontsize=11, loc='left')
ax.legend(fontsize=8, framealpha=0, labelcolor=SUB, loc='upper right')

# B — paired improvement of OV-context over MSE encoders
ax = axB
ax.axhline(0, color='#b9b8b3', linewidth=1, linestyle='--', zorder=1)
imp_lin = [l - c for l, c, g_ in zip(LIN, CTX, DEGEN) if not g_]
mb_lin = [m_ for m_, g_ in zip(MB, DEGEN) if not g_]
imp_omp = [o - c for o, c in zip(OMP, CTX)]
ax.plot(mb_lin, imp_lin, '-o', color=AQUA, linewidth=2, markersize=6,
        label='vs MSE linear (same encoder)', zorder=3)
ax.plot(MB, imp_omp, '-s', color=BLUE, linewidth=2, markersize=6,
        label='vs MSE + OMP/LS', zorder=3)
for (n, k) in ANCHORS:
    mb = s0(n, k)['Mbits']
    diffs = [SW[f'n{n}_k{k}_s{s}']['dce_lin'] - SW[f'n{n}_k{k}_s{s}']['dce_ctx'] for s in (0, 1, 2)]
    mean = sum(diffs) / 3
    ax.errorbar([mb], [mean], yerr=[[mean - min(diffs)], [max(diffs) - mean]],
                color=AQUA, capsize=3, linewidth=1.2, zorder=5, fmt='none')
ax.text(0.03, 0.92, 'above zero = OV-context training wins', transform=ax.transAxes,
        fontsize=8.5, color=SUB)
ax.set_xscale('log')
ax.set_xlabel('description length (megabits, log scale)', color=SUB, fontsize=10)
ax.set_ylabel('ΔCE improvement from OV-context training (nats)', color=SUB, fontsize=10)
ax.set_title('B — where the OV objective helps (paired, per-seed bars)', color=INK,
             fontsize=11, loc='left')
ax.legend(fontsize=8, framealpha=0, labelcolor=SUB, loc='upper right')

fig.suptitle('Layer-0 QK dictionaries: in-depth Pareto sweep (tick 160) — FineWeb, 307k held-out '
             'predictions, 8 budgets × 2 objectives × 3 seeds at anchors',
             color=INK, fontsize=12, x=0.01, ha='left')
fig.text(0.01, -0.035,
         'OV-context objective = eq. † of ov_metric_explainer.md (scatter charged at T, systematic '
         'component at T², unigram-weighted), initialized from the MSE fit, identical bits. '
         'Low budgets: OV-training dominates (up to 2× at 2.5–3% raw). High budgets: MSE+OMP wins; '
         'the OV metric\'s unigram/i.i.d. approximation floor (~+0.005) binds.',
         fontsize=8, color=SUB, ha='left')
fig.savefig(f'{QK}/fig_qk_pareto.png', dpi=160, bbox_inches='tight', facecolor=SURF)
print('wrote fig_qk_pareto.png')
