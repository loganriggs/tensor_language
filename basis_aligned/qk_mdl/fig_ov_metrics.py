"""Figure (tick 158): the OV metric story — ladder ranking, cancellation index, and the fix.

  A. Spearman (vs FineWeb delta-CE) per metric rung; OV-containing rungs highlighted.
  B. Cancellation index per arm vs the orthogonal floor (1) and the signal's own coherence (31.6).
  C. Metric-vs-truth scatter: OV-Gram (non-monotone: flatters SVD) vs context-expected (monotone).
Reads qk_ovweight.json. Writes fig_ov_metrics.png. Companion: ov_metric_explainer.md.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
J = json.load(open(f'{QK}/qk_ovweight.json'))
ARMS = J['arms']
SP = J['spearman_vs_dce_fw']

INK, SUB, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e4', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

RUNG_LABEL = {'fac': 'factor FVU (plain)', 'pat_freq': 'pattern, frequency-weighted',
              'pat_ctx': 'context-expected OV  (cancellation split)', 'score': 'score FVU',
              'pat_rope': 'pattern + rotary', 'pat': 'pattern (s₁·s₂)',
              'pat_rope_ov': 'pattern + rotary + OV norm', 'pat_ov': 'OV norm-weighted',
              'pat_gram': 'OV-Gram (full cancellation)', 'pat_rope_gram': 'OV-Gram + rotary'}
OV_RUNGS = {'pat_ov', 'pat_rope_ov', 'pat_gram', 'pat_rope_gram', 'pat_ctx'}

FAM = {}                                                    # arm -> (family color, marker, label)
for a in ARMS:
    if a.startswith('svd'):
        FAM[a] = (BLUE, 'o', 'SVD')
    elif a.startswith('merge'):
        FAM[a] = (ORANGE, 's', 'merge')
    elif a.startswith('dict'):
        FAM[a] = (AQUA, 'D', 'dictionary')
    else:
        FAM[a] = (INK, '*', 'two-stage')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURF, constrained_layout=True)
for ax in axes:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d3')
    ax.tick_params(colors=SUB, labelsize=9)

# A — ladder ranking
ax = axes[0]
order = sorted(SP, key=lambda k: SP[k])
ys = range(len(order))
cols = [AQUA if k in OV_RUNGS else BLUE for k in order]
ax.barh(list(ys), [SP[k] for k in order], color=cols, height=0.62)
ax.set_yticks(list(ys))
ax.set_yticklabels([RUNG_LABEL[k] for k in order], fontsize=8.5, color=INK)
for y, k in zip(ys, order):
    ax.text(SP[k] + 0.012, y, f'{SP[k]:.2f}', va='center', fontsize=8.5, color=SUB)
ax.set_xlim(0, 1.06)
ax.set_xlabel('Spearman rank correlation with held-out ΔCE (FineWeb)', color=SUB, fontsize=10)
ax.set_title('A — which weight-side metric predicts behavior', color=INK, fontsize=11, loc='left')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=AQUA, label='folds in OV'), Patch(color=BLUE, label='no OV')],
          fontsize=8, framealpha=0, labelcolor=SUB, loc='lower right')

# B — cancellation index per arm
ax = axes[1]
arms_sorted = sorted(ARMS, key=lambda a: ARMS[a]['diag_cancel_err'])
ys = range(len(arms_sorted))
ax.barh(list(ys), [ARMS[a]['diag_cancel_err'] for a in arms_sorted],
        color=[FAM[a][0] for a in arms_sorted], height=0.62)
ax.set_yticks(list(ys))
ax.set_yticklabels([a.replace(' per-head-branch', '').replace(' token-', ' ')
                    .replace('merge2048 -> OMP dict n=512 k=8', '') for a in arms_sorted],
                   fontsize=8, color=INK)
sig = ARMS[arms_sorted[0]]['diag_cancel_sig']
ax.axvline(1, color='#b9b8b3', linewidth=1, linestyle='--')
ax.axvline(sig, color=INK, linewidth=1.2, linestyle='--')
ax.text(1.4, 0.15, 'orthogonal (=1)', fontsize=8, color=SUB, rotation=90, va='bottom')
ax.text(sig - 0.6, 0.0, f'signal itself ({sig:.1f})', fontsize=8, color=INK,
        rotation=90, va='bottom', ha='right')
ax.set_xlabel('cancellation index of the arm\'s error through OV\n'
              '(higher = errors reinforce; SVD lowest = biggest Gram discount)',
              color=SUB, fontsize=9)
ax.set_title('B — how much each arm\'s error self-cancels through OV', color=INK,
             fontsize=11, loc='left')

# C — the fix: gram (non-monotone) vs context-expected (monotone)
ax = axes[2]
gmax = max(ARMS[a]['pat_gram'] for a in ARMS)
cmax = max(ARMS[a]['pat_ctx'] for a in ARMS)
seen = set()
for a in ARMS:
    col, mk, fam = FAM[a]
    d = ARMS[a]['dce_fw']
    ax.scatter([d], [ARMS[a]['pat_gram'] / gmax], marker=mk, s=55, facecolors=SURF,
               edgecolors=ORANGE, linewidths=1.6, zorder=3)
    ax.scatter([d], [ARMS[a]['pat_ctx'] / cmax], marker=mk, s=55, color=BLUE,
               edgecolors=SURF, linewidths=1.0, zorder=4)
ax.scatter([], [], marker='o', facecolors=SURF, edgecolors=ORANGE, linewidths=1.6,
           label='OV-Gram (Spearman 0.57)')
ax.scatter([], [], marker='o', color=BLUE, label='context-expected (Spearman 0.90)')
ax.set_xlabel('held-out ΔCE, FineWeb (nats)', color=SUB, fontsize=10)
ax.set_ylabel('metric value (normalized to its max across arms)', color=SUB, fontsize=10)
ax.set_title('C — splitting cancellation restores monotonicity', color=INK, fontsize=11, loc='left')
ax.legend(fontsize=8, framealpha=0, labelcolor=SUB, loc='upper left')
ax.text(0.03, 0.60, 'marker shape = family\n(○ SVD  □ merge  ◇ dict  ★ two-stage)\n'
        'hollow orange Gram points: SVD sits low-left\nof dictionaries it actually loses to',
        transform=ax.transAxes, fontsize=8, color=SUB)

fig.suptitle('Folding the output-value circuit into the QK metric: what failed and the fix '
             '(companion to ov_metric_explainer.md)', color=INK, fontsize=12.5, x=0.01, ha='left')
fig.text(0.01, -0.03,
         'All quantities weight-only (+ unigram frequencies for the context-expected rung). '
         'Cancellation index = ‖ΔP·U‖² / Σ ΔP²‖u‖²; context-expected metric charges scatter at T '
         'and the systematic component at T² (T=512).', fontsize=8, color=SUB, ha='left')
fig.savefig(f'{QK}/fig_ov_metrics.png', dpi=160, bbox_inches='tight', facecolor=SURF)
print('wrote fig_ov_metrics.png')
