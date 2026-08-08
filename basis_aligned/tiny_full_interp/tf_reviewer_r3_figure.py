"""The reviewer's replacement figure for FINDING 12.

Two panels, because the review changed two different things:

  LEFT   the frontier SPLIT BY CLASS -- descriptions that merely recode the
         model's own weights vs descriptions that assert structure -- with the
         four honest denominators marked instead of the single fp32 star.
  RIGHT  the compression factor itself, against each denominator, so
         "5.7x smaller than the model" can be read next to the number that
         survives an honest baseline.

Palette: categorical slots 1-3 of the validated default palette (light mode),
the same instance `tf_compress_frontier.py` uses; identity is carried
redundantly by marker shape as well as hue, and every series is direct-labelled
as well as legended.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAL = ['#2a78d6', '#eb6834', '#1baf7a']
MARK = ['o', 's', '^']
INK, INK2, MUTED = '#2b2b2b', '#5c5c5c', '#8f8f8f'
GRID = '#e6e6e6'


def main():
    rev = json.load(open(f'{HERE}/tf_reviewer_round_3_compression.json'))
    O7 = rev['O7_recoding_vs_structure_and_CE']
    O1 = rev['O1_fair_denominator']
    fp32 = O7['fp32_bits']
    cls = O7['classes']
    names = [('a_recoding_of_the_weights',
              'recodes the model’s own weights'),
             ('b_hybrid_structure_plus_coded_residual',
              'structure + coded residual'),
             ('b_pure_structure', 'pure structure (dictionary, factor, rule)')]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.6, 6.2), dpi=170,
        gridspec_kw={'width_ratios': [1.55, 1.0]})
    for a in (ax, bx):
        a.set_facecolor('white')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            a.spines[sp].set_color('#bdbdbd')

    # ------------------------------------------------------------- LEFT
    FLOOR = 1.5e-5      # the honest floor: fp16 reference storage, see O2
    for i, (key, lab) in enumerate(names):
        fr = cls[key]['pareto']
        xs = [p['bits'] / 1e6 for p in fr]
        ys = [max(p['kl'], FLOOR) for p in fr]
        ax.plot(xs, ys, color=PAL[i], lw=2.0, alpha=0.9, zorder=3)
        ax.scatter(xs, ys, s=52, marker=MARK[i], facecolor=PAL[i],
                   edgecolor='white', linewidth=1.2, zorder=4, label=lab)
    # the honest denominators
    dens = [('fp32, the number the finding quoted', fp32),
            ('best lossless recompression', O1['lossless_best']),
            ('fp16 (KL at the floor)', 16 * O1['n_params']),
            ('12-bit uniform (KL below the floor)', 16449536)]
    for lab, b in dens:
        ax.axvline(b / 1e6, color=MUTED, ls=':', lw=1.1, zorder=1)
        ax.text(b / 1e6, 6.5, lab, rotation=90, ha='right', va='top',
                fontsize=7.6, color=INK2)
    # the matched-KL naive baseline curve
    ent = sorted((v['bits'], v['kl']) for k, v in O1['standard_encodings'].items()
                 if k.endswith('_perrow_entropy'))
    ax.plot([b / 1e6 for b, k in ent], [k for b, k in ent], color=INK,
            lw=1.6, ls='--', zorder=2,
            label='naive per-row quantisation + entropy coding')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(FLOOR * 0.6, 9)
    ax.set_xlim(0.4, 60)
    ax.set_xlabel('description length (megabits — everything charged)')
    ax.set_ylabel('KL from the model on held text (nats/token)')
    ax.set_title('Every structural class is at or behind plain recoding',
                 fontsize=11.5, color=INK)
    ax.grid(True, which='major', color=GRID, lw=0.8, zorder=0)
    ax.legend(fontsize=8.4, loc='lower left', frameon=False)

    # ------------------------------------------------------------- RIGHT
    comp = [c for c in O1['frontier_vs_honest_baselines']
            if c['scheme'] == 'embT768+body8'][0]
    labs = ['vs fp32\n(the quoted number)', 'vs best lossless\nrecompression',
            'vs fp16\n(same behaviour)', 'vs 12-bit uniform\n(same behaviour)',
            'vs naive quantisation\nAT MATCHED KL']
    vals = [comp['x_vs_fp32'],
            O1['lossless_best'] / comp['bits'],
            comp['x_vs_bf16'],
            16449536 / comp['bits'],
            comp['x_vs_entropy_ptq']]
    cols = [MUTED, MUTED, MUTED, MUTED, PAL[1]]
    y = np.arange(len(labs))[::-1]
    bx.barh(y, vals, height=0.52, color=cols, edgecolor='white', linewidth=2)
    for yy, v in zip(y, vals):
        bx.text(v + 0.12, yy, f'{v:.2f}×', va='center', ha='left',
                fontsize=10.5, color=INK)
    bx.axvline(1.0, color=INK, lw=1.2)
    bx.text(1.04, len(labs) - 0.45, 'no better than the baseline', fontsize=8,
            color=INK2, va='top')
    bx.set_yticks(y)
    bx.set_yticklabels(labs, fontsize=8.6, color=INK)
    bx.set_xlim(0, 6.6)
    bx.set_xlabel('how much smaller the 7.59 Mbit description is')
    bx.set_title('The compression factor depends entirely on the denominator',
                 fontsize=11.5, color=INK)
    bx.grid(True, axis='x', color=GRID, lw=0.8, zorder=0)
    bx.set_axisbelow(True)

    fig.suptitle('Reviewer round 3 — rung-5 compression frontier, '
                 'depth 1, width 128, V = 8192, 1.34M parameters',
                 fontsize=12.5, color=INK, y=1.005)
    fig.tight_layout()
    p = f'{HERE}/fig_tf_compression_frontier_review.png'
    fig.savefig(p, bbox_inches='tight')
    print('wrote', p)


if __name__ == '__main__':
    main()
