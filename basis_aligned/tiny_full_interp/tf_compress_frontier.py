"""Assemble every measured description onto one (bits, KL) plane, compute the
Pareto frontier, and render the figure.  Reads the JSON written by
`tf_compress_run.py`; adds no measurement of its own, so the figure and the
RESULTS table can never disagree.

The table view in RESULTS.md is the accessible companion to the figure (the
palette's relief rule): every plotted point is also a table row with its bits,
its KL and its itemised bill.
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# categorical slots 1-8 of the validated default palette, light mode.  Identity
# is carried REDUNDANTLY by marker shape as well as hue, because a scatter puts
# all pairs on screen at once and only the first three slots clear the
# all-pairs colour-vision floors.
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
       '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# Eight series is the cap: the palette has eight categorical slots and a ninth
# is never a generated hue, so low rank and product quantisation -- both
# linear-algebraic recodings of the embedding -- share one slot.
FAMILIES = [
    ('model quantised uniformly', [('A_self_quantisation', 'uniform_')]),
    ('embedding: token clustering', [('C_embedding', 'cluster_')]),
    ('embedding: low rank / product quantisation',
     [('C_embedding', 'lowrank_'), ('C_embedding', 'pq_')]),
    ('embedding: anchor rows + tail', [('D_anchor', 'anchor')]),
    ('best scalar codes x quantised body',
     [('F_combined', ''), ('G_codes', '')]),
    ('conditional on spelling / corpus statistics',
     [('M_conditional_combined', ''), ('K_features', ''),
      ('L_corpus_stats', '')]),
    ('distilled (tables re-fit on est)', [('I_distilled', '')]),
    ('weights-free tables', [('H_weightsfree', '')]),
]


def collect(d):
    pts = []
    for fi, (name, srcs) in enumerate(FAMILIES):
      for sec, pref in srcs:
        for r in d.get(sec, {}).get('rows', []):
            if not r['scheme'].startswith(pref):
                continue
            if r.get('kl') is None:
                continue
            b = r.get('bits')
            if b is None:
                b = r.get('bits_total_with_fp32_body')
            if b is None:
                continue
            pts.append({'family': name, 'fi': fi, 'scheme': r['scheme'],
                        'bits': float(b), 'kl': float(r['kl']),
                        'ce': r.get('ce')})
    return pts


def pareto(pts):
    """Lower bits and lower KL both better."""
    out = []
    for p in sorted(pts, key=lambda q: (q['bits'], q['kl'])):
        if not out or p['kl'] < out[-1]['kl'] - 1e-12:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=f'{HERE}/tf_vanilla_d1_w128_b8192_s0'
                                      '_compress.json')
    ap.add_argument('--png', default=f'{HERE}/fig_tf_compression_frontier.png')
    a = ap.parse_args()
    d = json.load(open(a.json))
    pts = collect(d)
    fp32 = d['model']['fp32_bits']
    fr = pareto([p for p in pts if p['family'] != 'weights-free tables'])
    print(f'model fp32 = {fp32/1e6:.3f} Mbit at KL 0')
    print('PARETO FRONTIER (bits, KL, scheme):')
    for p in fr:
        print(f'  {p["bits"]/1e6:8.3f} Mbit  KL {p["kl"]:.5f}  '
              f'{fp32/p["bits"]:5.1f}x  {p["scheme"]}')
    json.dump({'fp32_bits': fp32, 'frontier': fr, 'all_points': pts},
              open(a.json.replace('.json', '_frontier.json'), 'w'), indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FLOOR = float(d['positive_control']['kl_identity'])      # 1.5e-6
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=170)
    ax.set_facecolor('white')

    def yv(k):
        return max(k, FLOOR)

    for fi, (name, _) in enumerate(FAMILIES):
        g = [p for p in pts if p['fi'] == fi]
        if not g:
            continue
        ax.scatter([p['bits'] / 1e6 for p in g], [yv(p['kl']) for p in g],
                   s=46, marker=MARK[fi % 8], facecolor=PAL[fi % 8],
                   edgecolor='white', linewidth=0.8, label=name, zorder=3)
    ax.plot([p['bits'] / 1e6 for p in fr], [yv(p['kl']) for p in fr],
            color='#3f3f3f', lw=2.0, alpha=0.85, zorder=2,
            label='Pareto frontier')
    ax.axhline(FLOOR, color='#9a9a9a', ls='--', lw=1.0, zorder=1)
    ax.text(0.985, FLOOR * 1.35, 'measurement floor (fp32 round-off)',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            fontsize=8.5, color='#6b6b6b')
    ax.scatter([fp32 / 1e6], [yv(0.0)], s=150, marker='*', color='#3f3f3f',
               zorder=5)
    ax.annotate('THE MODEL ITSELF\n%.1f Mbit (fp32), KL 0' % (fp32 / 1e6),
                xy=(fp32 / 1e6, yv(0.0)), xytext=(fp32 / 1e6 * 1.15, 8e-5),
                ha='left', fontsize=9.5, color='#3f3f3f',
                arrowprops=dict(arrowstyle='->', color='#3f3f3f', lw=1.2))
    # direct labels on the points the write-up quotes (never on every point)
    quoted = {}
    for p in fr:
        for tag, lim in (('a', 0.05), ('b', 0.005)):
            if p['kl'] <= lim and tag not in quoted:
                quoted[tag] = p
    OFF = {'a': (1.12, 0.20, 'left'), 'b': (1.55, 5.5, 'left')}
    for tag, p in quoted.items():
        fx, fy, ha = OFF[tag]
        ax.annotate(f"{p['bits']/1e6:.2f} Mbit, KL {p['kl']:.3f}\n"
                    f"{fp32/p['bits']:.1f}x smaller than the model",
                    xy=(p['bits'] / 1e6, yv(p['kl'])),
                    xytext=(p['bits'] / 1e6 * fx, yv(p['kl']) * fy),
                    fontsize=8.8, color='#3f3f3f', ha=ha,
                    arrowprops=dict(arrowstyle='-', color='#8a8a8a', lw=0.9))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(FLOOR * 0.55, 12)
    ax.set_xlabel('description length (megabits — tables, codebooks, indices, '
                  'scales, all charged)')
    ax.set_ylabel('KL from the true model on held text (nats/token)')
    ax.set_title('Rung 5 compression frontier — depth 1, width 128, seed 0, '
                 'V = 8192, 1.34M parameters', fontsize=11.5)
    ax.grid(True, which='major', color='#e6e6e6', lw=0.8, zorder=0)
    ax.grid(True, which='minor', color='#f4f4f4', lw=0.5, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#bdbdbd')
    ax.legend(fontsize=8.8, framealpha=1.0, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(a.png, bbox_inches='tight')
    print('wrote', a.png)


if __name__ == '__main__':
    main()
