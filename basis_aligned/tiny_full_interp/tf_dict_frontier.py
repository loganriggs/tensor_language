"""Frontier extraction + figure for the ported dictionary method.

Panel A -- THE METHOD TEST, on the parent program's own axis: held cross-entropy
against the bits spent describing the exact folded object.  This is where
"does the ported objective beat MSE, does the dictionary beat low rank, do
anchors port" is settled.

Panel B -- THE DESCRIPTION TEST, on Logan's axis: held cross-entropy against the
TOTAL description length, with the model's own length and cross-entropy marked
and the quantisation line drawn as a clearly labelled REFERENCE (recoding, not
explanation).

Colour is assigned by family in a fixed order from a colourblind-safe set
(Okabe-Ito); no family is ever recoloured by rank, no dual axes, one legend per
panel, grid recessive.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

OI = {'blue': '#0072B2', 'vermillion': '#D55E00', 'green': '#009E73',
      'orange': '#E69F00', 'purple': '#CC79A7', 'sky': '#56B4E9',
      'yellow': '#B8A400', 'black': '#222222', 'grey': '#8C8C8C'}


def pareto(points, xkey='bits', ykey='ce'):
    """Lower-left frontier."""
    pts = sorted(points, key=lambda p: p[xkey])
    best, out = float('inf'), []
    for p in pts:
        if p[ykey] < best - 1e-12:
            best = p[ykey]
            out.append(p)
    return out


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else 'tf_vanilla_d1_w128_b8192_s0'
    A = json.load(open(f'{HERE}/{stem}_dict_fold.json'))
    B = json.load(open(f'{HERE}/{stem}_dict_emb.json')) \
        if os.path.exists(f'{HERE}/{stem}_dict_emb.json') else {}
    OLD = json.load(open(f'{HERE}/{stem}_compress_frontier.json')) \
        if os.path.exists(f'{HERE}/{stem}_compress_frontier.json') else {}

    ce_model = A['controls']['fold_identity']['ce']
    bits_model = A['controls']['model_bits']
    raw_fold = A['controls']['raw_fold_bits']

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # ------------------------------------------------------------- panel A
    a = ax[0]
    series = [
        ('dictionary, joint, MSE', OI['grey'], 'o', '--',
         [r for r in A['sweep'] if r['mode'] == 'joint' and r['obj'] == 'mse']),
        ('dictionary, joint, context-OV objective', OI['blue'], 'o', '-',
         [r for r in A['sweep'] if r['mode'] == 'joint' and r['obj'] == 'ctx']),
        ('dictionary, per-head-branch, MSE', OI['orange'], 's', '--',
         [r for r in A['sweep'] if r['mode'] == 'perhb' and r['obj'] == 'mse']),
        ('dictionary, per-head-branch, context-OV', OI['vermillion'], 's', '-',
         [r for r in A['sweep'] if r['mode'] == 'perhb' and r['obj'] == 'ctx']),
        ('low rank (SVD) of the same object', OI['purple'], '^', '-',
         A['lowrank']),
    ]
    for name, c, m, ls, rows in series:
        rows = sorted(rows, key=lambda r: r['fold_bits'])
        a.plot([r['fold_bits'] / 1e6 for r in rows], [r['ce'] for r in rows],
               marker=m, ls=ls, color=c, lw=2, ms=6, label=name)
    anc = [r for r in A.get('anchors', []) if r['attr'] != 'random']
    if anc:
        fr = pareto([{'bits': r['fold_bits'], 'ce': r['ce']} for r in anc])
        a.plot([p['bits'] / 1e6 for p in fr], [p['ce'] for p in fr],
               marker='D', ls='-', color=OI['green'], lw=2.4, ms=7,
               label='anchor hybrid (exact rows + dictionary tail)')
    rnd = [r for r in A.get('anchors', []) if r['attr'] == 'random'] + \
        A.get('nulls', [])
    if rnd:
        a.scatter([r['fold_bits'] / 1e6 for r in rnd], [r['ce'] for r in rnd],
                  marker='x', color=OI['black'], s=42, zorder=5,
                  label='nulls (random anchors / random dictionary)')
    a.axhline(ce_model, color=OI['black'], lw=1.4, ls=':')
    a.annotate(f'the model itself, CE {ce_model:.4f}',
               (a.get_xlim()[0], ce_model), xytext=(3, 4),
               textcoords='offset points', fontsize=9, color=OI['black'])
    a.axvline(raw_fold / 1e6, color=OI['grey'], lw=1, ls=':')
    a.annotate(f'exact fold, {raw_fold / 1e6:.0f} Mbit', (raw_fold / 1e6, 5.0),
               rotation=90, fontsize=8, color=OI['grey'], ha='right')
    a.set_xscale('log')
    a.set_xlabel('bits spent describing the exact folded object (Mbit, log)')
    a.set_ylabel('held cross-entropy (nats/token)')
    a.set_title('A. The method, on the parent program\'s object\n'
                'depth-1 width-128 seed-0, held = 256 seqs x 256 tokens',
                fontsize=11, loc='left')
    a.grid(alpha=0.25, lw=0.6)
    a.legend(fontsize=8.5, framealpha=0.9)

    # ------------------------------------------------------------- panel B
    # y is CE ABOVE THE MODEL, because the FINDING-12 frontier was scored on a
    # 64-sequence held set (model CE 4.7114) and everything here on a
    # 256-sequence one (model CE 4.7556); raw cross-entropies from the two are
    # not comparable, differences from the model on the same set are.
    b = ax[1]
    CE_OLD_MODEL = 4.711396515369415
    if OLD:
        old = sorted(OLD['all_points'], key=lambda p: p['bits'])
        fr = pareto([{'bits': p['bits'], 'ce': p['ce'] - CE_OLD_MODEL}
                     for p in old])
        b.plot([p['bits'] / 1e6 for p in fr], [p['ce'] for p in fr],
               color=OI['grey'], lw=2.4, ls='--', marker='.',
               label='quantisation / recoding frontier (FINDING 12)\n'
                     'REFERENCE ONLY -- recoding, not explanation')
    fa = pareto([{'bits': r['bits_total'], 'ce': r['dce_vs_model']}
                 for r in A['sweep'] + A['lowrank'] + A.get('anchors', [])])
    b.plot([p['bits'] / 1e6 for p in fa], [p['ce'] for p in fa],
           color=OI['blue'], lw=2, marker='o', ms=5,
           label='A: description stores the FOLD (dictionary/anchors)')
    if B:
        rows = B.get('sweep', []) + B.get('anchors', [])
        fb = pareto([{'bits': r['bits_total'], 'ce': r['dce_vs_model']}
                     for r in rows])
        b.plot([p['bits'] / 1e6 for p in fb], [p['ce'] for p in fb],
               color=OI['vermillion'], lw=2.4, marker='D', ms=6,
               label='B: sparse token dictionary in the fold\'s metric')
        ref = B.get('reference', [])
        if ref:
            fr2 = pareto([{'bits': r['bits_total'], 'ce': r['dce_vs_model']}
                          for r in ref])
            b.plot([p['bits'] / 1e6 for p in fr2], [p['ce'] for p in fr2],
                   color=OI['grey'], lw=1.6, ls=':', marker='.',
                   label='same harness: quantise / low-rank the embedding')
        rf = B.get('ce_refit', [])
        if rf:
            b.scatter([r['bits_total'] / 1e6 for r in rf],
                      [r['dce_vs_model'] for r in rf], marker='*', s=170,
                      color=OI['green'], zorder=6,
                      label='B + coefficients refit on est CE (same bits)')
    b.scatter([bits_model / 1e6], [0.0], marker='*', s=260,
              color=OI['black'], zorder=7, label='the model itself')
    b.axhline(0.0, color=OI['black'], lw=1.2, ls=':')
    b.set_yscale('symlog', linthresh=0.01)
    b.set_xscale('log')
    b.set_xlabel('TOTAL description length (Mbit, log) -- every table charged')
    b.set_ylabel('held cross-entropy ABOVE the model (nats/token, symlog)')
    b.set_title('B. The description, on Logan\'s axis\n'
                'cross-entropy against the DATA; 0 = the model, and the '
                'model\'s whole\nadvantage over the unigram floor is 2.573 '
                'nats', fontsize=11, loc='left')
    b.grid(alpha=0.25, lw=0.6)
    b.legend(fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out = f'{HERE}/fig_tf_dict_frontier.png'
    fig.savefig(out, dpi=155)
    print('wrote', out)

    # ------------------------------------------------------------- tables
    def tab(rows, keys, title):
        print('\n### ' + title)
        print('| ' + ' | '.join(keys) + ' |')
        print('|' + '---|' * len(keys))
        for r in rows:
            print('| ' + ' | '.join(
                (f'{r[k]:.4f}' if isinstance(r.get(k), float) else
                 str(r.get(k, ''))) for k in keys) + ' |')

    # ------------------------------------------------------------ headline
    import math

    def bits_at(rows, ce_t, bk='fold_bits'):
        """Linear-in-log-bits interpolation of the bits needed to reach a CE."""
        pts = sorted([(r[bk], r['ce']) for r in rows])
        for (b0, c0), (b1, c1) in zip(pts[:-1], pts[1:]):
            if (c0 - ce_t) * (c1 - ce_t) <= 0 and c0 != c1:
                f = (c0 - ce_t) / (c0 - c1)
                return math.exp(math.log(b0) + f * (math.log(b1) - math.log(b0)))
        return None

    dj = [r for r in A['sweep'] if r['mode'] == 'joint']
    hp = {'ce_model': A['controls']['fold_identity']['ce']}
    for tgt in (4.85, 4.80, 4.78, 4.76):
        hp[f'bits_at_ce_{tgt}'] = {
            'dict_joint_mse': bits_at([r for r in dj if r['obj'] == 'mse'], tgt),
            'dict_joint_ctx': bits_at([r for r in dj if r['obj'] == 'ctx'], tgt),
            'dict_perhb_ctx': bits_at([r for r in A['sweep']
                                       if r['mode'] == 'perhb'
                                       and r['obj'] == 'ctx'], tgt),
            'svd': bits_at(A['lowrank'], tgt),
            'anchor_best': bits_at(pareto([{'fold_bits': r['fold_bits'],
                                            'ce': r['ce']}
                                           for r in A.get('anchors', [])
                                           if r['attr'] != 'random'],
                                          xkey='fold_bits'), tgt),
            'anchor_random_null': bits_at([r for r in A.get('anchors', [])
                                           if r['attr'] == 'random'], tgt)}
    hp['ctx_minus_mse_at_matched_bits'] = [
        {'mode': m, 'n': r['n'], 'k': r['k'],
         'd_ce': r['ce'] - o['ce'], 'd_kl': r['kl'] - o['kl']}
        for m in ('joint', 'perhb')
        for r in A['sweep'] if r['mode'] == m and r['obj'] == 'ctx'
        for o in A['sweep'] if o['mode'] == m and o['obj'] == 'mse'
        and o['n'] == r['n'] and o['k'] == r['k']]
    print('\n### HEADLINE\n' + json.dumps(hp, indent=1))
    json.dump(hp, open(f'{HERE}/{stem}_dict_headline.json', 'w'), indent=1)

    tab(sorted(A['sweep'], key=lambda r: r['fold_bits']),
        ['mode', 'obj', 'n', 'k', 'fold_bits', 'pct_raw', 'ce', 'dce_vs_model',
         'kl', 'fvu'], 'Description A -- dictionaries on the fold')
    tab(A['lowrank'], ['rank', 'fold_bits', 'pct_raw', 'ce', 'kl', 'fvu'],
        'Description A -- low-rank reference')
    tab(sorted(A.get('anchors', []), key=lambda r: r['fold_bits']),
        ['attr', 'mode', 'B', 'n', 'k', 'fold_bits', 'ce', 'kl'],
        'Description A -- anchor hybrid')
    if B:
        tab(sorted(B.get('sweep', []), key=lambda r: r['bits_emb']),
            ['obj', 'n', 'k', 'bits_emb', 'bits_total', 'ce', 'dce_vs_model',
             'kl'], 'Description B -- dictionaries on the embedding')
        tab(sorted(B.get('reference', []), key=lambda r: r['bits_emb']),
            ['family', 'b', 'rank', 'bits_per_row', 'bits_emb', 'bits_total',
             'ce', 'kl'], 'Description B -- recoding reference')
        tab(sorted(B.get('anchors', []), key=lambda r: r['bits_emb']),
            ['attr', 'B', 'n', 'k', 'bits_emb', 'bits_total', 'ce', 'kl'],
            'Description B -- anchor hybrid')


if __name__ == '__main__':
    main()
