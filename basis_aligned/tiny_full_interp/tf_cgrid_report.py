"""Aggregate the per-cell compressibility scalars into the curve that answers
Logan's question: does the ratio GROW with model size (our negative is a
small-model artifact and the programme should scale up) or is it FLAT/SHRINKING
(the negative is a property of this architecture family)?

Emits `tf_cgrid_table.md`, `tf_cgrid_summary.json` and
`fig_tf_compressibility_vs_size.png`.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ordinal blue ramp, depth 1..4 (dataviz reference palette, ordinal steps: the
# lightest must clear 2:1 on the light surface, so start at step 250)
DEPTH_COLOR = {1: '#86b6ef', 2: '#3987e5', 3: '#256abf', 4: '#104281'}
SURFACE = '#fcfcfb'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8c8b85'


def load():
    rows = []
    for f in sorted(glob.glob(f'{HERE}/tf_vanilla_d*_w*_b8192_s*_cgrid.json')):
        d = json.load(open(f))
        s = d['summary']
        rows.append({
            'stem': d['stem'], 'depth': d['depth'], 'width': d['width'],
            'seed': int(d['stem'].split('_s')[-1]),
            'n_params': d['n_params'], 'n_body': d['n_body'],
            'emb_share': d['embedding_share_of_params'],
            'held_ce': d['model']['held_ce'],
            'headroom': d['headroom_over_unigram_nats'],
            'ratio': s['ratio_median'], 'ratio_min': s['ratio_min'],
            'ratio_max': s['ratio_max'],
            'ratio_strong': s['ratio_strong_median'],
            'ratio_struct': s['ratio_structure_only_median'],
            'ratio_struct_strong': s['ratio_structure_only_strong_median'],
            'levels_structure_reaches': s['levels_structure_can_reach'],
            'n_levels': s['n_levels'],
            'naive_overhead_share_b4':
                d['naive_overhead']['emb_b4']['scale_share'],
            'rows': d['matched_score_ratios']})
    return rows


def slope(x, y):
    """OLS slope of y on log(x), with its standard error and a two-sided t."""
    lx = np.log(np.asarray(x, float))
    y = np.asarray(y, float)
    n = len(lx)
    if n < 3:
        return None
    b, a = np.polyfit(lx, y, 1)
    yh = a + b * lx
    s2 = ((y - yh) ** 2).sum() / (n - 2)
    se = math.sqrt(s2 / ((lx - lx.mean()) ** 2).sum())
    return {'slope_per_efold': float(b), 'se': float(se),
            'intercept': float(a), 'n': n,
            't': float(b / se) if se > 0 else None}


def main():
    rows = load()
    if not rows:
        print('no cgrid JSONs yet')
        return
    s0 = [r for r in rows if r['seed'] == 0]
    out = {'n_cells': len(rows), 'n_cells_seed0': len(s0)}

    for key in ('ratio', 'ratio_strong', 'ratio_struct', 'ratio_struct_strong'):
        v = [(r['n_params'], r[key]) for r in s0 if r[key]]
        if len(v) >= 3:
            out[f'trend_{key}_vs_params'] = slope([a for a, _ in v],
                                                  [b for _, b in v])
        # depth held fixed at 1-2 (the width axis alone)
        vw = [(r['n_params'], r[key]) for r in s0
              if r[key] and r['depth'] in (1, 2)]
        if len(vw) >= 3:
            out[f'trend_{key}_vs_params_depth12'] = slope(
                [a for a, _ in vw], [b for _, b in vw])
    # depth effect at fixed width
    de = {}
    for w in (64, 128, 256):
        pts = sorted([(r['depth'], r['ratio']) for r in s0
                      if r['width'] == w and r['ratio']])
        if len(pts) >= 2:
            de[f'w{w}'] = {'by_depth': pts,
                           'max_minus_min': max(p[1] for p in pts)
                           - min(p[1] for p in pts)}
    out['depth_effect_at_fixed_width'] = de
    # seed spread, for the "is a trend bigger than the noise" question
    sp = {}
    for key in ('ratio', 'ratio_strong', 'ratio_struct'):
        g = {}
        for r in rows:
            g.setdefault((r['depth'], r['width']), []).append(r[key])
        d = [float(np.std(v, ddof=1)) for v in g.values()
             if len(v) >= 2 and all(x for x in v)]
        sp[key] = {'mean_sd_over_seeds': float(np.mean(d)) if d else None,
                   'n_cells_with_seeds': len(d)}
    out['seed_spread'] = sp

    # ------------------------------------------------------------- verdict
    t = out.get('trend_ratio_strong_vs_params') or \
        out.get('trend_ratio_vs_params')
    if t:
        b, se = t['slope_per_efold'], t['se']
        grows = b - 2 * se > 0.05
        shrinks = b + 2 * se < -0.05
        out['verdict'] = {
            'slope_per_efold_of_parameters': b, 'se': se,
            'registered_P5': 'FLAT: |slope| < 0.05 per e-fold',
            'call': ('GROWS -- the negative is a small-model artifact, scale up'
                     if grows else
                     ('SHRINKS -- the family gets LESS compressible with size'
                      if shrinks else
                      'FLAT -- P5 CONFIRMED: a property of the family, '
                      'not of the smallest model')),
            'range_over_cells': [min(r['ratio_strong'] or r['ratio']
                                     for r in s0),
                                 max(r['ratio_strong'] or r['ratio']
                                     for r in s0)]}
    below1 = [r for r in s0 if r['ratio_struct'] and r['ratio_struct'] < 1.0]
    out['P7_structure_only_below_1'] = {
        'cells_below_1': len(below1), 'cells_measured':
            len([r for r in s0 if r['ratio_struct']]),
        'call': 'CONFIRMED' if len(below1) == len(
            [r for r in s0 if r['ratio_struct']]) else 'PARTIAL'}
    json.dump(out, open(f'{HERE}/tf_cgrid_summary.json', 'w'), indent=2)

    # ------------------------------------------------------------- table
    L = ['# Compressibility across the grid',
         '',
         'One scalar per cell: bits(best description) against bits(the SAME',
         'weights naively quantised) at a MATCHED score, median over nine',
         'matched-score levels (five held-CE / KL levels expressed as a',
         'fraction of that cell\'s own headroom over the unigram floor, four',
         'absolute KL levels).  `strong` uses the stronger of the two naive',
         'scale groupings as the denominator, which removes the per-row scale',
         'overhead that would otherwise make small-width cells look more',
         'compressible for a trivial reason.  `structure only` restricts the',
         'numerator to descriptions made out of an INTERPRETATION (low rank,',
         'row prototypes, subspace codebooks, exact anchor rows, and each of',
         'those plus a coded remainder) -- recodings excluded.',
         '',
         '| depth | width | seed | params | emb share | held CE | ratio '
         '(median) | ratio range | ratio vs STRONG naive | structure only | '
         'structure only vs strong | levels structure reaches |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['depth'], r['width'], r['seed'])):
        f = lambda v: '—' if v is None else f'{v:.3f}'
        L.append(f"| {r['depth']} | {r['width']} | {r['seed']} | "
                 f"{r['n_params']:,} | {r['emb_share']:.0%} | "
                 f"{r['held_ce']:.4f} | **{f(r['ratio'])}** | "
                 f"{f(r['ratio_min'])}–{f(r['ratio_max'])} | "
                 f"{f(r['ratio_strong'])} | {f(r['ratio_struct'])} | "
                 f"{f(r['ratio_struct_strong'])} | "
                 f"{r['levels_structure_reaches']}/{r['n_levels']} |")
    L += ['', '## Trend', '', '```', json.dumps(
        {k: v for k, v in out.items() if k != 'depth_effect_at_fixed_width'},
        indent=2), '```']
    open(f'{HERE}/tf_cgrid_table.md', 'w').write('\n'.join(L) + '\n')

    # ------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                                  # pragma: no cover
        print('no matplotlib:', e)
        print('\n'.join(L[:40]))
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    panels = [('ratio_strong',
               'Best description vs naive quantisation',
               'everything we can build, including recodings'),
              ('ratio_struct',
               'Descriptions made out of an INTERPRETATION',
               'low rank, prototypes, codebooks, anchors (+ coded remainder)')]
    for ax, (key, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        for sp_ in ('left', 'bottom'):
            ax.spines[sp_].set_color(MUTED)
            ax.spines[sp_].set_linewidth(0.8)
        ax.grid(True, which='major', color='#e6e5e1', lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(1.0, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.annotate('no gain over naive quantisation', (1.0, 1.0),
                    xycoords=('axes fraction', 'data'),
                    xytext=(-4, 4), textcoords='offset points',
                    ha='right', va='bottom', fontsize=8, color=MUTED)
        for dep in (1, 2, 3, 4):
            pts = sorted([(r['n_params'], r[key]) for r in rows
                          if r['depth'] == dep and r['seed'] == 0 and r[key]])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            c = DEPTH_COLOR[dep]
            ax.plot(x, y, '-', color=c, lw=2, zorder=3)
            ax.plot(x, y, 'o', color=c, ms=8, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f'depth {dep}', (x[-1], y[-1]),
                        xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=9, color=INK2)
            # seed replicates as small open marks
            rep = [(r['n_params'], r[key]) for r in rows
                   if r['depth'] == dep and r['seed'] != 0 and r[key]]
            if rep:
                ax.plot([p[0] for p in rep], [p[1] for p in rep], 'o',
                        mfc='none', mec=c, ms=6, mew=1.4, zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel('parameters', fontsize=9, color=INK2)
        ax.set_ylabel('bits(naive) / bits(best)  at matched score',
                      fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc='left', pad=14)
        ax.annotate(sub, (0, 1.015), xycoords='axes fraction', fontsize=8.5,
                    color=MUTED, va='bottom')
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_xlim(right=max(r['n_params'] for r in rows) * 2.2)
    fig.suptitle('Does interpretable structure compress better as the model '
                 'grows?', fontsize=12.5, color=INK, x=0.008, ha='left',
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f'{HERE}/fig_tf_compressibility_vs_size.png', dpi=150,
                facecolor=SURFACE)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
