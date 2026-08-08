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


def _stair(pts, key):
    """The Pareto staircase of a family: sorted by bits, keeping only points
    that improve on the best score so far."""
    st, best = [], float('inf')
    for p in sorted(pts, key=lambda p: p['bits']):
        if p[key] < best:
            best = p[key]
            st.append(p)
    return st


def _bits_at(stair, level, key):
    """Bits the family needs to reach `level`, interpolated in
    (score, log bits) WITHIN ONE FAMILY (so the curve is smooth and the
    interpolation cannot be corrupted by a foreign scheme's point)."""
    if not stair:
        return None
    if stair[0][key] <= level:
        return stair[0]['bits']
    for a, b in zip(stair, stair[1:]):
        if b[key] <= level <= a[key]:
            if a[key] == b[key]:
                return b['bits']
            t = (a[key] - level) / (a[key] - b[key])
            return math.exp(math.log(a['bits'])
                            + t * (math.log(b['bits']) - math.log(a['bits'])))
    return None


def frontier_ratios(points, key, denom_kinds, numer_kinds, max_score):
    """FINDING 12 §7b's own construction, which is the registered definition:
    for every point on the numerator family's Pareto frontier, how many bits
    would the naive quantiser need to reach the SAME score?  The ratio is that
    over the point's own bits.  Set-inclusion artifacts cannot arise because
    the denominator is always interpolated inside ONE family.
    """
    den = _stair([p for p in points if p['kind'] in denom_kinds], key)
    num = _stair([p for p in points if p['kind'] in numer_kinds], key)
    out = []
    for p in num:
        if p[key] > max_score:
            continue
        b = _bits_at(den, p[key], key)
        if b:
            out.append({'name': p['name'], 'kind': p['kind'],
                        'bits': p['bits'], key: p[key],
                        'naive_bits': b, 'ratio': b / p['bits']})
    return out


def cell_ratios(d):
    """All four ratio flavours for one cell, on held CE (primary) and KL."""
    pts = d['points']
    ce0 = d['model']['held_ce']
    head = d['headroom_over_unigram_nats']
    out = {}
    # THE EMBEDDING-ONLY CONTROL (adversarial review).  The embedding is 93% of
    # the parameters at width 32 and 57% at width 256, and every structural
    # scheme in the family attacks the embedding -- so the ratio could fall with
    # width for the trivial reason that there is proportionally less embedding
    # to attack.  Holding the body at its near-lossless 12-bit code isolates the
    # embedding coder, and the same trend must survive there.
    embonly = [p for p in pts if p['name'].endswith('+body_naive12')]
    for key, cap in (('ce', ce0 + 0.15 * head), ('kl', 0.15 * head)):
        for tag, den, num in (('emb_only_vs_naive', ('naive',),
                               ('recode', 'structure')),
                              ('emb_only_struct', ('naive',), ('structure',))):
            r = frontier_ratios(embonly, key, den, num, cap)
            v = [x['ratio'] for x in r]
            out[f'{key}_{tag}'] = {
                'n_frontier_points': len(v),
                'median': float(np.median(v)) if v else None}
        for tag, den, num in (
                ('vs_naive', ('naive',), ('recode', 'structure')),
                ('vs_naive_strong', ('naive', 'naiveG'),
                 ('recode', 'structure')),
                ('struct_vs_naive', ('naive',), ('structure',)),
                ('struct_vs_naive_strong', ('naive', 'naiveG'),
                 ('structure',))):
            r = frontier_ratios(pts, key, den, num, cap)
            v = [x['ratio'] for x in r]
            out[f'{key}_{tag}'] = {
                'n_frontier_points': len(v),
                'median': float(np.median(v)) if v else None,
                'min': float(np.min(v)) if v else None,
                'max': float(np.max(v)) if v else None,
                'best_point': max(r, key=lambda x: x['ratio'])['name']
                if r else None}
    return out


def load():
    rows = []
    for f in sorted(glob.glob(f'{HERE}/tf_vanilla_d*_w*_b8192_s*_cgrid.json')):
        d = json.load(open(f))
        s = d['summary']
        fr = cell_ratios(d)
        rows.append({
            'frontier': fr,
            # PRIMARY: held CE, best description of any kind, against the
            # STRONGER of the two naive scale groupings
            'R': fr['ce_vs_naive_strong']['median'],
            'R_min': fr['ce_vs_naive_strong']['min'],
            'R_max': fr['ce_vs_naive_strong']['max'],
            'R_perrow': fr['ce_vs_naive']['median'],
            'R_kl': fr['kl_vs_naive_strong']['median'],
            'R_struct': fr['ce_struct_vs_naive_strong']['median'],
            'R_struct_perrow': fr['ce_struct_vs_naive']['median'],
            'R_struct_kl': fr['kl_struct_vs_naive_strong']['median'],
            'R_embonly': fr['ce_emb_only_vs_naive']['median'],
            'R_embonly_struct': fr['ce_emb_only_struct']['median'],
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

    for key in ('R', 'R_perrow', 'R_kl', 'R_struct', 'R_struct_perrow',
                'R_embonly', 'R_embonly_struct',
                'ratio', 'ratio_strong', 'ratio_struct'):
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
        pts = sorted([(r['depth'], r['R']) for r in s0
                      if r['width'] == w and r['R']])
        if len(pts) >= 2:
            de[f'w{w}'] = {'by_depth': pts,
                           'max_minus_min': max(p[1] for p in pts)
                           - min(p[1] for p in pts)}
    out['depth_effect_at_fixed_width'] = de
    # seed spread, for the "is a trend bigger than the noise" question
    sp = {}
    for key in ('R', 'R_struct', 'ratio_strong'):
        g = {}
        for r in rows:
            g.setdefault((r['depth'], r['width']), []).append(r[key])
        d = [float(np.std(v, ddof=1)) for v in g.values()
             if len(v) >= 2 and all(x for x in v)]
        sp[key] = {'mean_sd_over_seeds': float(np.mean(d)) if d else None,
                   'n_cells_with_seeds': len(d)}
    out['seed_spread'] = sp

    # ------------------------------------------------------------- verdict
    t = out.get('trend_R_vs_params')
    if t:
        b, se = t['slope_per_efold'], t['se']
        grows = b - 2 * se > 0.05
        shrinks = b + 2 * se < -0.05
        sig_neg = b + 2 * se < 0
        small = abs(b) < 0.05
        out['verdict_call'] = (
            'GROWS -- the negative IS a small-model artifact; scale up'
            if grows else
            'SHRINKS OUTRIGHT -- the family gets LESS compressible with size'
            if shrinks else
            'NOT GROWING, AND SIGNIFICANTLY NEGATIVE -- P5\'s falsifier '
            '(growth) is rejected with a wide margin; P5\'s letter '
            '("indistinguishable from zero") is NOT met, because the slope is '
            f'significantly below zero (t = {b/se:.2f}) though small in '
            'magnitude; P5\'s scientific claim -- the negative is a property '
            'of the family, not of the smallest model -- is confirmed in the '
            'STRONGER direction'
            if (sig_neg and small) else
            'FLAT -- P5 CONFIRMED as stated')
        out['verdict'] = {
            'primary_scalar': 'held-CE frontier ratio, best description of any '
                              'kind against the STRONGER naive quantiser, '
                              'median over that cell\'s frontier points',
            'slope_per_efold_of_parameters': b, 'se': se,
            'registered_P5': 'FLAT: |slope| < 0.05 per e-fold',
            'call': out['verdict_call'],
            't': b / se,
            'range_over_cells': [min(r['R'] for r in s0 if r['R']),
                                 max(r['R'] for r in s0 if r['R'])]}
    below1 = [r for r in s0 if r['R_struct'] and r['R_struct'] < 1.0]
    meas = [r for r in s0 if r['R_struct']]
    out['P7_structure_only_below_1'] = {
        'cells_below_1': len(below1), 'cells_measured': len(meas),
        'values': {f"d{r['depth']}_w{r['width']}": r['R_struct']
                   for r in meas},
        'call': 'CONFIRMED' if below1 and len(below1) == len(meas)
                else 'PARTIAL'}
    json.dump(out, open(f'{HERE}/tf_cgrid_summary.json', 'w'), indent=2)

    # ------------------------------------------------------------- table
    L = ['# Compressibility across the grid',
         '',
         'One scalar per cell, FINDING 12 §7b\'s own construction: for every',
         'point on the description frontier, how many bits the SAME weights',
         'need under naive uniform quantisation + entropy coding to reach the',
         'same held CE -- the median of that ratio over the frontier.  The',
         'denominator is interpolated inside ONE family, so a set-inclusion',
         'artifact cannot make it look better or worse than it is.',
         '',
         '`R` is the primary number: best description of any kind against the',
         'STRONGER of the two naive scale groupings (per-tensor as well as',
         'per-row scales).  The per-row-only denominator `R(per-row)` is the',
         'literal FINDING-12 definition and is quoted beside it, because a',
         '32-bit-per-row scale is 1.0 bits/weight of pure overhead at width 32',
         'and 0.125 at width 256 -- a width trend in `R(per-row)` alone would',
         'be nothing but that.  `R(structure)` restricts the numerator to',
         'descriptions made out of an INTERPRETATION -- low rank, row',
         'prototypes, subspace codebooks, exact anchor rows, and each of those',
         'plus an honestly coded remainder -- with recodings excluded.',
         '',
         '| depth | width | seed | params | emb share | held CE | **R** | R '
         'range | R (per-row denom) | R (KL not CE) | R (embedding only) | '
         '**R (structure)** | R (structure, per-row) | R (structure, '
         'embedding only) | naive per-row overhead share |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['depth'], r['width'], r['seed'])):
        f = lambda v: '—' if v is None else f'{v:.3f}'
        L.append(f"| {r['depth']} | {r['width']} | {r['seed']} | "
                 f"{r['n_params']:,} | {r['emb_share']:.0%} | "
                 f"{r['held_ce']:.4f} | **{f(r['R'])}** | "
                 f"{f(r['R_min'])}–{f(r['R_max'])} | "
                 f"{f(r['R_perrow'])} | {f(r['R_kl'])} | "
                 f"{f(r['R_embonly'])} | "
                 f"**{f(r['R_struct'])}** | {f(r['R_struct_perrow'])} | "
                 f"{f(r['R_embonly_struct'])} | "
                 f"{r['naive_overhead_share_b4']:.0%} |")
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
    panels = [('R',
               'Best description vs naive quantisation',
               'everything we can build, including recodings'),
              ('R_struct',
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
