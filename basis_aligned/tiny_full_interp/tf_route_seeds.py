"""SEED REPLICATION OF THE DEPTH-LADDER ROUTE MAGNITUDES (FINDING 14 §2-§3).

FINDING 14 quoted every route number from SEED 0 and said so.  On this
programme's own record a single seed has been wrong three times, so this reads
the same quantities out of all three seeds of every depth-3/4 cell and states,
per claim, whether it survives.

Reported per cell, mean +- sd over seeds and the full per-seed list:
  * for every layer l >= 1, every upstream source deleted from that layer's
    Q/K/V read only, in BOTH the zeroing and the resampling flavour;
  * the largest attention-to-attention source and its share of the DOMINANT
    MLP term, computed PER SEED and then aggregated (the identity of the
    largest source is itself a per-seed fact and is reported);
  * layer-0 attention into every downstream read (the claimed-mute channel);
  * the route-USE test: the fraction of the induction score removed by
    cutting each attention-to-attention route, from `*_routeuse.json`.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ms(v):
    v = [float(x) for x in v]
    return (float(np.mean(v)),
            float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, v)


def load_cells(pat='tf_vanilla_d*_w*_b8192_s*_interp3.json'):
    cells = defaultdict(dict)
    for f in sorted(glob.glob(f'{HERE}/{pat}')):
        m = re.search(r'_d(\d+)_w(\d+)_b8192_s(\d+)_interp3\.json$', f)
        if not m:
            continue
        d = json.load(open(f))
        if 'read_ablation_causal' not in d:
            continue
        dep, w, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells[(dep, w)][s] = d
    return cells


def per_seed_route(rk, L):
    """Per LAYER: dominant source, dominant MLP, largest attention source and
    the attention/dominant-MLP ratio.  Harsher of {zero, resample} per source,
    exactly as tf_depth_report.route_table does, so the two agree by
    construction."""
    out = {}
    for l in range(1, L):
        z = {k.split('_')[-1]: v for k, v in rk.items()
             if k.startswith(f'l{l}_read_zero_')}
        r = {k.split('_')[-1]: v for k, v in rk.items()
             if k.startswith(f'l{l}_read_resample_')}
        h = {k: max(z.get(k, 0.0), r.get(k, 0.0)) for k in set(z) | set(r)}
        mlp = {k: v for k, v in h.items() if k.startswith('M')}
        att = {k: v for k, v in h.items() if k.startswith('A')}
        dom_mlp = max(mlp, key=mlp.get) if mlp else None
        top_a = max(att, key=att.get) if att else None
        out[l] = {
            'dominant_source': max(h, key=h.get),
            'dominant_mlp': dom_mlp,
            'dominant_mlp_kl': mlp[dom_mlp] if dom_mlp else None,
            'largest_attention_source': top_a,
            'largest_attention_zero': z.get(top_a),
            'largest_attention_resample': r.get(top_a),
            'attention_over_dominant_mlp':
                (att[top_a] / mlp[dom_mlp])
                if (top_a and dom_mlp and mlp[dom_mlp] > 0) else None,
            'A0_zero': z.get('A0'), 'A0_resample': r.get('A0')}
    return out


def main():
    cells = load_cells()
    rep = {'what': 'seed replication of the FINDING 14 route magnitudes',
           'cells': {}}
    for (dep, w), by_seed in sorted(cells.items()):
        if dep < 2:
            continue
        seeds = sorted(by_seed)
        rts = {s: per_seed_route(by_seed[s]['read_ablation_causal']
                                 ['kl_from_model'], dep) for s in seeds}
        cell = {'depth': dep, 'width': w, 'seeds': seeds,
                'n_seeds': len(seeds), 'layers': {}}
        for l in range(1, dep):
            srcs = [rts[s][l]['largest_attention_source'] for s in seeds]
            zs = [rts[s][l]['largest_attention_zero'] for s in seeds]
            rs = [rts[s][l]['largest_attention_resample'] for s in seeds]
            fr = [rts[s][l]['attention_over_dominant_mlp'] for s in seeds]
            a0z = [rts[s][l]['A0_zero'] for s in seeds]
            a0r = [rts[s][l]['A0_resample'] for s in seeds]
            dm = [rts[s][l]['dominant_mlp'] for s in seeds]
            dmk = [rts[s][l]['dominant_mlp_kl'] for s in seeds]
            e = {'largest_attention_source_per_seed': srcs,
                 'largest_attention_source_agrees_across_seeds':
                     len(set(srcs)) == 1,
                 'dominant_mlp_per_seed': dm,
                 'dominant_mlp_agrees_across_seeds': len(set(dm)) == 1}
            for nm, v in (('attn_zero', zs), ('attn_resample', rs),
                          ('attn_over_dominant_mlp', fr),
                          ('dominant_mlp_kl', dmk),
                          ('A0_zero', a0z), ('A0_resample', a0r)):
                mu, sd, vals = ms(v)
                e[nm] = {'mean': mu, 'sd': sd, 'per_seed': vals,
                         'min': min(vals), 'max': max(vals),
                         'cv': (sd / abs(mu)) if mu else None}
            cell['layers'][l] = e
        # the cell headline: the largest attention/MLP ratio over layers
        pm = [max((rts[s][l]['attention_over_dominant_mlp'] or 0.0)
                  for l in range(1, dep)) for s in seeds]
        mu, sd, vals = ms(pm)
        cell['max_attention_over_dominant_mlp'] = {
            'mean': mu, 'sd': sd, 'per_seed': vals,
            'min': min(vals), 'max': max(vals)}
        # A0 into ANY downstream read, worst case over layers and flavours
        a0 = [max(max(rts[s][l]['A0_zero'] or 0.0,
                      rts[s][l]['A0_resample'] or 0.0)
                  for l in range(1, dep)) for s in seeds]
        mu, sd, vals = ms(a0)
        cell['A0_into_any_downstream_read_worst'] = {
            'mean': mu, 'sd': sd, 'per_seed': vals, 'max': max(vals)}
        # route USE
        ru = {}
        for s in seeds:
            f = (f'{HERE}/tf_vanilla_d{dep}_w{w}_b8192_s{s}_routeuse.json')
            if os.path.exists(f):
                j = json.load(open(f))
                ru[s] = {'baseline_induction': j['baseline_induction'],
                         'fraction_removed':
                             j['fraction_of_induction_removed'],
                         'bag': {k: v['bag_score_mean']
                                 for k, v in j['arms'].items()}}
        if ru:
            arms = sorted(set().union(*[set(v['fraction_removed'])
                                        for v in ru.values()]))
            cell['route_use'] = {'n_seeds': len(ru), 'seeds': sorted(ru),
                                 'baseline_induction': ms(
                                     [ru[s]['baseline_induction']
                                      for s in sorted(ru)])[:2]}
            cell['route_use']['fraction_removed'] = {}
            for a in arms:
                v = [ru[s]['fraction_removed'].get(a) for s in sorted(ru)
                     if ru[s]['fraction_removed'].get(a) is not None]
                if not v:
                    continue
                mu, sd, vals = ms(v)
                cell['route_use']['fraction_removed'][a] = {
                    'mean': mu, 'sd': sd, 'per_seed': vals,
                    'min': min(vals), 'max': max(vals)}
        rep['cells'][f'd{dep}_w{w}'] = cell

    # ------------------------------------------------- claim-by-claim verdict
    C = rep['cells']
    v = {}
    deep = [k for k in C if C[k]['depth'] in (3, 4)]
    v['route_opens_at_depth_3'] = {
        'claim': 'the largest attention-to-attention read is 17-39% of the '
                 'dominant MLP term at every depth-3/4 cell, against 1e-5 at '
                 'depth 2 (seed-0 numbers in FINDING 14 sec 2a)',
        'per_cell_mean_sd_min_max': {
            k: [C[k]['max_attention_over_dominant_mlp'][x]
                for x in ('mean', 'sd', 'min', 'max')] for k in sorted(deep)},
        'depth2_reference': {
            k: [C[k]['max_attention_over_dominant_mlp'][x]
                for x in ('mean', 'sd', 'min', 'max')]
            for k in sorted(C) if C[k]['depth'] == 2},
        'survives': all(C[k]['max_attention_over_dominant_mlp']['min'] > 0.01
                        for k in deep),
        'n_seeds': {k: C[k]['n_seeds'] for k in sorted(deep)}}
    v['A0_stays_mute'] = {
        'claim': 'layer-0 attention costs 1e-6 to 3e-5 nats into every '
                 'downstream read at every depth and width',
        'worst_over_all_deep_cells_and_seeds': max(
            C[k]['A0_into_any_downstream_read_worst']['max'] for k in deep),
        'per_cell': {k: C[k]['A0_into_any_downstream_read_worst']['per_seed']
                     for k in sorted(deep)},
        'survives': all(C[k]['A0_into_any_downstream_read_worst']['max'] < 1e-4
                        for k in deep)}
    ruc = {k: C[k]['route_use'] for k in sorted(deep) if 'route_use' in C[k]}
    v['algorithm_runs_on_the_route'] = {
        'claim': 'cutting layer-1 attention out of layer 2 read removes 94.5% '
                 'of the induction score at depth 3 width 128 (seed 0), while '
                 'cutting layer-0 attention removes 0.0%',
        'per_cell': {k: {a: [d['mean'], d['sd'], d['min'], d['max']]
                         for a, d in ruc[k]['fraction_removed'].items()}
                     for k in ruc},
        'n_seeds': {k: ruc[k]['n_seeds'] for k in ruc}}
    rep['verdicts'] = v
    json.dump(rep, open(f'{HERE}/tf_route_seeds.json', 'w'), indent=2)

    # ----------------------------------------------------------- markdown
    L = ['# Depth-ladder route magnitudes at THREE SEEDS',
         '',
         'FINDING 14 quoted these from seed 0. Mean +- sd over the seeds that '
         'exist, with the per-seed values, because on this programme a single '
         'seed has been wrong three times.', '',
         '## The attention-to-attention route (harsher of zero/resample over '
         'the dominant MLP term, computed per seed)', '',
         '| cell | seeds | largest attention source (per seed) | zero KL | '
         'resample KL | share of dominant MLP | per-seed shares |',
         '|---|---|---|---|---|---|---|']
    for k in sorted(C, key=lambda x: (C[x]['depth'], C[x]['width'])):
        c = C[k]
        for l, e in c['layers'].items():
            src = '/'.join(e['largest_attention_source_per_seed'])
            L.append(
                f"| {k} L{l} | {c['n_seeds']} | {src} | "
                f"{e['attn_zero']['mean']:.4g} ± {e['attn_zero']['sd']:.2g} | "
                f"{e['attn_resample']['mean']:.4g} ± "
                f"{e['attn_resample']['sd']:.2g} | "
                f"**{e['attn_over_dominant_mlp']['mean']:.3g} ± "
                f"{e['attn_over_dominant_mlp']['sd']:.2g}** | "
                + ', '.join(f'{x:.3g}'
                            for x in e['attn_over_dominant_mlp']['per_seed'])
                + ' |')
    L += ['', '## Layer-0 attention into any downstream read (the mute '
          'channel)', '', '| cell | worst over layers and flavours, per seed |',
          '|---|---|']
    for k in sorted(C, key=lambda x: (C[x]['depth'], C[x]['width'])):
        L.append(f"| {k} | " + ', '.join(
            f'{x:.2g}' for x in
            C[k]['A0_into_any_downstream_read_worst']['per_seed']) + ' |')
    L += ['', '## Route USE: fraction of the induction score removed', '',
          '| cell | seeds | arm | mean ± sd | per seed |', '|---|---|---|---|---|']
    for k in sorted(ruc, key=lambda x: (C[x]['depth'], C[x]['width'])):
        for a, d in ruc[k]['fraction_removed'].items():
            L.append(f"| {k} | {ruc[k]['n_seeds']} | {a} | "
                     f"{d['mean']:+.3f} ± {d['sd']:.3f} | "
                     + ', '.join(f'{x:+.3f}' for x in d['per_seed']) + ' |')
    L += ['', '## Verdicts', '', '```',
          json.dumps(rep['verdicts'], indent=2), '```']
    open(f'{HERE}/tf_route_seeds_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
