"""THE DEPTH-3 ARCHITECTURE-VARIANT SLICE: verdict against the predictions
registered in `tf_d3_variant_predictions.json` before the first training step.

The question: at depth 2 the five interpretable architectures opened a residual
route the plain model left empty AND acquired induction at width 128 where the
plain model needed 256.  At depth 3 the plain model opens that route by itself
and inducts at width 128.  So are the architectures an ACCELERANT (a), do they
still ADD something (b), or do they INTERFERE with what depth supplies (c)?

Everything is read out of the same instruments the depth-2 slice and the depth
ladder used: `tf_interp3.py` verbatim and `tf_depth_addendum.py`.  Every route
number is quoted as [zero, resample] and every induction score against its own
per-cell planted-oracle power floor.

ROUND-4 REVIEW COMPLIANCE.  The independent review of the depth ladder
(`tf_reviewer_round_4.json`) established that a read-ablation KL is a quadratic
function of the write's norm share of the read it enters (slope 1.99, r =
0.994, residual 0.26 dex over 243 pairs), so a bare route number is a
MAGNITUDE statement, not evidence of a gated channel.  This report therefore
prints the write's norm share beside every route KL and refuses to use the word
'open'.  It also decides the induction question over MODEL SEEDS rather than
against a probe-noise floor, because that floor shrinks with the number of
probe seeds and is not a property of the model.
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ['vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink']
T_CRIT_2DF = 3.182                      # two-sided 95% with three seeds


def load(depth=3, width=128, suffix=''):
    out = defaultdict(dict)
    for var in VARIANTS:
        for f in sorted(glob.glob(
                f'{HERE}/tf_{var}_d{depth}_w{width}_b8192_s*{suffix}'
                f'_interp3.json')):
            m = re.search(r'_s(\d+)' + re.escape(suffix) + r'_interp3\.json$', f)
            if not m:
                continue
            out[var][int(m.group(1))] = json.load(open(f))
    return out


def ms(v):
    v = [float(x) for x in v]
    return {'mean': float(np.mean(v)),
            'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': v, 'n': len(v)}


def route(d, li, src):
    rk = d['read_ablation_causal']['kl_from_model']
    z = rk.get(f'l{li}_read_zero_{src}')
    r = rk.get(f'l{li}_read_resample_{src}')
    if z is None and r is None:
        return None
    mlp = {k: max(rk.get(f'l{li}_read_zero_{k}', 0.0),
                  rk.get(f'l{li}_read_resample_{k}', 0.0))
           for k in [f'M{j}' for j in range(li)]}
    dom = max(mlp.values()) if mlp else None
    tot = sum(max(rk.get(f'l{li}_read_zero_{k}', 0.0),
                  rk.get(f'l{li}_read_resample_{k}', 0.0))
              for k in ['e'] + [f'A{j}' for j in range(li)]
              + [f'M{j}' for j in range(li)])
    sg = d.get('stream_geometry', {})
    srcs = ['e'] + [f'A{j}' for j in range(li)] + [f'M{j}' for j in range(li)]
    den = math.sqrt(sum(sg.get(f'{k}_norm', 0.0) ** 2 for k in srcs)) \
        if sg else None
    return {'zero': z, 'resample': r,
            'share_of_dominant_mlp': (max(z or 0, r or 0) / dom) if dom else None,
            'share_of_total_read_mass': (max(z or 0, r or 0) / tot) if tot else None,
            'write_norm': sg.get(f'{src}_norm') if sg else None,
            'write_norm_share_of_read':
                (sg.get(f'{src}_norm') / den) if (sg and den) else None}


def main():
    pred = json.load(open(f'{HERE}/tf_d3_variant_predictions.json'))
    cells = load()
    rep = {'registered_predictions': pred, 'depth': 3, 'width': 128,
           'variants': {}}
    for var, by_seed in cells.items():
        seeds = sorted(by_seed)
        ds = [by_seed[s] for s in seeds]
        v = {'n_seeds': len(seeds), 'seeds': seeds,
             'variant': ds[0]['variant'],
             'n_slots': ds[0]['config'].get('n_slots'),
             'slot': ds[0]['config'].get('slot'),
             'stream_width': ds[0]['config'].get('n_slots', 1)
                             * (ds[0]['config'].get('slot') or 0) or None,
             'params': ds[0]['params']['total'],
             'all_gates_pass': all(d['fold_gate']['pass']
                                   and d['decomposition_control']['pass']
                                   for d in ds),
             'held_ce': ms([d['train']['final_held_ce'] for d in ds]),
             'bits_per_byte': ms([d['train']['bits_per_byte'] for d in ds]),
             'induction': ms([d['rung3_induction']['induction_score_mean']
                              for d in ds]),
             'induction_floor': ms(
                 [d['induction_power']['detectable_effect_floor_nats_3se']
                  for d in ds]),
             'natural_swap': ms([d['natural_induction']['ORDER_ONLY_patch_swap']
                                 ['mean'] for d in ds])}
        i = v['induction']
        se = i['sd'] / math.sqrt(i['n']) if i['n'] > 1 else None
        v['model_seed_t'] = (i['mean'] / se) if se else None
        v['model_seed_test_positive'] = bool(se and i['mean'] / se > T_CRIT_2DF)
        v['seeds_above_own_probe_floor'] = int(sum(
            d['rung3_induction']['induction_score_mean']
            > d['induction_power']['detectable_effect_floor_nats_3se']
            for d in ds))
        v['routes'] = {}
        for li, src in ((1, 'A0'), (2, 'A0'), (2, 'A1')):
            rs = [route(d, li, src) for d in ds]
            rs = [r for r in rs if r]
            if not rs:
                continue
            v['routes'][f'{src}_into_layer{li}_read'] = {
                k: ms([r[k] for r in rs if r[k] is not None])
                for k in ('zero', 'resample', 'share_of_dominant_mlp',
                          'share_of_total_read_mass',
                          'write_norm_share_of_read')}
        ru = {}
        for s in seeds:
            f = (f'{HERE}/tf_{var}_d3_w128_b8192_s{s}_routeuse.json')
            if os.path.exists(f):
                ru[s] = json.load(open(f))
        if ru:
            arms = sorted(set().union(*[set(j['fraction_of_induction_removed'])
                                        for j in ru.values()]))
            v['route_use'] = {'n_seeds': len(ru),
                              'baseline_induction': ms(
                                  [ru[s]['baseline_induction']
                                   for s in sorted(ru)]),
                              'fraction_removed': {}}
            for a in arms:
                xs = [ru[s]['fraction_of_induction_removed'].get(a)
                      for s in sorted(ru)]
                xs = [x for x in xs if x is not None]
                if xs:
                    v['route_use']['fraction_removed'][a] = ms(xs)
        rep['variants'][var] = v

    V = rep['variants']
    van = V.get('vanilla')
    verd = {}
    if van:
        # ---------------- PD1
        verd['PD1_qualitative_advantage'] = {
            'vanilla_induction': van['induction'],
            'vanilla_seeds_above_own_probe_floor':
                van['seeds_above_own_probe_floor'],
            'vanilla_model_seed_test_positive':
                van['model_seed_test_positive'],
            'call': ('CONFIRMED - the plain model inducts at this cell, so '
                     '"the variants induct where the plain model cannot" is '
                     'no longer true at depth 3 width 128'
                     if van['model_seed_test_positive'] else
                     'REFUTED - the plain model does not clear the model-seed '
                     'test at depth 3 width 128')}
        # ---------------- PD2  the decision rule
        ratios = {}
        for var, v in V.items():
            if var == 'vanilla' or van['induction']['mean'] == 0:
                continue
            ratios[var] = v['induction']['mean'] / van['induction']['mean']
        non_pred = {k: r for k, r in ratios.items() if k != 'predicate'}
        n_above = sum(r > 2.0 for r in ratios.values())
        n_below = sum(r < 0.5 for r in ratios.values())
        if n_above >= 3:
            outcome = ('(b) PERSISTS - the architectures still add induction '
                       'magnitude at a depth where the plain model has it')
        elif n_below >= 3:
            outcome = ('(c) INVERTS - depth and the architectures interfere; '
                       'the architectures are WORSE than the plain model at '
                       'depth 3')
        else:
            outcome = ('(a) ACCELERANT - the advantage the architectures had '
                       'at depth 2 is what depth supplies by itself at depth 3')
        verd['PD2_magnitude_advantage'] = {
            'induction_ratio_to_vanilla': ratios,
            'excluding_predicate': non_pred,
            'n_variants_above_2x': n_above, 'n_variants_below_0.5x': n_below,
            'call': outcome}
        # ---------------- PD3  the variants' own layer-0 channel
        a0 = {var: (v['routes'].get('A0_into_layer1_read', {})
                    .get('zero', {}).get('mean'))
              for var, v in V.items()}
        a0n = {var: (v['routes'].get('A0_into_layer1_read', {})
                     .get('write_norm_share_of_read', {}).get('mean'))
               for var, v in V.items()}
        open_v = [k for k, x in a0.items()
                  if k != 'vanilla' and x is not None and x >= 0.05]
        verd['PD3_variant_layer0_channel'] = {
            'A0_into_layer1_read_zero_kl': a0,
            'A0_write_norm_share_of_that_read': a0n,
            'n_variants_at_or_above_0.05_nats': len(open_v),
            'call': ('CONFIRMED' if len(open_v) >= 4 else
                     ('REFUTED' if len(open_v) <= 2 else 'PARTIAL')),
            'review_caveat': 'per tf_reviewer_round_4.json O2b this is a '
                             'MAGNITUDE statement about how much the first '
                             'attention block writes, not a claim that a '
                             'channel is open or closed; the write norm share '
                             'is printed beside it for exactly that reason'}
        # ---------------- PD4  the depth-supplied route
        a1 = {var: (v['routes'].get('A1_into_layer2_read', {})
                    .get('share_of_dominant_mlp', {}).get('mean'))
              for var, v in V.items()}
        verd['PD4_depth_supplied_route'] = {
            'A1_into_layer2_share_of_dominant_mlp': a1,
            'all_six_between_0.10_and_0.70':
                all(x is not None and 0.10 <= x <= 0.70 for x in a1.values()),
            'variant_mean_at_or_above_vanilla':
                (np.mean([x for k, x in a1.items()
                          if k != 'vanilla' and x is not None])
                 >= (a1.get('vanilla') or 0)) if len(a1) > 1 else None}
        # ---------------- PD5  route use
        ruv = {var: (v.get('route_use', {}).get('fraction_removed', {}))
               for var, v in V.items()}
        verd['PD5_route_use'] = {
            'A1_out_of_layer2_read': {
                k: r.get('A1_out_of_layer2_read', {}).get('mean')
                for k, r in ruv.items()},
            'A0_out_of_layer1_read': {
                k: r.get('A0_out_of_layer1_read', {}).get('mean')
                for k, r in ruv.items()},
            'A0_out_of_layer2_read': {
                k: r.get('A0_out_of_layer2_read', {}).get('mean')
                for k, r in ruv.items()},
            'note': 'fractions whose baseline induction is at or below its own '
                    'floor are meaningless ratios and must not be read'}
        # ---------------- PD6 / PD7  loss
        verd['PD6_held_ce'] = {k: v['held_ce'] for k, v in V.items()}
        d2 = load(depth=2, width=128)
        verd['PD7_depth_gain'] = {}
        for var, v in V.items():
            if var in d2 and d2[var]:
                c2 = float(np.mean([d['train']['final_held_ce']
                                    for d in d2[var].values()]))
                verd['PD7_depth_gain'][var] = {
                    'held_ce_depth2': c2,
                    'held_ce_depth3': v['held_ce']['mean'],
                    'gain': c2 - v['held_ce']['mean']}
    rep['verdicts'] = verd
    json.dump(rep, open(f'{HERE}/tf_d3_variant_slice.json', 'w'), indent=2,
              default=str)

    # ------------------------------------------------------------- markdown
    L = ['# The six architectures at DEPTH 3, width 128',
         '',
         'Registered predictions: `tf_d3_variant_predictions.json`, written '
         'before the first training step. Instruments: `tf_interp3.py` '
         'verbatim and `tf_depth_addendum.py` - the same code path as the '
         'depth-2 slice and the depth ladder.',
         '',
         'SLOT GEOMETRY: 128 is not divisible by 2*depth = 6, so `slots` and '
         '`shrink` run with 8 slots of 16 rather than depth 2\'s 4 of 32; '
         '`bandwidth`/`predicate`/`codebook` scatter into 6 solved slots '
         '(stream 168). Controls in `GRID.md`.',
         '',
         '| variant | seeds | n_slots x slot | params | held CE (T512) | '
         'induction ± sd | model-seed t | above own probe floor | '
         'A0 into layer 1 [zero, resample] | A1 into layer 2 [zero, resample] '
         '| A1 share of dominant MLP |', '|' + '---|' * 11]
    for var in VARIANTS:
        v = V.get(var)
        if not v:
            continue
        r0 = v['routes'].get('A0_into_layer1_read', {})
        r1 = v['routes'].get('A1_into_layer2_read', {})

        def g(d_, k):
            x = d_.get(k, {}).get('mean')
            return '—' if x is None else f'{x:.4g}'
        tt = '—' if v['model_seed_t'] is None else f"{v['model_seed_t']:.2f}"
        L.append(
            f"| {var} | {v['n_seeds']} | {v['n_slots']}x{v['slot']} | "
            f"{v['params']:,} | {v['held_ce']['mean']:.4f} ± "
            f"{v['held_ce']['sd']:.4f} | {v['induction']['mean']:+.4f} ± "
            f"{v['induction']['sd']:.4f} | {tt} | "
            f"{v['seeds_above_own_probe_floor']}/{v['n_seeds']} | "
            f"[{g(r0,'zero')}, {g(r0,'resample')}] | "
            f"[{g(r1,'zero')}, {g(r1,'resample')}] | "
            f"{g(r1,'share_of_dominant_mlp')} |")
    L += ['', '## Verdicts against the registered predictions', '', '```',
          json.dumps(verd, indent=2, default=str), '```']
    open(f'{HERE}/tf_d3_variant_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
