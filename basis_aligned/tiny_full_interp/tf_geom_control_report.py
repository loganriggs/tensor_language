"""SLOT-GEOMETRY CONTROLS for the depth-3 six-architecture slice (round 5).

The depth-3 slice forced the two masked-decoder arms (`slots`, `shrink`) onto
8 slots of 16 rather than depth 2's 4 slots of 32, because 128 is not divisible
by 2*depth = 6.  Those are also the two arms that look worst at depth 3.  This
script reads out the two controls that price the deviation, with the SAME
instruments the slice used (`tf_interp3.py` / `tf_depth_addendum.py`):

  (a) DEPTH-2 GEOMETRY CONTROL - the same n_slots change at the depth-2 cell
      whose n_slots=4 answer is already published.  Any difference here is
      geometry alone: same depth, same width, same parameter count.

  (b) WIDTH-192 CONTROL - depth 3 where n_slots = 6 x slot 32 is exact, so the
      masked arms get depth 2's slot size.  Plus a geometry-only contrast at
      the same width (n_slots 8 x slot 24), which separates "width 192 is
      easier" from "6x32 is the right geometry".

Every comparison is over MODEL seeds with a Welch t-test, per the round-4 rule
that a probe-seed floor is not a detection threshold.  Route KLs are quoted as
[zero, resample] beside the write's norm share of the read they enter, per the
round-4 rule that a read-ablation KL is a magnitude, not a routing claim.
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ms(v):
    v = [float(x) for x in v]
    return {'mean': float(np.mean(v)),
            'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': v, 'n': len(v)}


def welch(a, b):
    """Welch t and a normal-approximation two-sided p (n=3, so quote t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return None, None
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df)


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
    sg = d.get('stream_geometry', {})
    srcs = ['e'] + [f'A{j}' for j in range(li)] + [f'M{j}' for j in range(li)]
    den = math.sqrt(sum(sg.get(f'{k}_norm', 0.0) ** 2 for k in srcs)) \
        if sg else None
    return {'zero': z, 'resample': r,
            'share_of_dominant_mlp': (max(z or 0, r or 0) / dom) if dom else None,
            'write_norm_share_of_read':
                (sg.get(f'{src}_norm') / den) if (sg and den) else None}


def arm(variant, depth, width, suffix=''):
    """Load every seed of one arm and summarise it with the slice's own keys."""
    by_seed = {}
    pat = f'{HERE}/tf_{variant}_d{depth}_w{width}_b8192_s*{suffix}_interp3.json'
    for f in sorted(glob.glob(pat)):
        m = re.search(r'_s(\d+)' + re.escape(suffix) + r'_interp3\.json$', f)
        if m:
            by_seed[int(m.group(1))] = json.load(open(f))
    if not by_seed:
        return None
    seeds = sorted(by_seed)
    ds = [by_seed[s] for s in seeds]
    out = {'arm': f'{variant} d{depth} w{width}{suffix}',
           'variant': variant, 'depth': depth, 'width': width,
           'suffix': suffix, 'seeds': seeds, 'n_seeds': len(seeds),
           'n_slots': ds[0]['config'].get('n_slots'),
           'slot': ds[0]['config'].get('slot'),
           'params': ds[0]['params']['total'],
           'body_params': ds[0]['params']['body'],
           'all_gates_pass': all(d['fold_gate']['pass']
                                 and d['decomposition_control']['pass']
                                 for d in ds),
           'held_ce': ms([d['train']['final_held_ce'] for d in ds]),
           'bits_per_byte': ms([d['train']['bits_per_byte'] for d in ds]),
           'induction': ms([d['rung3_induction']['induction_score_mean']
                            for d in ds]),
           'probe_floor': ms([d['induction_power']
                              ['detectable_effect_floor_nats_3se'] for d in ds]),
           'seeds_above_own_probe_floor': int(sum(
               d['rung3_induction']['induction_score_mean']
               > d['induction_power']['detectable_effect_floor_nats_3se']
               for d in ds))}
    i = out['induction']
    se = i['sd'] / math.sqrt(i['n']) if i['n'] > 1 else None
    out['model_seed_t'] = (i['mean'] / se) if se else None
    out['model_seed_test_positive'] = bool(se and i['mean'] / se > 3.182)
    out['routes'] = {}
    for li in range(1, depth):
        for src in [f'A{j}' for j in range(li)]:
            rs = [route(d, li, src) for d in ds]
            rs = [r for r in rs if r]
            if not rs:
                continue
            out['routes'][f'{src}_into_layer{li}_read'] = {
                k: ms([r[k] for r in rs if r[k] is not None])
                for k in ('zero', 'resample', 'share_of_dominant_mlp',
                          'write_norm_share_of_read')}
    ru = {}
    for s in seeds:
        f = (f'{HERE}/tf_{variant}_d{depth}_w{width}_b8192_s{s}{suffix}'
             f'_routeuse.json')
        if os.path.exists(f):
            ru[s] = json.load(open(f))
    if ru:
        arms = sorted(set().union(*[set(j['fraction_of_induction_removed'])
                                    for j in ru.values()]))
        out['route_use'] = {
            'n_seeds': len(ru),
            'baseline_induction': ms([ru[s]['baseline_induction']
                                      for s in sorted(ru)]),
            'fraction_removed': {
                a: ms([ru[s]['fraction_of_induction_removed'][a]
                       for s in sorted(ru)
                       if a in ru[s]['fraction_of_induction_removed']])
                for a in arms}}
    return out


def compare(a, b, label):
    """a vs b on the two instruments that decide the verdict."""
    if a is None or b is None:
        return {'label': label, 'status': 'MISSING',
                'have_a': a is not None, 'have_b': b is not None}
    ti, dfi = welch(a['induction']['per_seed'], b['induction']['per_seed'])
    tc, dfc = welch(a['held_ce']['per_seed'], b['held_ce']['per_seed'])
    return {
        'label': label, 'a': a['arm'], 'b': b['arm'],
        'a_geometry': f"{a['n_slots']}x{a['slot']}",
        'b_geometry': f"{b['n_slots']}x{b['slot']}",
        'a_params': a['params'], 'b_params': b['params'],
        'induction_a': a['induction'], 'induction_b': b['induction'],
        'induction_delta': a['induction']['mean'] - b['induction']['mean'],
        'induction_ratio': (a['induction']['mean'] / b['induction']['mean']
                            if b['induction']['mean'] else None),
        'induction_welch_t': ti, 'induction_welch_df': dfi,
        'induction_separated_at_95':
            bool(ti is not None and abs(ti) > 4.303 and dfi and dfi >= 2),
        'held_ce_a': a['held_ce'], 'held_ce_b': b['held_ce'],
        'held_ce_delta': a['held_ce']['mean'] - b['held_ce']['mean'],
        'held_ce_welch_t': tc,
        'A0_into_layer1_a': a['routes'].get('A0_into_layer1_read'),
        'A0_into_layer1_b': b['routes'].get('A0_into_layer1_read'),
    }


def main():
    out = {'what': 'slot-geometry controls for the depth-3 variant slice',
           'rule': 'the depth-3 verdict for slots/shrink is a GEOMETRY '
                   'ARTIFACT if the same n_slots change reproduces the '
                   'depth-3 deficit at depth 2 (control a) and/or the '
                   'deficit disappears at width 192 where 6x32 is exact '
                   '(control b).',
           'arms': {}, 'controls': {}}

    A = out['arms']
    for key, spec in {
        # published depth-2 answers
        'slots_d2_w128_n4': ('slots', 2, 128, ''),
        'shrink_d2_w128_n4': ('shrink', 2, 128, ''),
        'vanilla_d2_w128': ('vanilla', 2, 128, ''),
        # control (a): the same geometry change at depth 2
        'slots_d2_w128_n8': ('slots', 2, 128, '_g8'),
        'shrink_d2_w128_n8': ('shrink', 2, 128, '_g8'),
        # the depth-3 cells the verdict was read from
        'vanilla_d3_w128': ('vanilla', 3, 128, ''),
        'slots_d3_w128_n8': ('slots', 3, 128, ''),
        'shrink_d3_w128_n8': ('shrink', 3, 128, ''),
        # control (b): width 192, exact 6x32
        'vanilla_d3_w192': ('vanilla', 3, 192, ''),
        'slots_d3_w192_n6': ('slots', 3, 192, ''),
        'shrink_d3_w192_n6': ('shrink', 3, 192, ''),
        'slots_d3_w192_n8': ('slots', 3, 192, '_g8'),
    }.items():
        A[key] = arm(*spec)

    C = out['controls']
    C['a_depth2_geometry_slots'] = compare(
        A['slots_d2_w128_n8'], A['slots_d2_w128_n4'],
        'CONTROL A - slots at depth 2 width 128: 8x16 vs the published 4x32. '
        'Same depth, same width, same parameters; only the slot geometry '
        'moves. If 8x16 costs induction HERE, the depth-3 slots deficit is '
        'not evidence about the architecture.')
    C['a_depth2_geometry_shrink'] = compare(
        A['shrink_d2_w128_n8'], A['shrink_d2_w128_n4'],
        'CONTROL A - shrink at depth 2 width 128: 8x16 vs the published 4x32.')
    C['b_width192_slots_vs_plain'] = compare(
        A['slots_d3_w192_n6'], A['vanilla_d3_w192'],
        'CONTROL B - depth 3 width 192, where 6 slots x 32 is EXACT: slots '
        'against the plain model at the same width. This is the depth-3 '
        'verdict re-run with the geometry the architecture wants.')
    C['b_width192_shrink_vs_plain'] = compare(
        A['shrink_d3_w192_n6'], A['vanilla_d3_w192'],
        'CONTROL B - shrink against plain at depth 3 width 192.')
    C['b2_width192_geometry_only'] = compare(
        A['slots_d3_w192_n8'], A['slots_d3_w192_n6'],
        'CONTROL B2 - the geometry contrast at FIXED width 192: 8x24 (two '
        'dead slots) against 6x32. Isolates slot geometry from width.')
    C['w128_slots_vs_plain'] = compare(
        A['slots_d3_w128_n8'], A['vanilla_d3_w128'],
        'REFERENCE - the depth-3 width-128 comparison the verdict was read '
        'from (slots 8x16 vs plain).')
    C['w128_shrink_vs_plain'] = compare(
        A['shrink_d3_w128_n8'], A['vanilla_d3_w128'],
        'REFERENCE - shrink 8x16 vs plain at depth 3 width 128.')

    # ---- the verdict the task asks for, computed not narrated -------------
    v = {}
    ca, cb = C['a_depth2_geometry_slots'], C['b_width192_slots_vs_plain']
    if ca.get('status') != 'MISSING':
        v['geometry_costs_induction_at_depth2_slots'] = {
            'published_4x32': ca['induction_b'], 'same_cell_8x16': ca['induction_a'],
            'ratio_8x16_over_4x32': ca['induction_ratio'],
            'welch_t': ca['induction_welch_t'],
            'held_ce_cost_of_8x16': ca['held_ce_delta']}
    cas = C['a_depth2_geometry_shrink']
    if cas.get('status') != 'MISSING':
        v['geometry_costs_induction_at_depth2_shrink'] = {
            'published_4x32': cas['induction_b'],
            'same_cell_8x16': cas['induction_a'],
            'ratio_8x16_over_4x32': cas['induction_ratio'],
            'welch_t': cas['induction_welch_t'],
            'held_ce_cost_of_8x16': cas['held_ce_delta']}
    if cb.get('status') != 'MISSING':
        v['depth3_exact_geometry_slots_vs_plain'] = {
            'induction_ratio': cb['induction_ratio'],
            'held_ce_delta': cb['held_ce_delta'],
            'clears_2x_bar': bool(cb['induction_ratio'] and
                                  cb['induction_ratio'] > 2.0),
            'below_0.5x_bar': bool(cb['induction_ratio'] is not None and
                                   cb['induction_ratio'] < 0.5)}
    cbs = C['b_width192_shrink_vs_plain']
    if cbs.get('status') != 'MISSING':
        v['depth3_exact_geometry_shrink_vs_plain'] = {
            'induction_ratio': cbs['induction_ratio'],
            'held_ce_delta': cbs['held_ce_delta'],
            'clears_2x_bar': bool(cbs['induction_ratio'] and
                                  cbs['induction_ratio'] > 2.0),
            'below_0.5x_bar': bool(cbs['induction_ratio'] is not None and
                                   cbs['induction_ratio'] < 0.5)}
    # ---- the registered rule re-run with the masked arms read at the
    # geometry they were designed for.  This is NOT the registered rule (it
    # mixes two widths) and is labelled as indicative only; it exists to answer
    # the one question the controls were trained for: could the geometry
    # deviation alone have flipped the verdict word?
    ratios = {}
    if A['vanilla_d3_w128'] and A['vanilla_d3_w192']:
        v128 = A['vanilla_d3_w128']['induction']['mean']
        v192 = A['vanilla_d3_w192']['induction']['mean']
        for k, arm_ in (('bandwidth', 'bandwidth_d3_w128'),
                        ('predicate', 'predicate_d3_w128'),
                        ('codebook', 'codebook_d3_w128')):
            a_ = arm(k, 3, 128)
            if a_:
                ratios[k] = a_['induction']['mean'] / v128
        for k, key in (('slots', 'slots_d3_w192_n6'),
                       ('shrink', 'shrink_d3_w192_n6')):
            if A[key]:
                ratios[k] = A[key]['induction']['mean'] / v192
        n_above = sum(x > 2.0 for x in ratios.values())
        n_below = sum(x < 0.5 for x in ratios.values())
        out['indicative_rule_with_masked_arms_at_their_own_geometry'] = {
            'caveat': 'MIXES TWO WIDTHS - the three small-decoder arms are '
                      'read at width 128 against the plain model at width 128, '
                      'the two masked-decoder arms at width 192 against the '
                      'plain model at width 192. It is not the registered rule '
                      'and must never be quoted as it; it answers only '
                      '"could the forced geometry alone have flipped the '
                      'verdict word?"',
            'ratios': ratios, 'n_above_2x': n_above, 'n_below_0.5x': n_below,
            'verdict': ('PERSISTS' if n_above >= 3 else
                        ('INVERTS' if n_below >= 3 else 'ACCELERANT')),
            'n_seeds_per_masked_arm_at_w192': {
                k: (A[key]['n_seeds'] if A[key] else 0)
                for k, key in (('slots', 'slots_d3_w192_n6'),
                               ('shrink', 'shrink_d3_w192_n6'))}}
    out['verdict_inputs'] = v
    json.dump(out, open(f'{HERE}/tf_geom_controls.json', 'w'), indent=2,
              default=str)

    # -------------------------------------------------------------- table
    L = ['# Slot-geometry controls for the depth-3 variant slice', '',
         'Same instruments as the slice (`tf_interp3.py`, '
         '`tf_depth_addendum.py`). Induction decided over MODEL seeds; route '
         'KLs quoted [zero, resample] beside the write norm share.', '',
         '| arm | seeds | n_slots x slot | params | held CE | induction ± sd '
         '| model-seed t | above own probe floor | A0 into layer-1 read '
         '[zero, resample] | A0 write norm share |',
         '|' + '---|' * 10]
    for k, a in A.items():
        if a is None:
            L.append(f'| **{k}** | MISSING | | | | | | | | |')
            continue
        r0 = a['routes'].get('A0_into_layer1_read', {})

        def g(d_, kk):
            x = d_.get(kk, {}).get('mean')
            return '—' if x is None else f'{x:.4g}'
        tt = '—' if a['model_seed_t'] is None else f"{a['model_seed_t']:.2f}"
        L.append(f"| **{k}** | {a['n_seeds']} | {a['n_slots']}x{a['slot']} | "
                 f"{a['params']:,} | {a['held_ce']['mean']:.4f} ± "
                 f"{a['held_ce']['sd']:.4f} | {a['induction']['mean']:+.4f} ± "
                 f"{a['induction']['sd']:.4f} | {tt} | "
                 f"{a['seeds_above_own_probe_floor']}/{a['n_seeds']} | "
                 f"[{g(r0,'zero')}, {g(r0,'resample')}] | "
                 f"{g(r0,'write_norm_share_of_read')} |")
    L += ['', '## Controls', '', '```',
          json.dumps({k: {kk: vv for kk, vv in c.items()
                          if kk not in ('A0_into_layer1_a',
                                        'A0_into_layer1_b')}
                      for k, c in C.items()}, indent=2, default=str),
          '```', '', '## Verdict inputs', '', '```',
          json.dumps(v, indent=2, default=str), '```']
    open(f'{HERE}/tf_geom_controls.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
