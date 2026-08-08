"""ROUND-5 INDEPENDENT REVIEW of the depth-3 six-architecture slice.

The reviewer did not produce the slice.  Every objection below is answered with
a number computed here from the checkpoints and the slice's own artifacts, not
from the write-up.

O1  decision-rule sensitivity  -- what the pre-registered 2.0x/0.5x rule
    returns at 1.5x and 3.0x, and whether the answer is stable under the seed
    spread actually measured (leave-one-seed-out over BOTH arms, and a
    seed-pairing bootstrap).
O2  parameter and compute fairness -- nominal, effective and body parameter
    counts, wall-clock seconds, and the parameter-matched plain control.
O3  the named-attention arm -- is its 25.4x handed over rather than learned?
    Zero the named scalars and re-measure induction AND held CE.
O4  claims resting on one seed / one probe / one ablation method, and every
    claim whose between-seed spread is the size of the effect.
O5  is the headline CE column reproducible from the artifacts?

Run:  python tf_reviewer_r5.py            (full, needs the GPU for O3's CE arm)
      python tf_reviewer_r5.py --no-gpu   (everything except O3's CE arm)
"""
import argparse
import glob
import json
import math
import os
import re
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ['vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink']
T_CRIT_2DF = 3.182


def cells(depth=3, width=128, suffix=''):
    out = {}
    for var in VARIANTS:
        by = {}
        for f in sorted(glob.glob(
                f'{HERE}/tf_{var}_d{depth}_w{width}_b8192_s*{suffix}'
                f'_interp3.json')):
            m = re.search(r'_s(\d+)' + re.escape(suffix) + r'_interp3\.json$', f)
            if m:
                by[int(m.group(1))] = json.load(open(f))
        if by:
            out[var] = by
    return out


def ms(v):
    v = [float(x) for x in v]
    return {'mean': float(np.mean(v)),
            'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            'per_seed': v, 'n': len(v)}


def ind(by):
    return [by[s]['rung3_induction']['induction_score_mean'] for s in sorted(by)]


def ce(by):
    return [by[s]['train']['final_held_ce'] for s in sorted(by)]


# ===================================================================== O1
def o1_rule_sensitivity(C):
    """The registered rule is: ratio of 3-seed-mean induction to the plain
    model's, >2.0 for >=3 of 5 => PERSISTS, <0.5 for >=3 of 5 => INVERTS,
    otherwise ACCELERANT.  Two things to test: the bar, and the seeds."""
    van = np.mean(ind(C['vanilla']))
    ratios = {v: float(np.mean(ind(C[v])) / van) for v in VARIANTS
              if v != 'vanilla'}

    def call(rs, hi):
        lo = 1.0 / hi
        na = sum(r > hi for r in rs.values())
        nb = sum(r < lo for r in rs.values())
        if na >= 3:
            return 'PERSISTS', na, nb
        if nb >= 3:
            return 'INVERTS', na, nb
        return 'ACCELERANT', na, nb

    bars = {}
    for hi in (1.25, 1.5, 2.0, 2.5, 3.0, 5.0):
        v, na, nb = call(ratios, hi)
        bars[f'{hi}x'] = {'verdict': v, 'n_above': na, 'n_below': nb,
                          'lower_bar': round(1.0 / hi, 4)}
    # the registered rule pairs 2.0 with 0.5, i.e. a symmetric bar; also report
    # the asymmetric registered literal (2.0 / 0.5) which is the same thing.

    # ---- seed stability 1: leave-one-seed-out of BOTH arms simultaneously
    loo = {}
    for drop in range(3):
        keep = [s for s in range(3) if s != drop]
        vk = np.mean([C['vanilla'][s]['rung3_induction']['induction_score_mean']
                      for s in keep if s in C['vanilla']])
        rr = {}
        for v in ratios:
            xs = [C[v][s]['rung3_induction']['induction_score_mean']
                  for s in keep if s in C[v]]
            rr[v] = float(np.mean(xs) / vk)
        for hi in (1.5, 2.0, 3.0):
            loo.setdefault(f'{hi}x', {})[f'drop_seed_{drop}'] = {
                'verdict': call(rr, hi)[0],
                'ratios': {k: round(x, 3) for k, x in rr.items()}}

    # ---- seed stability 2: bootstrap over seed PAIRINGS.  3 seeds per arm, so
    # enumerate every (variant seed, vanilla seed) product exactly -- 3^6 = 729
    # combinations of one seed per arm, which is a complete enumeration, not a
    # sample.
    seeds = sorted(C['vanilla'])
    verdicts = {f'{hi}x': {} for hi in (1.5, 2.0, 3.0)}
    per_variant_above = {v: {f'{hi}x': 0 for hi in (1.5, 2.0, 3.0)}
                         for v in ratios}
    n = 0
    for combo in product(seeds, repeat=6):
        vs = dict(zip(VARIANTS, combo))
        vk = C['vanilla'][vs['vanilla']]['rung3_induction']['induction_score_mean']
        rr = {v: C[v][vs[v]]['rung3_induction']['induction_score_mean'] / vk
              for v in ratios}
        n += 1
        for hi in (1.5, 2.0, 3.0):
            v0 = call(rr, hi)[0]
            verdicts[f'{hi}x'][v0] = verdicts[f'{hi}x'].get(v0, 0) + 1
            for v, r in rr.items():
                if r > hi:
                    per_variant_above[v][f'{hi}x'] += 1
    single_seed = {k: {kk: round(vv / n, 4) for kk, vv in d.items()}
                   for k, d in verdicts.items()}
    per_variant_above = {v: {k: round(c / n, 4) for k, c in d.items()}
                         for v, d in per_variant_above.items()}

    # ---- how close is each variant to the bar, in units of its own seed sd?
    margin = {}
    for v in ratios:
        a, b = np.array(ind(C[v])), np.array(ind(C['vanilla']))
        # delta method sd of the ratio
        ma, mb = a.mean(), b.mean()
        sa, sb = a.std(ddof=1) / math.sqrt(len(a)), b.std(ddof=1) / math.sqrt(len(b))
        sr = abs(ma / mb) * math.sqrt((sa / ma) ** 2 + (sb / mb) ** 2)
        margin[v] = {'ratio': float(ma / mb), 'ratio_se_delta_method': float(sr),
                     'z_above_2x': float((ma / mb - 2.0) / sr) if sr else None,
                     'z_above_1.5x': float((ma / mb - 1.5) / sr) if sr else None,
                     'z_above_3x': float((ma / mb - 3.0) / sr) if sr else None,
                     'ci95_delta': [float(ma / mb - 1.96 * sr),
                                    float(ma / mb + 1.96 * sr)]}
    return {'ratios_3seed_mean': ratios,
            'verdict_by_bar': bars,
            'leave_one_seed_out': loo,
            'complete_single_seed_enumeration_729': single_seed,
            'fraction_of_seed_combinations_this_variant_clears_the_bar':
                per_variant_above,
            'ratio_margins': margin,
            'reading': 'the verdict word is stable only if every bar and every '
                       'seed subset gives it'}


# ===================================================================== O2
def o2_parameters(C):
    rows = {}
    for v in VARIANTS:
        d0 = C[v][sorted(C[v])[0]]
        p = d0['params']
        rows[v] = {
            'nominal_total': p['total'],
            'body': p['body'],
            'embedding': p.get('embedding', p['total'] - p['body']),
            'effective_total': p.get('effective_total', p['total']),
            'codebook_buffer_floats': p.get('codebook_buffer_floats', 0),
            'stream_width': p.get('stream_width'),
            'wall_seconds': ms([C[v][s]['train']['wall_seconds']
                                for s in sorted(C[v])]),
            'held_ce': ms(ce(C[v])), 'induction': ms(ind(C[v]))}
    base = rows['vanilla']['nominal_total']
    for v, r in rows.items():
        r['nominal_vs_plain'] = round(r['nominal_total'] / base, 4)
        r['effective_vs_plain'] = round(r['effective_total']
                                        / rows['vanilla']['effective_total'], 4)
    # the parameter-matched plain control, if it has landed
    matched = {}
    for w in (144,):
        by = {}
        for f in sorted(glob.glob(
                f'{HERE}/tf_vanilla_d3_w{w}_b8192_s*_interp3.json')):
            m = re.search(r'_s(\d+)_interp3\.json$', f)
            if m:
                by[int(m.group(1))] = json.load(open(f))
        if by:
            d0 = by[sorted(by)[0]]
            matched[f'vanilla_d3_w{w}'] = {
                'n_seeds': len(by), 'params': d0['params']['total'],
                'held_ce': ms(ce(by)), 'induction': ms(ind(by)),
                'wall_seconds': ms([by[s]['train']['wall_seconds']
                                    for s in sorted(by)])}
    out = {'per_variant': rows, 'parameter_matched_plain_control': matched}
    if matched:
        mc = list(matched.values())[0]
        out['does_size_explain_it'] = {}
        for v in ('predicate', 'bandwidth', 'codebook'):
            out['does_size_explain_it'][v] = {
                'variant_params': rows[v]['nominal_total'],
                'matched_plain_params': mc['params'],
                'plain_has_more_parameters': mc['params'] > rows[v]['nominal_total'],
                'held_ce_variant_minus_matched_plain':
                    rows[v]['held_ce']['mean'] - mc['held_ce']['mean'],
                'induction_ratio_to_matched_plain':
                    rows[v]['induction']['mean'] / mc['induction']['mean'],
                'induction_ratio_to_w128_plain':
                    rows[v]['induction']['mean'] / rows['vanilla']['induction']['mean']}
    return out


# ===================================================================== O3
def o3_named_terms(C, do_ce=True):
    """Is the named-attention arm's induction INSTALLED rather than learned?

    The slice already ran `predicate_induction_split` at all three seeds: it
    zeroes the named scalars IN PLACE and re-runs the identical battery.  That
    is read out here rather than re-derived.  What is NEW is the loss arm: if
    the capability is installed, the CE win should also collapse when the same
    scalars are zeroed -- and if it does not, the CE win and the induction win
    are two different claims."""
    out = {'source': 'predicate_induction_split in each seed\'s _interp3.json '
                     '(zeroes pred_b / pred_c / pred_prof in place, restores '
                     'them, and re-runs the identical induction battery)'}
    by = C['predicate']
    seeds = sorted(by)
    sp = {s: by[s]['predicate_induction_split'] for s in seeds}
    arms = ['all_named_terms_on', 'zero_prev_token_match_b',
            'zero_same_token_match_c', 'zero_positional_profile',
            'zero_all_named_terms']
    out['induction'] = {a: ms([sp[s][a]['induction_score_mean'] for s in seeds])
                        for a in arms}
    out['fraction_removed'] = {
        a: ms([sp[s]['fraction_removed'][a] for s in seeds]) for a in arms}
    out['per_layer_zero_prev_match'] = {
        f'layer{li}': ms([sp[s][f'zero_prev_token_match_b_layer{li}']
                          ['induction_score_mean'] for s in seeds])
        for li in range(3)
        if all(f'zero_prev_token_match_b_layer{li}' in sp[s] for s in seeds)}
    out['per_head_layer0_zero_prev_match'] = {
        h: ms([sp[s]['zero_prev_match_one_head_at_a_time_layer0'][h]
               for s in seeds])
        for h in sorted(sp[seeds[0]]['zero_prev_match_one_head_at_a_time_layer0'])}
    # the two comparators
    out['comparators'] = {
        'plain_d3_w128_induction': ms(ind(C['vanilla'])),
        'plain_d2_w128_null': {'mean': -0.0034, 'sd': 0.0099,
                               'source': 'RESULTS.md FINDING 11 Table B'},
        'predicate_own_probe_floor': ms(
            [by[s]['induction_power']['detectable_effect_floor_nats_3se']
             for s in seeds])}
    z = out['induction']['zero_all_named_terms']
    p = out['comparators']['plain_d3_w128_induction']
    se = math.sqrt((z['sd'] ** 2 + p['sd'] ** 2) / 3)
    out['with_named_terms_off_vs_plain_d3'] = {
        'delta': z['mean'] - p['mean'],
        'welch_t': (z['mean'] - p['mean']) / se if se else None,
        'named_off_clears_own_probe_floor':
            z['mean'] > out['comparators']['predicate_own_probe_floor']['mean'],
        'reading': 'if the named-off score is at or below zero while the plain '
                   'model at the same cell is clearly positive, the arm does '
                   'not merely INSTALL the capability on top of a learned one '
                   '- the rest of its network learned LESS induction than the '
                   'plain model did'}
    if do_ce:
        out['held_ce_with_named_terms_zeroed'] = named_off_ce(seeds)
    return out


def named_off_ce(seeds):
    """Held CE of the depth-3 predicate cells with the named scalars zeroed,
    measured with `tf_train.eval_held` -- the same function that produced every
    `final_held_ce` in the programme, on the same held split."""
    import torch
    import tf_fold
    import tf_train
    corpus = tf_train.Corpus(8192, 'bpe')
    out = {}
    for s in seeds:
        stem = f'tf_predicate_d3_w128_b8192_s{s}'
        model, cfg, ck = tf_fold.load_checkpoint(stem, tf_train.DEV)
        model.requires_grad_(False)
        P = {'b': model.pred_b.data, 'c': model.pred_c.data,
             'prof': model.pred_prof.data}
        saved = {k: v.detach().clone() for k, v in P.items()}
        row = {}
        row['all_named_terms_on'] = tf_train.eval_held(model, corpus)[0]
        try:
            for keys, tag in ((('b',), 'zero_prev_token_match_b'),
                              (('b', 'c', 'prof'), 'zero_all_named_terms')):
                for k in keys:
                    P[k].zero_()
                row[tag] = tf_train.eval_held(model, corpus)[0]
                for k in keys:
                    P[k].copy_(saved[k])
        finally:
            for k, v in saved.items():
                P[k].copy_(v)
        row['gate_restored_matches_original'] = abs(
            tf_train.eval_held(model, corpus)[0] - row['all_named_terms_on']) < 1e-9
        out[f'seed{s}'] = row
        del model
        if tf_train.DEV == 'cuda':
            torch.cuda.empty_cache()
    agg = {}
    for k in ('all_named_terms_on', 'zero_prev_token_match_b',
              'zero_all_named_terms'):
        agg[k] = ms([out[f'seed{s}'][k] for s in seeds])
    out['aggregate'] = agg
    out['note'] = ('this is an INFERENCE-TIME knockout of a trained model, not '
                   'a retraining: it says how much of the trained model\'s '
                   'behaviour the named terms carry, not what the same '
                   'architecture would reach without them.')
    return out


# ===================================================================== O4
def o4_fragile_claims(C):
    """Every headline quantity, with its between-seed spread beside it, and a
    flag when the spread is the size of the effect."""
    rows = {}
    van_i = np.array(ind(C['vanilla']))
    for v in VARIANTS:
        d = {'induction': ms(ind(C[v])), 'held_ce': ms(ce(C[v]))}
        i = d['induction']
        d['induction_sd_over_mean'] = (i['sd'] / abs(i['mean'])
                                       if i['mean'] else None)
        d['induction_model_seed_t'] = (i['mean'] / (i['sd'] / math.sqrt(i['n']))
                                       if i['sd'] else None)
        if v != 'vanilla':
            a = np.array(ind(C[v]))
            va = a.var(ddof=1) / 3 + van_i.var(ddof=1) / 3
            d['vs_plain_welch_t'] = float((a.mean() - van_i.mean())
                                          / math.sqrt(va)) if va else None
            d['seed_ranges_overlap_plain'] = bool(
                a.min() <= van_i.max() and van_i.min() <= a.max())
        # route numbers, per seed
        for li, src in ((1, 'A0'), (2, 'A1')):
            zs, rs = [], []
            for s in sorted(C[v]):
                rk = C[v][s]['read_ablation_causal']['kl_from_model']
                if f'l{li}_read_zero_{src}' in rk:
                    zs.append(rk[f'l{li}_read_zero_{src}'])
                    rs.append(rk[f'l{li}_read_resample_{src}'])
            if zs:
                d[f'{src}_into_l{li}_zero'] = ms(zs)
                d[f'{src}_into_l{li}_resample'] = ms(rs)
                d[f'{src}_into_l{li}_zero_over_resample'] = (
                    float(np.mean(zs) / np.mean(rs)) if np.mean(rs) else None)
        # route-USE, per seed
        ru = []
        for s in sorted(C[v]):
            f = f'{HERE}/tf_{v}_d3_w128_b8192_s{s}_routeuse.json'
            if os.path.exists(f):
                ru.append(json.load(open(f)))
        if ru:
            keys = sorted(set().union(*[set(j['fraction_of_induction_removed'])
                                        for j in ru]))
            d['route_use'] = {k: ms([j['fraction_of_induction_removed'][k]
                                     for j in ru
                                     if k in j['fraction_of_induction_removed']])
                              for k in keys}
            d['route_use_baseline_induction'] = ms(
                [j['baseline_induction'] for j in ru])
        rows[v] = d
    flags = []
    for v, d in rows.items():
        i = d['induction']
        if i['sd'] >= abs(i['mean']):
            flags.append(f'{v}: induction between-seed sd ({i["sd"]:.4f}) is '
                         f'at least the size of the effect ({i["mean"]:+.4f})')
        if d.get('vs_plain_welch_t') is not None \
                and abs(d['vs_plain_welch_t']) < 4.303:
            flags.append(
                f'{v}: induction vs plain is NOT separated at 95% over model '
                f'seeds (Welch t = {d["vs_plain_welch_t"]:.2f}, needs 4.30 at '
                f'2 df) - the ratio quoted for it is a point estimate only')
        for li, src in ((1, 'A0'), (2, 'A1')):
            k = f'{src}_into_l{li}_zero'
            if k in d and d[k]['sd'] >= 0.5 * abs(d[k]['mean']):
                flags.append(f'{v}: {k} between-seed sd is >= half the mean '
                             f'({d[k]["mean"]:.4g} +- {d[k]["sd"]:.4g})')
        for k, r in d.get('route_use', {}).items():
            if r['sd'] >= abs(r['mean']) * 0.9 and abs(r['mean']) > 1e-3:
                flags.append(
                    f'{v}: route-USE {k} between-seed sd is the size of the '
                    f'effect ({r["mean"]:.3f} +- {r["sd"]:.3f}) - not a '
                    f'quotable fraction at three seeds')
        rub = d.get('route_use_baseline_induction')
        if rub and 'route_use' in d:
            fl = ms([C[v][s]['induction_power']
                     ['detectable_effect_floor_nats_3se']
                     for s in sorted(C[v])])['mean']
            if rub['mean'] < 3 * fl:
                flags.append(
                    f'{v}: route-USE fractions are ratios on a baseline '
                    f'induction of {rub["mean"]:.4f} against a probe floor of '
                    f'{fl:.4f} - a small-denominator ratio, do not read it')
    return {'per_variant': rows, 'fragility_flags': flags}


# ===================================================================== O5
def o5_headline_reproducibility(C):
    """The mailbox / commit headline quotes a held-CE column. Does it match the
    artifacts the report generator reads?"""
    quoted = {'predicate': 4.3046, 'bandwidth': 4.4654, 'vanilla': 4.4481,
              'codebook': 4.5762, 'shrink': 4.6204, 'slots': 4.6488}
    got = {v: float(np.mean(ce(C[v]))) for v in VARIANTS}
    rows = {v: {'quoted_in_mailbox_and_commit': quoted[v],
                'final_held_ce_in_artifacts': round(got[v], 5),
                'delta': round(quoted[v] - got[v], 5)} for v in VARIANTS}
    return {'per_variant': rows,
            'quoted_predicate_minus_vanilla': round(
                quoted['predicate'] - quoted['vanilla'], 4),
            'artifact_predicate_minus_vanilla': round(
                got['predicate'] - got['vanilla'], 4),
            'all_deltas_equal': len({r['delta'] for r in rows.values()}) == 1,
            'reading': 'a constant offset would mean a different but valid '
                       'evaluation; unequal offsets mean the column cannot be '
                       'reconstructed from any single artifact field'}


# ===================================================================== O6
def o6_ablation_method(C):
    """Every route number in the slice is quoted [zero, resample].  How much of
    the headline depends on which one you read?  The programme's own record
    (README) says resample is the HARSHER ablation at 13 of 14 layer-cells, so
    a variant whose zero number is many times its resample number is showing
    something about zeroing a private slot -- whose per-slot RMSNorm then
    renormalises a zero vector -- not about how much the route carries."""
    out = {'per_variant': {}}
    for v in VARIANTS:
        row = {}
        for li, src in ((1, 'A0'), (2, 'A1')):
            zs, rs = [], []
            for s in sorted(C[v]):
                rk = C[v][s]['read_ablation_causal']['kl_from_model']
                if f'l{li}_read_zero_{src}' in rk:
                    zs.append(rk[f'l{li}_read_zero_{src}'])
                    rs.append(rk[f'l{li}_read_resample_{src}'])
            if zs:
                z, r = float(np.mean(zs)), float(np.mean(rs))
                row[f'{src}_into_l{li}'] = {
                    'zero': z, 'resample': r,
                    'zero_over_resample': (z / r) if r else None,
                    'resample_is_harsher': r > z}
        out['per_variant'][v] = row
    # PD3's verdict recomputed on the resample number instead of zeroing
    pd3 = {}
    for v in VARIANTS:
        r = out['per_variant'][v].get('A0_into_l1')
        if r:
            pd3[v] = r['resample']
    n = sum(1 for k, x in pd3.items() if k != 'vanilla' and x >= 0.05)
    out['PD3_recomputed_on_resample'] = {
        'A0_into_layer1_resample_kl': pd3,
        'n_variants_at_or_above_0.05_nats': n,
        'call': 'CONFIRMED' if n >= 4 else ('REFUTED' if n <= 2 else 'PARTIAL'),
        'note': 'PD3 was registered on the zeroing number; it survives on the '
                'resample number too, but the magnitudes are 1.0-9.2x smaller'}
    return out


# ===================================================================== O7
def o7_norm_share_regression(C):
    """THE ROUND-4 RULE, APPLIED TO THE VARIANTS AT LAST.

    `tf_reviewer_round_4.json` O2b established on 243 plain-model write/read
    pairs that the read-ablation KL is a quadratic function of the write's own
    norm share of the read it enters (slope 1.99, r = 0.994) -- i.e. nothing in
    the plain model gates a direction.  The round-4 review said explicitly that
    the regression MUST be re-derived on variant checkpoints before it is
    applied to them, and the depth-3 handoff called that the first analysis to
    run once the cells landed.  It was not run.  It is run here.

    If the variants fall on the same line, PD3 ('the variants use a channel the
    plain model leaves empty') is a statement about how big their first
    attention write is and nothing more.  If their layer-0 attention pairs sit
    systematically ABOVE the line, they transmit more than their size predicts
    and the routing language would be earned."""
    rows = []
    for v in VARIANTS:
        for s in sorted(C[v]):
            d = C[v][s]
            rk = d['read_ablation_causal']['kl_from_model']
            sg = d.get('stream_geometry')
            if not sg:
                continue
            for l in range(1, d['depth']):
                srcs = (['e'] + [f'A{j}' for j in range(l)]
                        + [f'M{j}' for j in range(l)])
                den = math.sqrt(sum(sg[f'{k}_norm'] ** 2 for k in srcs))
                for k in srcs:
                    kl = max(rk.get(f'l{l}_read_zero_{k}', 0.0),
                             rk.get(f'l{l}_read_resample_{k}', 0.0))
                    if kl <= 0 or den <= 0:
                        continue
                    rows.append({'variant': v, 'seed': s, 'read': l,
                                 'source': k, 'kl': kl,
                                 'norm_share_of_read': sg[f'{k}_norm'] / den})
    x = np.log10([r['norm_share_of_read'] for r in rows])
    y = np.log10([r['kl'] for r in rows])
    sl, ic = np.polyfit(x, y, 1)
    res = y - (sl * x + ic)
    by = {}
    for key in ('variant', 'source'):
        agg = {}
        for r_, e_ in zip(rows, res):
            agg.setdefault(r_[key], []).append(float(e_))
        by[key] = {k: {'mean_residual_dex': float(np.mean(vv)),
                       'sd_dex': float(np.std(vv)), 'n': len(vv)}
                   for k, vv in sorted(agg.items())}
    # the plain model alone, fitted on the same variables, as the comparator
    pr = [(a, b) for a, b, r_ in zip(x, y, rows) if r_['variant'] == 'vanilla']
    psl, pic = np.polyfit([a for a, _ in pr], [b for _, b in pr], 1)
    # where do the variants' layer-0 attention pairs sit on the PLAIN line?
    off = {}
    for r_, xx, yy in zip(rows, x, y):
        if r_['source'] == 'A0':
            off.setdefault(r_['variant'], []).append(float(yy - (psl * xx + pic)))
    # conditioning: how much dynamic range does each arm actually span?
    cond = {}
    for v in VARIANTS:
        xs = [r_['norm_share_of_read'] for r_ in rows if r_['variant'] == v]
        ys = [r_['kl'] for r_ in rows if r_['variant'] == v]
        if xs:
            lx, ly = np.log10(xs), np.log10(ys)
            s_, i_ = np.polyfit(lx, ly, 1)
            cond[v] = {'min_norm_share': float(min(xs)),
                       'max_norm_share': float(max(xs)),
                       'log10_range_dex': float(np.log10(max(xs) / min(xs))),
                       'own_slope': float(s_),
                       'own_pearson_r': float(np.corrcoef(lx, ly)[0, 1]),
                       'own_residual_sd_dex': float(
                           (ly - (s_ * lx + i_)).std(ddof=2)),
                       'n': len(xs)}
    return {
        'n_pairs': len(rows),
        'norm_share_dynamic_range_by_variant': cond,
        'pooled_fit_over_all_six_variants': {
            'pearson_r': float(np.corrcoef(x, y)[0, 1]),
            'slope_log_kl_per_log_norm_share': float(sl),
            'intercept': float(ic),
            'residual_sd_dex': float(res.std(ddof=2))},
        'plain_model_only_fit': {'slope': float(psl), 'intercept': float(pic),
                                 'n': len(pr)},
        'residual_by_variant': by['variant'],
        'residual_by_source': by['source'],
        'A0_offset_from_the_PLAIN_line_dex': {
            k: {'mean': float(np.mean(vv)), 'sd': float(np.std(vv)),
                'n': len(vv)} for k, vv in sorted(off.items())},
        'round4_plain_model_reference': {
            'slope': 1.992, 'pearson_r': 0.9944, 'residual_sd_dex': 0.264,
            'source': 'tf_reviewer_round_4.json O2b, 243 pairs, plain model, '
                      'depths 2-4, widths 64-256'},
        'reading': 'a slope near 2 with a small residual means the variants '
                   'obey the same magnitude law as the plain model and PD3 is '
                   'a size statement, not a routing one; a positive A0 offset '
                   'from the PLAIN line would mean the variants transmit more '
                   'than their write size predicts.',
        'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-gpu', action='store_true')
    a = ap.parse_args()
    C = cells()
    missing = [v for v in VARIANTS if v not in C or len(C[v]) < 3]
    rep = {'what': 'independent round-5 review of the depth-3 six-architecture '
                   'slice, by a reviewer who did not produce it',
           'cells_read': {v: sorted(C.get(v, {})) for v in VARIANTS},
           'incomplete_arms': missing,
           'O1_decision_rule_sensitivity': o1_rule_sensitivity(C),
           'O2_parameter_and_compute_fairness': o2_parameters(C),
           'O3_named_attention_installed_or_learned': o3_named_terms(
               C, do_ce=not a.no_gpu),
           'O4_fragile_claims': o4_fragile_claims(C),
           'O5_headline_reproducibility': o5_headline_reproducibility(C),
           'O6_ablation_method_dependence': o6_ablation_method(C),
           'O7_norm_share_regression_on_variants':
               o7_norm_share_regression(C)}
    json.dump(rep, open(f'{HERE}/tf_reviewer_r5_measurements.json', 'w'),
              indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('O4_fragile_claims',)}, indent=2,
                     default=str)[:12000])
    print('\nFRAGILITY FLAGS')
    for f in rep['O4_fragile_claims']['fragility_flags']:
        print(' -', f)


if __name__ == '__main__':
    main()
