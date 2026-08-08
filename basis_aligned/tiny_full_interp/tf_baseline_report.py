"""THE FOLDABILITY TAX: score the conventional softmax+GELU baseline against
the no-softmax bilinear family, cell by cell, against the predictions
registered in tf_baseline_predictions.json BEFORE the first training step.

Reads whatever is on disk and reports it, so it can be run while the chain is
still going; every table marks how many seeds each number rests on, and any
cell below two seeds is labelled PROVISIONAL.

Outputs  tf_baseline_std.json  and  tf_baseline_table.md
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = (1, 2, 3)
WIDTHS = (64, 128, 256)
MODEL_SEED_T_BAR = 4.30          # the bar RESULTS FINDING 16 used


def _mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return (float(np.mean(v)),
            float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def welch(a, b):
    """Welch t for mean(a) - mean(b); None when either side has < 2 values."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    s = math.sqrt(va + vb)
    return None if s == 0 else float((np.mean(a) - np.mean(b)) / s)


def one_sample_t(a):
    if len(a) < 2:
        return None
    s = np.std(a, ddof=1) / math.sqrt(len(a))
    return None if s == 0 else float(np.mean(a) / s)


# ------------------------------------------------------------------ loading
def family_cells():
    out = {}
    for f in sorted(glob.glob(f'{HERE}/tf_vanilla_d*_w*_b8192_s*.json')):
        m = re.search(r'tf_vanilla_d(\d+)_w(\d+)_b8192_s(\d+)\.json$', f)
        if not m:
            continue
        d, w, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        j = json.load(open(f))
        ce = j.get('run', {}).get('final_held_ce')
        if ce is None:
            continue
        rec = {'seed': s, 'held_ce': ce,
               'n_params': j['run'].get('n_params'),
               'body_params': j['run'].get('body_params'),
               'lr': j['run'].get('lr_muon')}
        ip = f'{HERE}/tf_vanilla_d{d}_w{w}_b8192_s{s}_interp3.json'
        if os.path.exists(ip):
            k = json.load(open(ip))
            r3 = k.get('rung3_induction', {})
            rec['induction'] = r3.get('induction_score_mean')
            rec['bag'] = r3.get('bag_score_mean')
            rec['probe_floor'] = k.get('induction_power', {}).get(
                'detectable_effect_floor_nats_3se')
            ni = k.get('natural_induction', {}).get('ORDER_ONLY_patch_swap', {})
            rec['natural_swap'] = ni.get('mean')
        out.setdefault((d, w), []).append(rec)
    return out


def std_cells():
    out = {}
    for f in sorted(glob.glob(f'{HERE}/tfb_std*_d*_w*_b8192_s*.json')):
        base = os.path.basename(f)[:-5]
        if base.endswith('_induction') or base.endswith('_control'):
            continue
        m = re.match(r'tfb_std(\d+)_d(\d+)_w(\d+)_b8192_s(\d+)(.*)$', base)
        if not m:
            continue
        e, d, w, s, suf = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           int(m.group(4)), m.group(5))
        j = json.load(open(f))
        ce = j.get('run', {}).get('final_held_ce')
        if ce is None:
            continue
        rec = {'seed': s, 'exp': e, 'suffix': suf, 'stem': base, 'held_ce': ce,
               'n_params': j['run'].get('n_params'),
               'body_params': j['run'].get('body_params'),
               'lr': j['run'].get('lr_muon'),
               'param_matched': j.get('params', {}).get(
                   'exactly_parameter_matched')}
        ip = f'{HERE}/{base}_induction.json'
        if os.path.exists(ip):
            k = json.load(open(ip))
            rec['induction'] = k.get('induction_score_mean')
            rec['bag'] = k.get('bag_score_mean')
            rec['probe_floor'] = k.get('detectable_effect_floor_nats_3se')
            rec['natural_swap'] = k.get('natural_induction', {}).get(
                'ORDER_ONLY_patch_swap', {}).get('mean')
        out.setdefault((d, w, e, suf), []).append(rec)
    return out


# ------------------------------------------------------------------ report
def build():
    pred = json.load(open(f'{HERE}/tf_baseline_predictions.json'))
    fam, std = family_cells(), std_cells()
    rep = {'registered_predictions': pred,
           'note': 'positive gap = the conventional model wins = the fold '
                   'costs prediction quality. All CE at T=512 on held rows '
                   '[0:1500], the training-protocol number.',
           'controls': {}, 'cells': {}, 'lr_arms': {}, 'qk_norm_arm': {}}
    for nm, f in (('model_and_harness', 'tf_baseline_controls.json'),
                  ('probe_shim', 'tf_baseline_probe_control.json')):
        p = f'{HERE}/{f}'
        if os.path.exists(p):
            c = json.load(open(p))
            rep['controls'][nm] = {k: v for k, v in c.items()
                                   if k != 'C2_parameter_identity'} \
                if nm == 'model_and_harness' else c
            if nm == 'model_and_harness':
                rep['controls'][nm]['C2_parameter_identity_pass'] = \
                    c.get('C2_parameter_identity', {}).get('pass')

    for d in DEPTHS:
        for w in WIDTHS:
            fr = sorted(fam.get((d, w), []), key=lambda r: r['seed'])
            cell = {'depth': d, 'width': w, 'family': {}, 'arms': {}}
            fce = [r['held_ce'] for r in fr]
            m_, s_, n_ = _mean_sd(fce)
            cell['family'] = {
                'n_seeds': n_, 'held_ce': m_, 'held_ce_sd': s_,
                'per_seed': {r['seed']: r['held_ce'] for r in fr},
                'n_params': fr[0]['n_params'] if fr else None,
                'body_params': fr[0]['body_params'] if fr else None,
                'induction': _mean_sd([r.get('induction') for r in fr])[0],
                'induction_sd': _mean_sd([r.get('induction') for r in fr])[1],
                'induction_per_seed': [r.get('induction') for r in fr],
                'induction_model_seed_t': one_sample_t(
                    [r['induction'] for r in fr if r.get('induction')
                     is not None]),
                'bag': _mean_sd([r.get('bag') for r in fr])[0],
                'natural_swap': _mean_sd([r.get('natural_swap')
                                          for r in fr])[0]}
            _fi = [r['induction'] for r in fr if r.get('induction') is not None]
            cell['family']['inducts'] = bool(
                (one_sample_t(_fi) or 0) > MODEL_SEED_T_BAR
                and (_mean_sd(_fi)[0] or 0) > 0)
            for e in (4, 7):
                sr = sorted(std.get((d, w, e, ''), []), key=lambda r: r['seed'])
                if not sr:
                    continue
                sce = [r['held_ce'] for r in sr]
                mm, ss, nn = _mean_sd(sce)
                ind = [r['induction'] for r in sr if r.get('induction')
                       is not None]
                arm = {
                    'expansion': e,
                    'label': ('nominal 4x (conventional model is the SMALLER '
                              'one)' if e == 4 else
                              'matched 7x (total parameter count identical)'),
                    'n_seeds': nn, 'held_ce': mm, 'held_ce_sd': ss,
                    'per_seed': {r['seed']: r['held_ce'] for r in sr},
                    'n_params': sr[0]['n_params'],
                    'body_params': sr[0]['body_params'],
                    'param_matched': sr[0].get('param_matched'),
                    'induction': _mean_sd(ind)[0], 'induction_sd': _mean_sd(ind)[1],
                    'induction_per_seed': [r.get('induction') for r in sr],
                    'induction_model_seed_t': one_sample_t(ind),
                    'induction_n_seeds': len(ind),
                    'bag': _mean_sd([r.get('bag') for r in sr])[0],
                    'natural_swap': _mean_sd([r.get('natural_swap')
                                              for r in sr])[0],
                    'probe_floor': _mean_sd([r.get('probe_floor')
                                             for r in sr])[0],
                    'seeds_clearing_own_probe_floor': sum(
                        1 for r in sr if r.get('induction') is not None
                        and r.get('probe_floor') is not None
                        and r['induction'] > r['probe_floor'])}
                if m_ is not None and mm is not None:
                    common = sorted(set(cell['family']['per_seed'])
                                    & set(arm['per_seed']))
                    paired = [cell['family']['per_seed'][s]
                              - arm['per_seed'][s] for s in common]
                    arm['gap_nats'] = m_ - mm
                    arm['gap_paired_by_seed'] = {
                        'seeds': common, 'per_seed': paired,
                        'mean': float(np.mean(paired)) if paired else None,
                        't': one_sample_t(paired)}
                    arm['gap_welch_t'] = welch(fce, sce)
                    arm['param_ratio_std_over_family'] = (
                        arm['n_params'] / cell['family']['n_params']
                        if cell['family']['n_params'] else None)
                    arm['provisional'] = bool(nn < 2 or n_ < 2)
                cell['arms'][f'x{e}'] = arm
            # induction: does the conventional model induct where the family
            # does not?  Decided over MODEL seeds.
            fi = [r['induction'] for r in fr if r.get('induction') is not None]
            cell['induction_verdict'] = {}
            for k, a in cell['arms'].items():
                ai = [x for x in a['induction_per_seed'] if x is not None]
                cell['induction_verdict'][k] = {
                    'family_mean': _mean_sd(fi)[0],
                    'family_model_seed_t': one_sample_t(fi),
                    'family_inducts': bool(
                        (one_sample_t(fi) or 0) > MODEL_SEED_T_BAR
                        and (_mean_sd(fi)[0] or 0) > 0),
                    'conventional_mean': _mean_sd(ai)[0],
                    'conventional_model_seed_t': one_sample_t(ai),
                    'conventional_inducts': bool(
                        (one_sample_t(ai) or 0) > MODEL_SEED_T_BAR
                        and (_mean_sd(ai)[0] or 0) > 0),
                    'welch_t_conventional_minus_family': welch(ai, fi),
                    'n_seeds': [len(fi), len(ai)]}
            rep['cells'][f'd{d}_w{w}'] = cell

    # ------------------------------------------------- learning-rate arms
    for (d, w, e, suf), rs in sorted(std.items()):
        m = re.match(r'_lr([0-9.]+)$', suf)
        if not m:
            continue
        rep['lr_arms'].setdefault(f'd{d}_w{w}_x{e}', {})[m.group(1)] = {
            'held_ce': _mean_sd([r['held_ce'] for r in rs])[0],
            'n_seeds': len(rs)}
    for k, v in rep['lr_arms'].items():
        d, w, e = re.match(r'd(\d+)_w(\d+)_x(\d+)', k).groups()
        base = rep['cells'].get(f'd{d}_w{w}', {}).get('arms', {}).get(f'x{e}')
        if base and base.get('held_ce') is not None:
            v['0.02'] = {'held_ce': base['held_ce'], 'n_seeds': base['n_seeds'],
                         'note': 'the primary arm'}
        got = {kk: vv['held_ce'] for kk, vv in v.items()
               if isinstance(vv, dict) and vv.get('held_ce') is not None}
        if got:
            best = min(got, key=got.get)
            v['best_lr'] = best
            v['best_held_ce'] = got[best]
            v['gain_over_0.02'] = (got.get('0.02', got[best]) - got[best])

    # ------------------------------------------------- QK-norm control
    for (d, w, e, suf), rs in sorted(std.items()):
        if 'noqknorm' not in suf:
            continue
        base = rep['cells'].get(f'd{d}_w{w}', {}).get('arms', {}).get(f'x{e}')
        mm = _mean_sd([r['held_ce'] for r in rs])
        rep['qk_norm_arm'][f'd{d}_w{w}_x{e}'] = {
            'no_qk_norm_held_ce': mm[0], 'no_qk_norm_sd': mm[1],
            'n_seeds': mm[2],
            'with_qk_norm_held_ce': base['held_ce'] if base else None,
            'cost_of_removing_qk_norm':
                (mm[0] - base['held_ce']) if (base and mm[0] is not None
                                              and base.get('held_ce')) else None,
            'induction_no_qk_norm': _mean_sd([r.get('induction')
                                              for r in rs])[0],
            'claim': 'per-head query/key RMSNorm is shared with the family and '
                     'is therefore kept in the conventional model; this arm '
                     'prices that choice.'}

    rep['verdicts'] = verdicts(rep)
    json.dump(rep, open(f'{HERE}/tf_baseline_std.json', 'w'), indent=2)
    return rep


def verdicts(rep):
    C = rep['cells']

    def gaps(arm):
        return {k: v['arms'][arm]['gap_nats'] for k, v in C.items()
                if arm in v['arms'] and v['arms'][arm].get('gap_nats')
                is not None}

    g4, g7 = gaps('x4'), gaps('x7')
    v = {}
    v['L1_conventional_wins_every_cell_by_0.05_to_0.20'] = {
        'registered': rep['registered_predictions']['logans_registered_'
                                                    'predictions']['L1'],
        'nominal_gaps': g4,
        'cells_won_by_conventional': sum(1 for x in g4.values() if x > 0),
        'cells_measured': len(g4),
        'cells_in_the_0.05_to_0.20_band': sum(1 for x in g4.values()
                                              if 0.05 <= x <= 0.20),
        'verdict': (None if not g4 else
                    'CONFIRMED' if (all(x > 0 for x in g4.values())
                                    and all(0.05 <= x <= 0.20
                                            for x in g4.values()))
                    else 'PARTLY CONFIRMED' if all(x > 0 for x in g4.values())
                    else 'REFUTED')}
    by_depth4 = {}
    for k, x in g4.items():
        by_depth4.setdefault(int(k[1]), []).append(x)
    md = {d: float(np.mean(xs)) for d, xs in sorted(by_depth4.items())}
    v['L2_A2_gap_grows_with_depth'] = {
        'registered': rep['registered_predictions']['logans_registered_'
                                                    'predictions']['L2'],
        'mean_nominal_gap_by_depth': md,
        'monotone_increasing': (None if len(md) < 2 else
                                all(b > a for a, b in zip(list(md.values()),
                                                          list(md.values())[1:]))),
        'deepest_minus_shallowest': (None if len(md) < 2 else
                                     list(md.values())[-1] - list(md.values())[0]),
        'verdict': (None if len(md) < 2 else
                    'CONFIRMED' if list(md.values())[-1] > list(md.values())[0]
                    else 'REFUTED')}
    both = {k: (g4[k], g7[k]) for k in g4 if k in g7}
    delta = {k: b - a for k, (a, b) in both.items()}
    v['L3_A3_matched_parameters'] = {
        'registered_L3': rep['registered_predictions']['logans_registered_'
                                                       'predictions']['L3'],
        'registered_A3': rep['registered_predictions']['analysts_registered_'
                                                       'predictions']
        ['A3_disagrees_with_L3_on_the_SIGN'],
        'nominal_gap': g4, 'matched_gap': g7,
        'matched_minus_nominal': delta,
        'mean_matched_minus_nominal': (float(np.mean(list(delta.values())))
                                       if delta else None),
        'direction_note': 'the family is the LARGER model at nominal '
                          'expansion, so matching parameters makes the '
                          'CONVENTIONAL model bigger and should widen the gap.',
        'L3_verdict': (None if not delta else
                       'CONFIRMED' if float(np.mean(list(delta.values()))) < 0
                       and abs(float(np.mean(list(delta.values()))))
                       >= abs(float(np.mean(list(g4.values())))) / 3
                       else 'REFUTED'),
        'A3_verdict': (None if not delta else
                       'CONFIRMED' if all(x > 0 for x in delta.values())
                       else 'PARTLY CONFIRMED'
                       if float(np.mean(list(delta.values()))) > 0
                       else 'REFUTED')}
    iv = {k: v_['induction_verdict'].get('x4', {}) for k, v_ in C.items()}
    flips = {k: x for k, x in iv.items()
             if x.get('conventional_inducts') and not x.get('family_inducts')}
    key = iv.get('d2_w128', {})
    v['A4_conventional_inducts_at_depth2_width128'] = {
        'registered': rep['registered_predictions']['analysts_registered_'
                                                    'predictions']
        ['A4_induction_the_interesting_half'],
        'cell': key,
        'verdict': (None if key.get('conventional_mean') is None else
                    'CONFIRMED' if key.get('conventional_inducts')
                    and not key.get('family_inducts') else 'REFUTED'),
        'all_cells_where_conventional_inducts_and_family_does_not':
            sorted(flips),
        'reframing': 'if any depth-2 cell flips, RESULTS FINDING 16 becomes a '
                     'statement about the no-softmax bilinear family and not '
                     'about transformers at this scale.'}
    d1 = {k: x for k, x in iv.items()
          if k.startswith('d1_') and x.get('conventional_mean') is not None}
    v['A5_depth1_null_in_both_families'] = {
        'per_cell': {k: {'family': x.get('family_inducts'),
                         'conventional': x.get('conventional_inducts')}
                     for k, x in d1.items()},
        'verdict': (None if not d1 else
                    'CONFIRMED' if not any(x.get('conventional_inducts')
                                           or x.get('family_inducts')
                                           for x in d1.values())
                    else 'REFUTED')}
    v['A9_learning_rate_fairness_bound'] = {
        'arms': rep['lr_arms'],
        'worst_gain_over_0.02': (max([x['gain_over_0.02']
                                      for x in rep['lr_arms'].values()
                                      if x.get('gain_over_0.02') is not None],
                                     default=None)),
        'verdict_note': 'if any conventional cell gains more than 0.02 nats '
                        'from a different learning rate, every gap quoted at '
                        '0.02 UNDERSTATES the conventional model and the whole '
                        'comparison is a LOWER BOUND on the conventional '
                        'advantage.'}
    return v


# ------------------------------------------------------------------ markdown
def table(rep):
    C = rep['cells']
    L = ['# The foldability tax — conventional softmax+GELU baseline',
         '',
         'Positive gap = the conventional model wins = what the exact fold '
         'costs in prediction quality. Held cross-entropy in nats/token at '
         'T=512 on held rows [0:1500]; both families share the tokenizer, so '
         'nats are directly comparable (bits/byte = nats / (ln2 × 3.755)).',
         '',
         '| depth | width | family CE | conventional ×4 CE | gap ×4 | '
         'conventional ×7 CE | gap ×7 | seeds (fam/×4/×7) | params '
         '(fam / ×4 / ×7) |',
         '|---|---|---|---|---|---|---|---|---|']
    for k, c in sorted(C.items(), key=lambda kv: (kv[1]['depth'],
                                                  kv[1]['width'])):
        f = c['family']
        a4 = c['arms'].get('x4', {})
        a7 = c['arms'].get('x7', {})

        def fm(x, n=4):
            return '—' if x is None else f'{x:.{n}f}'

        def pm(a):
            return ('—' if a.get('held_ce') is None else
                    f"{a['held_ce']:.4f} ± {a.get('held_ce_sd') or 0:.4f}")
        L.append(
            f"| {c['depth']} | {c['width']} | "
            f"{fm(f['held_ce'])} ± {fm(f.get('held_ce_sd') or 0)} | "
            f"{pm(a4)} | **{fm(a4.get('gap_nats'))}** | {pm(a7)} | "
            f"**{fm(a7.get('gap_nats'))}** | "
            f"{f['n_seeds']}/{a4.get('n_seeds', 0)}/{a7.get('n_seeds', 0)} | "
            f"{f.get('n_params')} / {a4.get('n_params', '—')} / "
            f"{a7.get('n_params', '—')} |")
    L += ['', '## Induction — the interesting half', '',
          'Synthetic order-sensitive score (shuffled − repeat), decided over '
          f'MODEL seeds at t > {MODEL_SEED_T_BAR}.', '',
          '| depth | width | family induction | family inducts? | '
          'conventional ×4 induction | conventional inducts? | '
          'family bag | conventional bag |',
          '|---|---|---|---|---|---|---|---|']
    for k, c in sorted(C.items(), key=lambda kv: (kv[1]['depth'],
                                                  kv[1]['width'])):
        iv = c['induction_verdict'].get('x4', {})
        a4 = c['arms'].get('x4', {})
        f = c['family']

        def fm(x, n=4):
            return '—' if x is None else f'{x:+.{n}f}'

        def tt(x):
            return '—' if x is None else f'{x:.1f}'
        L.append(
            f"| {c['depth']} | {c['width']} | "
            f"{fm(f.get('induction'))} ± {(f.get('induction_sd') or 0):.4f} "
            f"(t {tt(f.get('induction_model_seed_t'))}) | "
            f"{'YES' if f.get('inducts') else 'no'} | "
            f"{fm(a4.get('induction'))} ± {(a4.get('induction_sd') or 0):.4f} "
            f"(t {tt(iv.get('conventional_model_seed_t'))}) | "
            f"{'**YES**' if iv.get('conventional_inducts') else 'no'} | "
            f"{fm(f.get('bag'))} | {fm(a4.get('bag'))} |")
    v = rep['verdicts']
    L += ['', '## Registered predictions, scored', '',
          '| prediction | verdict |', '|---|---|']
    for k in ('L1_conventional_wins_every_cell_by_0.05_to_0.20',
              'L2_A2_gap_grows_with_depth',
              'A4_conventional_inducts_at_depth2_width128',
              'A5_depth1_null_in_both_families'):
        L.append(f"| {k} | **{v.get(k, {}).get('verdict')}** |")
    l3 = v.get('L3_A3_matched_parameters', {})
    L.append(f"| L3 (matched parameters shrink the gap by a third) | "
             f"**{l3.get('L3_verdict')}** |")
    L.append(f"| A3 (matched parameters WIDEN the gap, because the family is "
             f"the larger model) | **{l3.get('A3_verdict')}** |")
    if rep['lr_arms']:
        L += ['', '## Learning-rate fairness bound', '',
              '| cell | 0.01 | 0.02 (primary) | 0.04 | best | gain over 0.02 |',
              '|---|---|---|---|---|---|']
        for k, a in sorted(rep['lr_arms'].items()):
            def g(x):
                return ('—' if x not in a or a[x].get('held_ce') is None
                        else f"{a[x]['held_ce']:.4f}")
            L.append(f"| {k} | {g('0.01')} | {g('0.02')} | {g('0.04')} | "
                     f"{a.get('best_lr', '—')} | "
                     f"{(a.get('gain_over_0.02') or 0):.4f} |")
    if rep['qk_norm_arm']:
        L += ['', '## Query/key-norm control', '',
              '| cell | with QK-norm | without | cost of removing |',
              '|---|---|---|---|']
        for k, a in sorted(rep['qk_norm_arm'].items()):
            L.append(f"| {k} | {a.get('with_qk_norm_held_ce')} | "
                     f"{a.get('no_qk_norm_held_ce')} | "
                     f"{a.get('cost_of_removing_qk_norm')} |")
    L += ['', '## Controls', '']
    for nm, c in rep['controls'].items():
        L.append(f"- `{nm}`: pass = **{c.get('pass')}**")
    L += ['', 'Full record: `tf_baseline_std.json`; registered predictions: '
              '`tf_baseline_predictions.json`.']
    open(f'{HERE}/tf_baseline_table.md', 'w').write('\n'.join(L) + '\n')
    return '\n'.join(L)


if __name__ == '__main__':
    r = build()
    print(table(r))
