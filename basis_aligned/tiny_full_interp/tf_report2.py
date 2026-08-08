"""Aggregate the depth-1 and depth-2 interpretations into the comparison
tables RESULTS.md quotes, and run the two cross-checkpoint controls that no
single-cell run can do:

  * the matched DEPTH-1 NULL for the natural-text swap probe (a depth-1 model
    structurally cannot compose, so whatever it scores on that probe is the
    probe's confound, not induction);
  * the seed spread of every headline.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def cells():
    out = {}
    for f in glob.glob(f'{HERE}/tf_vanilla_d*_interp2.json'):
        d = json.load(open(f))
        m = re.search(r'_d(\d)_w(\d+)_(b|v)(\d+)_s(\d)', f)
        out[(int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(5)))] = d
    for f in glob.glob(f'{HERE}/tf_vanilla_d1_*_interp.json'):
        d = json.load(open(f))
        m = re.search(r'_d(\d)_w(\d+)_(b|v)(\d+)_s(\d)', f)
        k = (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(5)))
        out.setdefault(k, d)
    return out


def agg(rows, key):
    v = [r for r in rows if r is not None]
    if not v:
        return None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def kl(d, stage):
    L = d.get('rung5_ladder', {})
    return L[stage]['kl_from_model'] if stage in L else None


def main():
    C = cells()
    widths = sorted({k[1] for k in C if k[2] == 'b'})
    rep = {'cells': {f'd{k[0]}_w{k[1]}_{k[2]}_s{k[3]}': True for k in C}}

    # ---------------- headline table --------------------------------------
    stages = ['model_bigram', 'no_attention_at_all', 'past_attn_mean_ablated',
              'no_mlp', 'positional_only_pattern', 'no_rotary_pattern',
              'trunc_delta1_only', 'trunc_delta_le16']
    tab = {}
    for depth in (1, 2):
        for w in widths:
            ks = [k for k in C if k[0] == depth and k[1] == w and k[2] == 'b']
            if not ks:
                continue
            row = {'n_seeds': len(ks),
                   'ce': agg([C[k]['rung5_ladder']['_model_ce'] for k in ks], 0)}
            for s in stages:
                row[s] = agg([kl(C[k], s) for k in ks], s)
            # layer split (depth 2 only)
            for s in ['no_attn_layer0', 'no_attn_layer1', 'no_mlp_layer0',
                      'no_mlp_layer1', 'l1_reads_embedding',
                      'l1_qk_reads_embedding', 'l1_v_reads_embedding',
                      'l1_reads_e_plus_attn0', 'l1_reads_e_plus_mlp0']:
                a = agg([kl(C[k], s) for k in ks], s)
                if a:
                    row[s] = a
            # per-head additivity
            hd = []
            for k in ks:
                L = C[k].get('rung5_ladder', {})
                for li in range(depth):
                    dr = [L[f'drop_l{li}_head{h}']['kl_from_model']
                          for h in range(64) if f'drop_l{li}_head{h}' in L]
                    if not dr:
                        dr = [L[f'drop_head{h}']['kl_from_model']
                              for h in range(64) if f'drop_head{h}' in L]
                    jt = L.get(f'no_attn_layer{li}',
                               L.get('no_attention_at_all', {})
                               ).get('kl_from_model')
                    if dr and jt:
                        hd.append({'layer': li, 'sum_of_single_drops': sum(dr),
                                   'joint': jt, 'ratio': sum(dr) / jt,
                                   'per_head': dr})
            row['head_additivity'] = hd
            # induction
            row['induction'] = agg([C[k]['rung3_induction']['induction_score_mean']
                                    for k in ks], 0)
            row['bag'] = agg([C[k]['rung3_induction']['bag_score_mean']
                              for k in ks], 0)
            fl = [C[k].get('induction_power', {}).get(
                'detectable_effect_floor_nats_3se') for k in ks]
            row['induction_floor'] = agg([f for f in fl if f], 0)
            # ladder order
            oo = []
            for k in ks:
                o = C[k].get('ladder_order')
                if o is None:
                    p = (f'{HERE}/tf_vanilla_d{k[0]}_w{k[1]}_{k[2]}8192'
                         f'_s{k[3]}_order.json')
                    if os.path.exists(p):
                        o = json.load(open(p))
                if o:
                    oo.append(o)
            if oo:
                row['order'] = {
                    'attention_first': agg([o['attention_marginal_first']
                                            for o in oo], 0),
                    'attention_last': agg([o['attention_marginal_last']
                                           for o in oo], 0),
                    'ratio': agg([o['order_dependence_ratio_attention']
                                  for o in oo], 0),
                    'mlp_first': agg([o['mlp_marginal_first'] for o in oo], 0),
                    'mlp_last': agg([o['mlp_marginal_last'] for o in oo], 0)}
            tab[f'depth{depth}_width{w}'] = row
    rep['headline'] = tab

    # -------- matched depth-1 null for the natural-text swap probe --------
    import tf_interp2 as I2
    nat = {}
    for k in sorted(C):
        if k[2] != 'b':
            continue
        stem = f'tf_vanilla_d{k[0]}_w{k[1]}_b8192_s{k[3]}'
        if not os.path.exists(f'{HERE}/{stem}.pt'):
            continue
        D = I2.DeepFold(stem)
        r = I2.natural_induction(D, n_seq=1024)
        nat[stem] = r
        del D
        torch.cuda.empty_cache()
        print(stem, 'swap %+.4f (t %+.2f)'
              % (r['ORDER_ONLY_patch_swap']['mean'],
                 r['ORDER_ONLY_patch_swap']['t']), flush=True)
    rep['natural_induction'] = nat
    matched = {}
    for w in widths:
        d1 = [v['ORDER_ONLY_patch_swap'] for s, v in nat.items()
              if f'_d1_w{w}_' in s]
        d2 = [v['ORDER_ONLY_patch_swap'] for s, v in nat.items()
              if f'_d2_w{w}_' in s]
        if d1 and d2:
            m1 = float(np.mean([x['mean'] for x in d1]))
            m2 = float(np.mean([x['mean'] for x in d2]))
            se = float(np.sqrt(np.mean([x['se'] ** 2 for x in d1])
                               + np.mean([x['se'] ** 2 for x in d2])))
            matched[f'width{w}'] = {
                'depth1_null_mean': m1, 'depth2_mean': m2,
                'excess_depth2_minus_depth1': m2 - m1, 'se': se,
                't': (m2 - m1) / se,
                'verdict': 'induction' if (m2 - m1) / se > 3 else
                           'no induction above the matched null'}
    rep['natural_induction_matched'] = matched
    json.dump(rep, open(f'{HERE}/tf_report2.json', 'w'), indent=2)
    print(json.dumps(matched, indent=2))
    return rep


if __name__ == '__main__':
    main()
