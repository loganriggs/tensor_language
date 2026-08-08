"""The six-architecture comparison table (GRID.md phase V1).

Reads every `*_interp3.json` in the slice and answers the five questions the
slice was scoped to answer, with the seed spread already measured for vanilla
used as the significance yardstick:

  1. the rung-5 ladder stage by stage
  2. the composition budget -- does any variant OPEN the attention-to-attention
     path that is numerically closed in vanilla
  3. induction, with the probe's power floor and the depth-1 matched null
  4. selection-vs-content rank against the same-shape random nulls
  5. per-route ablations as a RANGE [zero, resample], never a point

Usage
    python tf_variant_compare.py            # writes tf_variant_compare.json
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# the yardsticks, measured on VANILLA before this slice existed
SEED_SD = {'ce_d2_w128': 0.0074,        # 4.65117 / 4.65005 / 4.63768
           'induction_d2_w256': 0.0086}

ORDER = ['vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink']
LADDER_STAGES = ['embed_only', 'plus_self_attn', 'model_bigram',
                 'no_attention_at_all', 'past_attn_mean_ablated', 'no_mlp',
                 'no_attn_layer0', 'no_attn_layer1', 'no_mlp_layer0',
                 'no_mlp_layer1', 'l1_reads_embedding',
                 'l1_reads_e_plus_attn0', 'l1_reads_e_plus_mlp0',
                 'trunc_delta1_only', 'trunc_delta_le4', 'positional_only_pattern',
                 'no_rotary_pattern']


def load():
    cells = {}
    for p in sorted(glob.glob(f'{HERE}/*_interp3.json')):
        r = json.load(open(p))
        cells[os.path.basename(p)[:-len('_interp3.json')]] = r
    return cells


def key_of(stem, r):
    """A short label: variant [+ arm suffix] @ depth/seed."""
    v = r.get('variant', '?')
    suf = ''
    for s in ('_writeinit_only', '_nolasso', '_slot32'):
        if stem.endswith(s):
            suf = s
    return f"{v}{suf}_d{r.get('depth')}_s{r['config'].get('seed')}"


def summarise(cells):
    out = {'seed_spread_yardstick': SEED_SD,
           'note': 'no difference smaller than the vanilla seed spread at this '
                   'exact cell (CE 0.0074 nats, induction 0.0086) is called a '
                   'difference'}
    rows = {}
    for stem, r in cells.items():
        if r.get('depth') != 2 or r.get('width') != 128:
            continue
        k = key_of(stem, r)
        L = r['rung5_ladder']
        cb = r['composition_budget']
        crd = cb['causal_read_deletion_kl']
        ra = r['resample_ablation']
        ru = r['rung2']
        # LIVE ROWS ONLY -- see rung2_v: an all-rows spectrum on a masked
        # decoder measures the mask (32 of 128 rows), not the content
        m0 = ru['layer0']['mlp']['mode0_unfolding_live']['entropy_rank']
        n0 = ru['layer0']['mlp']['random_factored_null_mode0_live'][
            'entropy_rank']
        sel = np.mean([ru['layer0']['branch_tables'][f'{b}_d{d}']
                       ['mean_entropy_rank']
                       for b in ('s1', 's2') for d in (0, 1, 8)])
        selnull = ru['layer0']['nulls']['random_branch_table_entropy_rank_mean']
        rows[k] = {
            'stem': stem,
            'variant': r['variant'],
            'params_total': r['params']['total'],
            'params_body': r['params']['body'],
            'params_embedding': r['params']['embedding'],
            'params_effective_total': r['params']['effective_total'],
            'stream_width': r['params']['stream_width'],
            'held_ce_T512_1500seq': r['train']['final_held_ce'],
            'bits_per_byte': r['train']['bits_per_byte'],
            'ladder_ce_T256_96seq': L['_model_ce'],
            'fold_gate_pass': r['fold_gate']['pass'],
            'fold_gate_fp64_logit_abs': r['fold_gate'][
                'fp64_fold_forward_max_logit_diff'],
            'pipeline_rel_logit_diff': r['decomposition_control'][
                'pipeline_rel_logit_diff'],
            # --- 1 ladder ---
            'ladder_kl': {s: L[s]['kl_from_model'] for s in LADDER_STAGES
                          if s in L},
            # --- 2 composition: THE NORMALISATION-INVARIANT ROWS ---
            # norm share is NOT normalisation invariant and is not evidence;
            # these five are (KL/CE of an intervention, and relative changes)
            'A2A_kl_range_zero_to_resample': [
                r['read_ablation_causal']['kl_from_model']['l1_read_zero_A0'],
                r['read_ablation_causal']['kl_from_model']['l1_read_resample_A0']],
            'M0_read_kl_range_zero_to_resample': [
                r['read_ablation_causal']['kl_from_model']['l1_read_zero_M0'],
                r['read_ablation_causal']['kl_from_model']['l1_read_resample_M0']],
            'A2A_pattern_rel_change_zero': r['read_ablation_causal'][
                'relative_change']['l1_read_zero_A0_pattern_rel'],
            'A2A_value_rel_change_zero': r['read_ablation_causal'][
                'relative_change']['l1_read_zero_A0_value_rel'],
            'norm_confound_A0_pattern_true_vs_imposed': [
                r['norm_confound_control'][
                    'true_norm_l1_pattern_rel_change_without_A0'],
                r['norm_confound_control'][
                    'imposed_slot_norm_l1_pattern_rel_change_without_A0']],
            'A2A_causal_kl_l1_read_without_A0': crd.get('l1_read_without_A0'),
            'A2A_causal_kl_l1_qk_read_without_A0': crd.get(
                'l1_qk_read_without_A0'),
            'causal_kl_l1_read_without_M0': crd.get('l1_read_without_M0'),
            'A0_logit_share': r['stream_geometry']['A0_share'],
            'A1_logit_share': r['stream_geometry']['A1_share'],
            'M0_logit_share': r['stream_geometry']['M0_share'],
            'M1_logit_share': r['stream_geometry']['M1_share'],
            'e_logit_share': r['stream_geometry']['e_share'],
            'l1_read_norm_share_A0': cb.get('l1_read_norm_share_A0'),
            'l1_read_norm_share_M0': cb.get('l1_read_norm_share_M0'),
            # --- 3 induction ---
            'induction_mean': r['rung3_induction']['induction_score_mean'],
            'induction_sd': r['rung3_induction']['induction_score_sd'],
            'induction_floor_3se': r['induction_power'][
                'detectable_effect_floor_nats_3se'],
            'bag_mean': r['rung3_induction']['bag_score_mean'],
            'natural_swap_mean': r['natural_induction'][
                'ORDER_ONLY_patch_swap']['mean'],
            'natural_swap_t': r['natural_induction']['ORDER_ONLY_patch_swap']['t'],
            'natural_swap_floor_3se': r['natural_induction'][
                'ORDER_ONLY_patch_swap']['detectable_effect_floor_nats_3se'],
            # --- 4 rank ---
            'content_mlp_entropy_rank': m0,
            'content_random_factored_null': n0,
            'content_over_null': m0 / max(n0, 1e-9),
            'selection_branch_entropy_rank': float(sel),
            'selection_random_null': selnull,
            'selection_over_null': float(sel) / max(selnull, 1e-9),
            # --- 5 ablation ranges ---
            'attn_all_kl_range_zero_to_resample': [ra['zero_attn_all'],
                                                   ra['resample_attn_all']],
            'attn_l0_kl_range_zero_to_resample': [ra['zero_attn_layer0'],
                                                  ra['resample_attn_layer0']],
            'attn_l1_kl_range_zero_to_resample': [ra['zero_attn_layer1'],
                                                  ra['resample_attn_layer1']],
            'ladder_order_attention_first': r['ladder_order'][
                'attention_marginal_first'],
            'ladder_order_attention_last': r['ladder_order'][
                'attention_marginal_last'],
            'ladder_order_interaction_nats': r['ladder_order'][
                'interaction_nats'],
        }
        # per-head induction localisation: which single head carries the most
        best = None
        for hk, hv in r['induction_by_head']['per_head'].items():
            d = (r['induction_by_head']['folded_pipeline_baseline'][
                'induction_score_mean'] - hv['induction_score_mean'])
            if best is None or d > best[1]:
                best = (hk, d)
        rows[k]['induction_carried_by'] = best[0]
        rows[k]['induction_removed_by_that_head'] = best[1]
        rows[k]['induction_baseline_folded'] = r['induction_by_head'][
            'folded_pipeline_baseline']['induction_score_mean']
        mech = r.get('mechanism', {})
        rows[k]['mean_live_slots_per_read'] = mech.get(
            'mean_live_slots_per_read')
        if 'codebook' in mech:
            rows[k]['codebook'] = mech['codebook']
            rows[k]['codebook_rel_quant_error'] = mech.get(
                'codebook_relative_quantisation_error')
        if 'predicate' in mech:
            rows[k]['predicate'] = mech['predicate']
            pl = r['rung5_ladder']
            rows[k]['predicate_term_kl'] = {
                w: pl[f'pred_{w}_all_layers']['kl_from_model']
                for w in ('branches', 'named', 'no_prev', 'no_same', 'no_prof')
                if f'pred_{w}_all_layers' in pl}
    out['cells'] = rows
    # ---- depth-1 matched nulls for the natural swap probe ----
    d1 = {}
    for stem, r in cells.items():
        if r.get('depth') == 1 and r.get('width') == 128:
            d1[r['variant']] = {
                'natural_swap_mean': r['natural_induction'][
                    'ORDER_ONLY_patch_swap']['mean'],
                'natural_swap_se': r['natural_induction'][
                    'ORDER_ONLY_patch_swap']['se'],
                'synthetic_induction_mean': r['rung3_induction'][
                    'induction_score_mean'],
                'stem': stem}
    out['depth1_matched_nulls'] = d1
    for k, v in rows.items():
        base = d1.get(v['variant'])
        if base:
            v['natural_swap_excess_over_depth1_null'] = (
                v['natural_swap_mean'] - base['natural_swap_mean'])
            v['synthetic_induction_excess_over_depth1_null'] = (
                v['induction_mean'] - base['synthetic_induction_mean'])
    # ---- seed aggregation ----
    agg = {}
    for k, v in rows.items():
        base = k.rsplit('_s', 1)[0]
        agg.setdefault(base, []).append(v)
    seeds = {}
    for base, vs in agg.items():
        if len(vs) < 2:
            continue
        def st(f):
            a = [x[f] for x in vs if x.get(f) is not None]
            return {'mean': float(np.mean(a)), 'sd': float(np.std(a, ddof=1)),
                    'n': len(a), 'values': [float(z) for z in a]}
        seeds[base] = {f: st(f) for f in
                       ('held_ce_T512_1500seq', 'induction_mean',
                        'A2A_causal_kl_l1_read_without_A0', 'A0_logit_share',
                        'A2A_pattern_rel_change_zero',
                        'natural_swap_mean', 'content_over_null',
                        'selection_over_null')}
        for f in ('A2A_kl_range_zero_to_resample',):
            a = np.array([x[f] for x in vs], dtype=float)
            seeds[base][f] = {'zero_mean': float(a[:, 0].mean()),
                              'zero_sd': float(a[:, 0].std(ddof=1)),
                              'resample_mean': float(a[:, 1].mean()),
                              'resample_sd': float(a[:, 1].std(ddof=1)),
                              'n': len(a)}
    out['across_seeds'] = seeds
    # ---- the verdict arithmetic ----
    v = {}
    van = rows.get('vanilla_d2_s0')
    if van:
        for k, r_ in rows.items():
            if k == 'vanilla_d2_s0':
                continue
            v[k] = {
                'ce_minus_vanilla': r_['held_ce_T512_1500seq']
                - van['held_ce_T512_1500seq'],
                'ce_in_units_of_vanilla_seed_sd':
                    (r_['held_ce_T512_1500seq'] - van['held_ce_T512_1500seq'])
                    / SEED_SD['ce_d2_w128'],
                'induction_minus_vanilla': r_['induction_mean']
                - van['induction_mean'],
                'induction_in_units_of_seed_sd':
                    (r_['induction_mean'] - van['induction_mean'])
                    / SEED_SD['induction_d2_w256'],
                'A2A_kl_minus_vanilla':
                    (r_['A2A_causal_kl_l1_read_without_A0'] or 0)
                    - (van['A2A_causal_kl_l1_read_without_A0'] or 0),
                'ladder_stage_max_abs_diff': max(
                    abs(r_['ladder_kl'][s] - van['ladder_kl'][s])
                    for s in van['ladder_kl'] if s in r_['ladder_kl']),
                'ladder_ordering_matches_vanilla': (
                    [s for s in sorted(van['ladder_kl'],
                                       key=lambda z: -van['ladder_kl'][z])]
                    == [s for s in sorted(r_['ladder_kl'],
                                          key=lambda z: -r_['ladder_kl'][z])]),
            }
    out['vs_vanilla'] = v
    return out


def render(o):
    R = o['cells']
    ks = [k for k in R if k.endswith('_s0')]
    ks.sort(key=lambda k: (ORDER.index(R[k]['variant'])
                           if R[k]['variant'] in ORDER else 9, k))
    w = 26
    def row(label, fn, fmt='{:>11}'):
        s = f'{label:<{w}}'
        for k in ks:
            try:
                s += fmt.format(fn(R[k]))
            except Exception:
                s += fmt.format('-')
        return s
    lines = []
    lines.append(' ' * w + ''.join(f'{k.replace("_d2_s0",""):>11}' for k in ks))
    lines.append('-' * (w + 11 * len(ks)))
    for lbl, f in [
            ('params total', lambda r: r['params_total']),
            ('params body', lambda r: r['params_body']),
            ('stream width', lambda r: r['stream_width']),
            ('held CE (T512)', lambda r: f"{r['held_ce_T512_1500seq']:.4f}"),
            ('bits/byte', lambda r: f"{r['bits_per_byte']:.4f}"),
            ('fold gate', lambda r: 'PASS' if r['fold_gate_pass'] else 'FAIL'),
            ('--- INDUCTION', lambda r: ''),
            ('synthetic score', lambda r: f"{r['induction_mean']:+.4f}"),
            ('  power floor 3se', lambda r: f"{r['induction_floor_3se']:.4f}"),
            ('  x floor', lambda r: f"{r['induction_mean']/r['induction_floor_3se']:+.1f}"),
            ('natural swap', lambda r: f"{r['natural_swap_mean']:+.4f}"),
            ('  over d1 null', lambda r: f"{r.get('natural_swap_excess_over_depth1_null', float('nan')):+.4f}"),
            ('  carried by', lambda r: r['induction_carried_by'].replace('drop_', '')),
            ('--- A->A PATH (causal)', lambda r: ''),
            ('A0 out of l1 read [z,r]', lambda r: '%.3f/%.3f' % tuple(r['A2A_kl_range_zero_to_resample'])),
            ('M0 out of l1 read [z,r]', lambda r: '%.3f/%.3f' % tuple(r['M0_read_kl_range_zero_to_resample'])),
            ('  l1 pattern move w/o A0', lambda r: f"{r['A2A_pattern_rel_change_zero']:.4f}"),
            ('  l1 value move w/o A0', lambda r: f"{r['A2A_value_rel_change_zero']:.4f}"),
            ('  norm-confound true/imp', lambda r: '%.4f/%.4f' % tuple(r['norm_confound_A0_pattern_true_vs_imposed'])),
            ('A0 logit share', lambda r: f"{r['A0_logit_share']:+.4f}"),
            ('M0 logit share', lambda r: f"{r['M0_logit_share']:+.4f}"),
            ('(norm share A0: NOT ev.)', lambda r: f"{r['l1_read_norm_share_A0']:.4f}"),
            ('--- RANK', lambda r: ''),
            ('content / null', lambda r: f"{r['content_over_null']:.3f}"),
            ('selection / null', lambda r: f"{r['selection_over_null']:.3f}"),
            ('--- ABLATION RANGE', lambda r: ''),
            ('attn all [zero,resamp]', lambda r: '%.2f-%.2f' % tuple(r['attn_all_kl_range_zero_to_resample'])),
            ('attn l0 [zero,resamp]', lambda r: '%.2f-%.2f' % tuple(r['attn_l0_kl_range_zero_to_resample'])),
            ('attn l1 [zero,resamp]', lambda r: '%.2f-%.2f' % tuple(r['attn_l1_kl_range_zero_to_resample'])),
            ('attn first / last', lambda r: '%.1f/%.1f' % (r['ladder_order_attention_first'], r['ladder_order_attention_last'])),
            ('--- LADDER KL', lambda r: ''),
    ]:
        lines.append(row(lbl, f))
    for s in LADDER_STAGES:
        lines.append(row('  ' + s[:23],
                         lambda r, s=s: f"{r['ladder_kl'].get(s, float('nan')):.3f}"))
    return '\n'.join(lines)


if __name__ == '__main__':
    o = summarise(load())
    json.dump(o, open(f'{HERE}/tf_variant_compare.json', 'w'), indent=2)
    txt = render(o)
    open(f'{HERE}/tf_variant_compare.txt', 'w').write(txt + '\n')
    print(txt)
