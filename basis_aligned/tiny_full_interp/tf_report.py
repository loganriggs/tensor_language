"""Aggregate the per-cell fold/interp JSONs into the program-level tables and
write the adversarial review section.

Outputs
    tf_identity_table.json     rung 1 gate numbers for every checkpoint
    tf_summary.json            rung 2-5 across widths and seeds
    tf_reviewer_round_1.json   the self-red-team, one entry per claim
Usage
    python tf_report.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _cells(pat='tf_vanilla_d*_*.json'):
    out = {}
    for f in sorted(glob.glob(f'{HERE}/{pat}')):
        b = os.path.basename(f)[:-5]
        if b.endswith(('_interp', '_induction')):
            continue
        out[b] = json.load(open(f))
    return out


def identity_table():
    rows = []
    for stem, d in _cells().items():
        fo = d.get('fold')
        if not fo:
            continue
        g = fo['identity_gate']
        rows.append({
            'stem': stem, 'pass': g['pass'],
            'fp32_mlp_tensor_rel': g['mlp_tensor_identity_l0_relmax'],
            'fp32_rmsnorm_gauge_rel': g['mlp_rmsnorm_gauge_l0_relmax'],
            'fp32_attn_table_rel': g['attn_layer0_table_identity_relmax'],
            'fp32_logit_abs': g['fold_forward_max_logit_diff'],
            'fp32_logit_rel': g['fold_forward_rel_logit_diff'],
            'fp64_mlp_tensor_rel': g['fp64_mlp_tensor_identity_l0_relmax'],
            'fp64_rmsnorm_gauge_rel': g['fp64_mlp_rmsnorm_gauge_l0_relmax'],
            'fp64_attn_table_rel': g['fp64_attn_layer0_table_identity_relmax'],
            'fp64_logit_abs': g['fp64_fold_forward_max_logit_diff'],
            'forward_fp32_self_noise_rel': g['fp32_forward_vs_fp64_forward_rel'],
            'fold_gap_over_self_noise': g['fold_vs_forward_over_fp32_selfnoise'],
            'planted_delta3_abs': fo['planted_known_answer'][
                'planted_delta3_maxdiff'],
            'gate_negative_control_pass': fo.get(
                'gate_negative_control', {}).get('pass'),
            'spectrum_qr_vs_eig_rel': fo.get('control_spectrum_qr_vs_eig_rel'),
            'spectrum_factor_vs_dense_svd_rel': fo.get(
                'control_spectrum_factor_vs_dense_svd_rel'),
            'dense_svd_tail_beyond_rank_bound': fo.get(
                'control_dense_svd_tail_over_rank_bound'),
            'held_ce': d.get('run', {}).get('final_held_ce'),
        })
    return rows


def summary():
    out = {}
    for f in sorted(glob.glob(f'{HERE}/*_interp.json')):
        stem = os.path.basename(f)[:-len('_interp.json')]
        d = json.load(open(f))
        if 'rung5_ladder' not in d:
            continue
        L, r2, g = d['rung5_ladder'], d['rung2'], d['stream_geometry']
        b = d['rung3_baselines']
        cell = json.load(open(f'{HERE}/{stem}.json'))
        ent = {
            'width': cell['config']['width'],
            'seed': cell['config']['seed'],
            'tok': cell['config']['tok'],
            'held_ce_full_eval': cell['run']['final_held_ce'],
            'ce': {k: v['ce'] for k, v in L.items() if isinstance(v, dict)},
            'kl': {k: v['kl_from_model'] for k, v in L.items()
                   if isinstance(v, dict)},
            'baselines': {k: v for k, v in b.items()
                          if isinstance(v, (int, float))},
            'logit_var_share': {k: g[f'{k}_logit_var_share']
                                for k in ('e', 'A0', 'Apast', 'M')},
            'term_norms': {k: g[f'{k}_norm'] for k in ('e', 'A0', 'Apast',
                                                       'M', 'r')},
            'entropy_rank': {
                'Q1': r2['factors']['Q1']['mean_entropy_rank'],
                'K1': r2['factors']['K1']['mean_entropy_rank'],
                'Vv': r2['factors']['Vv']['mean_entropy_rank'],
                's1_d0_table': r2['branch_tables']['s1_d0'][
                    'mean_entropy_rank'],
                's2_d0_table': r2['branch_tables']['s2_d0'][
                    'mean_entropy_rank'],
                'mlp_mode0': r2['mlp']['mode0_unfolding']['entropy_rank'],
                'null_random_factor': r2['nulls'][
                    'random_factor_entropy_rank_mean'],
                'null_random_table': r2['nulls'][
                    'random_branch_table_entropy_rank_mean'],
                'null_random_mlp_mode0': r2['mlp'][
                    'random_factored_null_mode0']['entropy_rank'],
                'bound_head_dim': r2['head_dim'],
                'bound_mlp_mode0': r2['mlp']['mode0_unfolding']['rank_bound'],
            },
            'induction': {k: v for k, v in d['rung3_induction'].items()
                          if k.endswith(('_mean', '_sd'))},
            'rung4': {
                'composed_rank1_share': [h['delta1']['composed_rank1_share']
                                         for h in d['rung4']['heads']],
                'effective_pair_fraction': [
                    h['delta1']['effective_pair_fraction']
                    for h in d['rung4']['heads']],
                'identity_pair_z': [
                    h['identity_pair_enrichment']['z_of_diag_vs_random']
                    for h in d['rung4']['heads']],
                'boosts_itself_rank_median': float(np.median([
                    e['boosts_itself_rank']
                    for ph in d['rung4']['what_attending_does']['per_head']
                    for e in ph['strongest_keys']])),
            },
            'decomposition_control': d['decomposition_control'],
        }
        out[stem] = ent
    # seed aggregation per (tok, width)
    agg = {}
    for stem, e in out.items():
        k = f"{e['tok']}_w{e['width']}"
        agg.setdefault(k, []).append(e)
    seeds = {}
    for k, es in agg.items():
        def ms(f):
            v = [f(e) for e in es]
            return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))
                    if len(v) > 1 else None, 'n': len(v)}
        seeds[k] = {
            'held_ce': ms(lambda e: e['held_ce_full_eval']),
            'kl_model_bigram': ms(lambda e: e['kl']['model_bigram']),
            'kl_no_attention_at_all': ms(lambda e: e['kl'][
                'no_attention_at_all']),
            'kl_past_attn_direct_route_only': ms(
                lambda e: e['kl']['past_attn_direct_route_only']),
            'kl_past_attn_mlp_route_only': ms(
                lambda e: e['kl']['past_attn_mlp_route_only']),
            'kl_mlp_write_only': ms(lambda e: e['kl']['mlp_write_only']),
            'M_logit_var_share': ms(lambda e: e['logit_var_share']['M']),
            'entropy_rank_s1_d0_table': ms(
                lambda e: e['entropy_rank']['s1_d0_table']),
            'entropy_rank_Vv': ms(lambda e: e['entropy_rank']['Vv']),
            'entropy_rank_mlp_mode0': ms(
                lambda e: e['entropy_rank']['mlp_mode0']),
            'induction_score': ms(lambda e: e['induction'][
                'induction_score_mean']),
            'bag_score': ms(lambda e: e['induction']['bag_score_mean']),
        }
    return {'per_cell': out, 'per_width_over_seeds': seeds}


# ---------------------------------------------------------------- review
def reviewer_round_1(ident, summ):
    """Adversarial self-review.  For every claim, the strongest objection a
    hostile reviewer would raise about the CLAIM or the TECHNIQUE, what was
    fixed, and what could not be."""
    S = summ['per_width_over_seeds']

    def g(k, f, d=None):
        return S.get(k, {}).get(f, d)
    n_pass = sum(1 for r in ident if r['pass'])
    return {
        'procedure': 'written after the interpretation, before reporting; '
                     'each entry names the objection, the fix, and the '
                     'residual that could NOT be fixed',
        'claims': [
            {
                'id': 'C1_gate_is_precision',
                'claim': 'the four failing cells failed on fp32 rounding, not '
                         'on a wrong fold; the fold is exact',
                'objection': 'you loosened a gate until your cells passed. '
                             'Threshold shopping after seeing the data is the '
                             'oldest way to make a failure disappear.',
                'fix': [
                    'the new gate is STRICTLY STRONGER, and that is machine '
                    'checked: gate_negative_control corrupts the MLP tensor by '
                    'a factor 1+1e-7 and the resulting fp32 absolute logit '
                    'difference is 1.19e-7 -- the SUPERSEDED absolute-1e-5 gate '
                    'would have passed that corruption, the new fp64 tier '
                    'fails it (9.9e-9 > 1e-9).',
                    'fp64 residuals across all cells: end-to-end 1.3e-14 to '
                    '4.4e-14 absolute at logit magnitude ~15 (about ten fp64 '
                    'ulps); algebraic identities 5e-16 to 1.5e-15 relative.',
                    'the fp32 gap is CALIBRATED, not just bounded: the '
                    "model's own fp32 forward differs from its fp64 forward by "
                    '6e-7 to 2.9e-6 relative, which at width 128 is LARGER '
                    'than the fold-vs-forward gap (1.7e-6).  The fold agrees '
                    'with the forward better than the forward agrees with '
                    'itself.',
                    'the dtype fix also improved two INDEPENDENT controls that '
                    'were not part of the complaint: the planted known-answer '
                    'table at delta=3 went 5.8e-9 -> 1.6e-14, and the fp64 '
                    'attention-table identity is 7e-16.  A threshold change '
                    'cannot do that; a real precision bug being fixed can.',
                ],
                'not_fixed': 'the fp64 arm rebuilds the rotary tables in fp64, '
                             'so the fp64 model is not bit-identical to the '
                             'fp32 one.  It is the same identity tested at a '
                             'different precision, which is what is wanted, '
                             'but it is not a proof about the fp32 object.',
                'verdict': f'{n_pass}/{len(ident)} checkpoints pass the '
                           f'corrected two-tier gate',
            },
            {
                'id': 'C2_mlp_carries_the_logits',
                'claim': 'the MLP write carries essentially 100% of the logit '
                         'variation; the embedding and attention writes are '
                         'invisible at the readout',
                'objection': 'this is a claim about vector NORMS, and RMSNorm '
                             'means only direction matters -- a large-norm term '
                             'can still be functionally irrelevant, and a '
                             'small one decisive.',
                'fix': [
                    'the share is not a norm ratio.  It is the exact '
                    'projection of each additive term on the CENTRED logit '
                    'vector after the gauge, and the four shares sum to 1 by '
                    'construction (checked to 5e-8 in every cell).',
                    'a causal knockout agrees: keeping ONLY the MLP write in '
                    'the residual and discarding the embedding and both '
                    'attention terms reproduces the model at KL '
                    f"{g('bpe_w128','kl_mlp_write_only',{}).get('mean')} "
                    '(width 128).',
                ],
                'not_fixed': 'this may be an architecture artifact rather than '
                             'a fact about transformers: these cells train with '
                             'write_init=false (c_proj and Down start at zero) '
                             'and a tied readout.  Not tested against a '
                             'nonzero-write-init arm.',
            },
            {
                'id': 'C3_attention_acts_through_the_mlp',
                'claim': "the past attention's whole causal effect is on the "
                         "MLP's INPUT; its direct contribution to the readout "
                         'residual is nil',
                'objection': 'the first version of this analysis published the '
                             'OPPOSITE headline -- that attention to the past '
                             'does nothing at all.  Why believe the second '
                             'version?',
                'fix': [
                    'the earlier ladder froze the MLP at its self-only input, '
                    'so it measured the DIRECT route only and read its ~0 '
                    'result as "attention does nothing".  That is exactly the '
                    'composition failure the standing sign/gauge rule warns '
                    'about, in a non-sign form: a term was scored without '
                    'composing it through the downstream nonlinearity.  The '
                    'corrected ladder scores both routes as mutually exclusive '
                    'ablations that BRACKET the model: direct-route-only lands '
                    'on the no-attention KL, mlp-route-only lands on 0.',
                    'the total attention effect is confirmed by a knockout '
                    'with a MEAN ablation (est-split mean of the past-attention '
                    'write) as well as by zeroing, so it is not a residual-norm '
                    'artifact.',
                    'RETRACTED: MAILBOX 2026-08-08 05:00 "the model attends, '
                    'but what it attends to does not matter" and the '
                    'accompanying distance-restriction table.  Corrected '
                    'numbers are in tf_summary.json.',
                ],
                'not_fixed': 'nothing outstanding; the retraction is the fix',
            },
            {
                'id': 'C4_selection_low_rank_content_spectral',
                'claim': 'the layer-0 score tables are far below their rank '
                         'bound while the value factors and the MLP tensor are '
                         'not',
                'objection': 'rank <= head_dim is arithmetic.  "Effective rank '
                             '3 of 16" could still be what any random pair of '
                             'V x 16 factors gives.',
                'fix': [
                    'nulls are computed and reported next to every number: iid '
                    'Gaussian V x 16 factors give entropy rank 15.996 +- 0.001 '
                    'and their product table 15.991 +- 0.001, against a trained '
                    'score-table entropy rank of '
                    f"{g('bpe_w128','entropy_rank_s1_d0_table',{}).get('mean')}"
                    '.  For the MLP the null is a random FACTORED tensor of '
                    'the same shape (entropy rank ~123 of 128) against the '
                    'trained '
                    f"{g('bpe_w128','entropy_rank_mlp_mode0',{}).get('mean')}"
                    ' -- i.e. the MLP is NOT low rank and is reported as such.',
                    'a causal version of the same claim is in the ladder: '
                    'keeping the top 4 of 8 rotary frequency pairs (the only '
                    'delta-equivariant way to cut a head) keeps most of the '
                    'attention benefit.',
                ],
                'not_fixed': 'spectral entropy rank is not the same as "number '
                             'of interpretable directions"; no substitution '
                             'gate was run on the score-table singular '
                             'directions, so rung 4 remains a description, not '
                             'a naming.',
            },
            {
                'id': 'C5_no_induction_at_depth_1',
                'claim': 'depth 1 shows no induction; the metric is calibrated',
                'objection': 'a null result from a metric that has never been '
                             'shown to detect the thing is worthless.',
                'fix': [
                    'the identical battery is run on a DEPTH-2 cell as the '
                    'positive control (tf_vanilla_d2_w64_b8192_s0), and the '
                    'conditions are matched on token multiset (repeat vs '
                    'shuffled-prefix vs fresh-prefix), so the induction score '
                    'is an order effect by construction.',
                    'five probe seeds per cell, sd reported.',
                    'the second number is named BAG score, not copy score: '
                    'rung 4 shows the attended token ranks near the BOTTOM of '
                    'what attending to it boosts, so calling the bag effect '
                    '"copying" would be naming a mechanism from a behavioural '
                    'delta.',
                ],
                'not_fixed': 'the probe sequences are iid draws from the train '
                             'unigram, which is out of distribution in '
                             'structure even though it is in distribution in '
                             'frequency.  A natural-text repeated-span probe '
                             'was not run.',
            },
            {
                'id': 'C6_beats_the_bigram',
                'claim': 'the trained cells beat the closed-form bigram table '
                         'at widths 64 and 128 (and lose to it at width 32)',
                'objection': 'unfair: the model sees the whole prefix and the '
                             'position; a bigram table sees one token.  And '
                             '"beats a 67M-parameter table" is not impressive '
                             'if the comparison is against nothing else.',
                'fix': [
                    'a POSITION PROFILE is reported: at position 0 the model '
                    'and the bigram see exactly the same context, and there '
                    'the width-32 cell LOSES to the bigram (6.43 vs 6.22).',
                    'same-parameter-count comparators are included: a '
                    'truncated-SVD low-rank bigram and a SPARSE bigram (top-m '
                    'counts, unigram backoff) at matched budgets.  The sparse '
                    'bigram is the strong one and it BEATS the weights-only '
                    'model-bigram stage at matched parameters -- reported, not '
                    'hidden.',
                ],
                'not_fixed': 'no trigram / n-gram-with-backoff baseline, which '
                             'is the natural comparator for a model whose '
                             'context clearly matters beyond distance 1.',
            },
            {
                'id': 'C7_token_class_claims',
                'claim': 'orthographic classes among the strongest value '
                         'directions',
                'objection': 'the vocabulary is 76% whitespace-initial '
                             'lowercase, so any top-200 list looks like that.',
                'fix': 'every class share is quoted against a '
                       'FREQUENCY-MATCHED null (400 draws) with a z-score, and '
                       'the write-up names a class only at |z| > 3.  Most '
                       'classes come out at |z| < 3 and are NOT named.',
                'not_fixed': 'the null matches unigram frequency but not token '
                             'length, and BPE merges longer pieces for more '
                             'frequent strings, so length is only partly '
                             'controlled.',
            },
            {
                'id': 'C8_single_seed',
                'claim': 'the structural results are properties of the size, '
                         'not the run',
                'objection': 'the parent program learned that single-seed '
                             'structure claims do not survive.',
                'fix': 'three seeds at each of widths 32, 64 and 128 on the '
                       'primary BPE corpus; every headline in tf_summary.json '
                       'is quoted as mean +- sd over seeds.',
                'not_fixed': 'the truncated-tokenizer comparison arm is still '
                             'one seed per width, and width 256 has no local '
                             'checkpoint at all (the scale box pushed only '
                             'JSONs, no .pt), so the width curve stops at 128.',
            },
            {
                'id': 'C9_fit_and_score_on_the_same_tokens',
                'claim': 'the rung-5 KL numbers are honest',
                'objection': 'were the tables fit on the text they are scored '
                             'on?',
                'fix': 'the model-side tables are computed from WEIGHTS ONLY '
                       'and touch no data at all.  The two fitted objects in '
                       'the ladder (the token-independent distance profile and '
                       'the mean past-attention write used for mean ablation) '
                       'are fit on the ESTIMATION split; the data baselines '
                       'are fit on train (counts) and est (alpha); everything '
                       'is scored on HELD, which no cell trained on.',
                'not_fixed': 'the held slice used for the ladder overlaps the '
                             'held slice used for the cells\' reported final '
                             'CE.  Both are pure eval, so this is not leakage, '
                             'but the two numbers are not independent samples.',
            },
        ],
        'sign_bearing_claims_audit': {
            'rule': 'sign is a gauge freedom; only complete paths to an '
                    'observable have invariant signs',
            'entries': [
                {'quantity': 'per-term logit variance share (e, A0, Apast, M)',
                 'composed_to_logits': True,
                 'causally_confirmed': True,
                 'note': 'each term is projected on the logit vector AFTER the '
                         'RMSNorm gauge; the negative share of the embedding '
                         'term is a complete path (e -> W_U) and is confirmed '
                         'by the mlp_write_only knockout'},
                {'quantity': 'composed copy score p_h(t,u,d) * (OV_h[u].W_U[u])',
                 'composed_to_logits': True,
                 'causally_confirmed': False,
                 'note': 'complete path, so the sign is invariant, but no '
                         'per-pair causal test was run; the head-level '
                         'knockouts (drop_head) are the causal evidence and '
                         'they are aggregate'},
                {'quantity': 'identity-pair enrichment z (negative in most '
                             'heads)',
                 'composed_to_logits': True,
                 'causally_confirmed': False,
                 'note': 'reported as "attending to a token does not push its '
                         'own logit up", NOT as suppression of that token'},
                {'quantity': 'raw pattern sign, raw Q/K/V factor signs',
                 'composed_to_logits': False,
                 'causally_confirmed': False,
                 'note': 'NEVER interpreted anywhere in this program; the '
                         'pattern is itself a product of two branches whose '
                         'individual signs are meaningless'},
                {'quantity': '"attention is inhibitory / anti-attending"',
                 'composed_to_logits': None,
                 'causally_confirmed': None,
                 'note': 'NOT CLAIMED.  No head is described as inhibitory '
                         'anywhere in this write-up'},
            ],
        },
    }


if __name__ == '__main__':
    ident = identity_table()
    summ = summary()
    json.dump(ident, open(f'{HERE}/tf_identity_table.json', 'w'), indent=2)
    json.dump(summ, open(f'{HERE}/tf_summary.json', 'w'), indent=2)
    rev = reviewer_round_1(ident, summ)
    json.dump(rev, open(f'{HERE}/tf_reviewer_round_1.json', 'w'), indent=2)
    for r in ident:
        print(f"{r['stem']:32s} pass={r['pass']} fp32rel {r['fp32_logit_rel']:.2e} "
              f"fp64abs {r['fp64_logit_abs']:.2e} ratio "
              f"{r['fold_gap_over_self_noise']:.2f}")
    print(json.dumps(summ['per_width_over_seeds'], indent=1))
