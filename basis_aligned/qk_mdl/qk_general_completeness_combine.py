"""Combine per-model completeness + hub jsons into qk_general_completeness.json, with the
bilin18 reference values, for the cross-architecture verdict."""
import json, os
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
def L(p):
    fp = f'{QK}/{p}'
    return json.load(open(fp)) if os.path.exists(fp) else None

# ---- bilin18 reference (already computed) ----
cov = L('qk_coverage_ledger.json'); sup = L('qk_mlp_superposition.json'); h1 = L('qk_mlp1_tail.json')
bilin18 = {
    'super_additivity_joint_over_sum': cov['ledger']['joint_over_sum_ratio_global'],
    'full_headroom': cov['ledger']['full_headroom']['dCE'],
    'joint_all': cov['ledger']['joint_234']['dCE'],
    'sum_of_single_paths': cov['ledger']['single_path_sum_global']['dCE'],
    'mlp_below_top_frac_of_full': cov['ledger']['mlp_below_top72']['frac_of_mlp_full'],
    'non_axis_residual_frac_of_headroom': cov['ledger']['coverage_fractions_of_headroom']['non_axis_aligned_residual']['frac'],
    'effective_rank_per_block': sup['effective_rank_per_block_for_pct_of_full'],
    'svd_vs_random_over_ratio': {k: v['svd_over_rand'] for k, v in sup['svd_vs_random_at_K'].items()},
    'top_tail_layers': [(r['layer'], r['tail_below_top4_dCE'], r['tail_share_across_layers'])
                        for r in sorted(sup['per_layer_tail'], key=lambda r: -r['tail_below_top4_dCE'])[:4]],
    'hub': {'layer': h1['meta']['layer'],
            'joint_top32_global_dCE': h1['summary']['full_MLP_L1_top32_global_dCE'],
            'sum_of_solos_global_dCE': h1['summary']['sum_per_dir_global_dCE'],
            'joint_over_sum': round(h1['summary']['full_MLP_L1_top32_global_dCE']/h1['summary']['sum_per_dir_global_dCE'], 3),
            'n_single_direction_nameable': h1['summary']['n_interpretable'],
            'bucket_counts': h1['summary']['bucket_counts']},
}

out = {'reference_bilin18': bilin18}
for mdl in ('swiglu18', 'bilin12'):
    comp = L(f'qk_general_completeness_{mdl}.json')
    hub = L(f'qk_general_completeness_{mdl}_hub.json')
    if comp is None:
        out[mdl] = {'status': 'not run / did not port'}
        continue
    entry = {
        'arch': comp['meta']['arch'],
        'super_additivity_joint_over_sum': comp['super_additivity']['joint_over_sum_ratio'],
        'full_headroom': comp['coverage_ledger']['full_headroom']['dCE'],
        'joint_all': comp['super_additivity']['joint_all_dCE'],
        'sum_of_single_paths': comp['super_additivity']['sum_of_single_paths_dCE'],
        'mlp_below_top_frac_of_full': comp['coverage_ledger']['mlp_below_top']['frac_of_mlp_full'],
        'non_axis_residual_frac_of_headroom': comp['coverage_ledger']['non_axis_aligned_residual']['frac_of_headroom'],
        'effective_rank_per_block': comp['rank_superposition']['effective_rank_per_block_for_pct_of_full'],
        'svd_vs_random_over_ratio': {k: v['svd_over_rand'] for k, v in comp['rank_superposition']['svd_vs_random_at_K'].items()},
        'top_tail_layers': [(r['layer'], r['tail_below_top4_dCE'], r['tail_share_across_layers'])
                            for r in sorted(comp['hub']['per_layer_tail'], key=lambda r: -r['tail_below_top4_dCE'])[:4]],
        'hub_layer_overall': comp['hub']['hub_layer_overall'], 'hub_layer_early_half': comp['hub']['hub_layer_early_half'],
    }
    if hub is not None:
        s = hub['summary']
        entry['hub'] = {'layer': s['hub_layer'],
                        'joint_top32_global_dCE': s['full_layer_top32_joint_global_dCE'],
                        'sum_of_solos_global_dCE': s['sum_per_dir_solo_global_dCE'],
                        'joint_over_sum': s['joint_over_sum_ratio'],
                        'n_single_direction_nameable': s['n_single_direction_nameable'],
                        'bucket_counts': s['bucket_counts'],
                        'frac_nameable_of_summed_solo_global_dCE': s['frac_nameable_of_summed_solo_global_dCE'],
                        'mean_output_entropy_nats': s['mean_output_entropy_nats'],
                        'log_vocab_nats': s['log_vocab_nats']}
    out[mdl] = entry

json.dump(out, open(f'{QK}/qk_general_completeness.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
print("\nSaved qk_general_completeness.json")
