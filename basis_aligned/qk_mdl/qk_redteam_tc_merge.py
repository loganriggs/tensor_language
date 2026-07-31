"""Merge the three red-team run outputs into the single deliverable qk_redteam_tc.json."""
import json, numpy as np, torch
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
r = json.load(open(f'{QK}/qk_redteam_tc.json'))
r2 = json.load(open(f'{QK}/qk_redteam_tc_2.json'))
r3 = json.load(open(f'{QK}/qk_redteam_tc_3.json'))
A1 = r['attack1_allocation_fairness']
A1['cells'].update(r2['cells'])
A1['extended_search'] = {'note': ('qk_redteam_tc_2.py / _3.py: finer output/input ratio sweep, richer '
                                  'term profiles (150% and ALL active terms), knob combinations, the '
                                  'uncapped-basis re-run of the one cell where the 768-column cap bound, '
                                  'and the best all-terms configuration at the 4x and 128x budgets.'),
                         'cells': r3['cells']}
allc = {k: v['dCE'] for k, v in A1['cells'].items()
        if not k.startswith('repro') and '4x' not in k and '128x' not in k}
allc.update({k: v['dCE'] for k, v in r3['cells'].items() if v['budget_tag'] == '16x'})
bk = min(allc, key=allc.get)
A1['best_variant'] = {'name': bk, 'dCE': allc[bk],
                      'rank_alloc_16x': 0.8032, 'term_104_best': 1.9009,
                      'gap_closed_frac': round((1.9009 - allc[bk])/(1.9009 - 0.8032), 4),
                      'still_loses_by_paired': (r3['cells'][bk]['paired_vs_rank_alloc']
                                                if bk in r3['cells']
                                                else A1['cells'][bk]['vs_rank_alloc_16x'])}
# paired shared-output vs per-term output comparison (§104 called it a tie)
C = torch.load(f'{QK}/qk_termcompress_ce2.pt', map_location='cpu', weights_only=False)['CE']
def pd(a, b):
    d = (C[a]-C[b]).flatten().double()
    return [round(float(d.mean()), 4), round(float(d.std()/np.sqrt(d.numel())), 5)]
A1['shared_vs_perterm_output_paired'] = {
    bt: pd(f'3c_sharedout_125pct_{bt}', f'3b_perterm_125pct_{bt}') for bt in ['128x', '16x', '4x']}
r['attack1_allocation_fairness'] = A1
r['meta']['scripts'] = ['qk_redteam_tc.py', 'qk_redteam_tc_2.py', 'qk_redteam_tc_3.py']
r['meta']['peak_gpu_MiB'] = max(r['meta'].get('peak_gpu_MiB', 0), r2.get('peak_gpu_MiB', 0),
                                r3.get('peak_gpu_MiB', 0))
json.dump(r, open(f'{QK}/qk_redteam_tc.json', 'w'), indent=1)
print('best 16x overall:', bk, allc[bk], A1['best_variant'])
print('shared-vs-perterm paired:', A1['shared_vs_perterm_output_paired'])
