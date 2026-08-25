"""Convert the bilinear_quotient optimal-ablation sweep into frozen anchor files
(spec section 9 / Invariant 4). Run after optimal_ablation_all completes; writes one
json per component into bench/anchors/ with clean CE, mean CE, optimal CE, deltas,
data budget, and provenance. CIs: TODO M1 (bootstrap over eval rows — requires
per-row CE dumps, to be added to the sweep script before anchor freeze)."""
import json, os, time

BQ = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
SRC = BQ + 'optimal_ablation_all_results.json'
DST = os.path.join(os.path.dirname(__file__), 'anchors')

r = json.load(open(SRC))
res = r['results']
os.makedirs(DST, exist_ok=True)
meta = {'anchor_version': 'v0-practice', 'source': SRC,
        'generated': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'clean_ce': r.get('clean'), 'data_budget': r.get('data_budget'),
        'partial': bool(r.get('partial')), 'n_components': len(res),
        'reference': 'Li & Janson arXiv 2409.09951'}
json.dump(meta, open(os.path.join(DST, '_meta.json'), 'w'), indent=1)
for name, v in res.items():
    json.dump({'component': name, 'clean': r.get('clean'),
               'ce_mean': v['ce_mean'], 'ce_opt': v['ce_opt'],
               'delta_mean': v['delta_mean'], 'delta_opt': v['delta_opt'],
               'opt_over_mean': v['opt_over_mean'],
               'converged_note': f"drift {v['rel_drift_from_mean']}"},
              open(os.path.join(DST, f'{name}.json'), 'w'), indent=1)
print(f"wrote {len(res)} anchors + _meta.json (partial={meta['partial']})")
