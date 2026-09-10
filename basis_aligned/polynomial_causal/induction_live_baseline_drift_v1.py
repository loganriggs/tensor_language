"""Measure full attention-write feedback in saved R594 joint interventions.

No inference that this full-write drift equals equality-component drift.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from induction_context_transport_v2 import digest


def main():
    poly = Path(__file__).parent
    envelope = json.loads((poly / 'INDUCTION_R594_MANAGED_RESULT.json').read_text())
    receipt = json.loads((poly / 'induction_centered_fixed_geometry_rung594_receipt.json').read_text())
    root = Path(envelope['raw_root']) / 'induction_centered_fixed_geometry_rung594_evidence/FIT'
    arrays = {}
    for name in ('layer_before.npy', 'layer_total.npy', 'layer_after.npy'):
        assert digest(root / name) == receipt['evidence_files']['FIT/' + name]['sha256']
        arrays[name] = np.load(root / name, mmap_mode='r')
    selected = [json.loads(x) for x in (poly / 'INDUCTION_CONTEXT_TRANSPORT_V2_ROWS.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in (root / 'directed_records.jsonl').read_text().splitlines()]
    locations = {r['directed_id']: i for i, r in enumerate(rows)}
    grouped = defaultdict(list)
    maximum_identity_error = 0.
    for row in selected:
        i = locations[row['directed_id']]
        before = np.asarray(arrays['layer_before.npy'][i], dtype=np.float64)
        total = np.asarray(arrays['layer_total.npy'][i], dtype=np.float64)
        after = np.asarray(arrays['layer_after.npy'][i], dtype=np.float64)
        assert not np.any(total[0]) and np.array_equal(before[0], after[0])
        drift = before[3] - before[0]
        assert not np.any(drift[0])  # no earlier edit before the first selected site
        observed = after[3] - after[0]
        rounding = after[3] - before[3] - total[3]
        error = float(np.max(np.abs(observed - drift - total[3] - rounding)))
        assert error < 1e-12
        maximum_identity_error = max(maximum_identity_error, error)
        for li, layer in enumerate((5, 7, 8)):
            d, p = drift[li], total[3, li]
            dn, pn = float(np.linalg.norm(d)), float(np.linalg.norm(p))
            grouped[(row['cell'], layer)].append(dict(
                drift_norm=dn, planned_norm=pn,
                ratio=dn / pn if pn else None,
                cosine=float(d @ p / (dn * pn)) if dn * pn else None))
    cells = {}
    for (key, layer), values in sorted(grouped.items()):
        out = dict(groups=len(values), layer=layer,
                   nonzero_plans=sum(v['planned_norm'] > 1e-8 for v in values))
        for metric in ('drift_norm', 'planned_norm', 'ratio', 'cosine'):
            a = [v[metric] for v in values if v[metric] is not None]
            out['median_' + metric] = float(np.median(a)) if a else None
        cells[key + '|L' + str(layer)] = out
    result = dict(experiment='induction_live_baseline_drift_v1', model_forwards=0,
                  post_result_exploratory=True, rows=len(selected), cells=cells,
                  maximum_additive_identity_error=maximum_identity_error,
                  first_site_drift_exactly_zero=True,
                  equality_component_drift_identified=False,
                  original_instrument_valid=True,
                  source_sha256=digest(__file__),
                  context_rows_sha256=digest(poly / 'INDUCTION_CONTEXT_TRANSPORT_V2_ROWS.jsonl'))
    with (poly / 'INDUCTION_LIVE_BASELINE_DRIFT_V1_RESULT.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}, indent=2))
    for layer in (5,7,8):
        for kind in ('selector_payload', 'copy_relation'):
            s=[v for k,v in cells.items() if k.startswith(kind) and v['layer']==layer]
            print(kind,layer,{metric:[min(v[metric] for v in s if v[metric] is not None),max(v[metric] for v in s if v[metric] is not None)] for metric in ('median_drift_norm','median_planned_norm','median_ratio')})


if __name__ == '__main__':main()
