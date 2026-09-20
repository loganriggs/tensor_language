"""Audit saved derivatives without changing the original native-run verdict."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
A = P.parent / 'bilinear_quotient/circuits/followups'
source = A / 'six_source_complement_v1_result.json'
r = json.loads(source.read_text())
files = {'opened': 'five_source_modal_null_v1_result.json',
         'ood_opened': 'source_ood_v1_result.json'}
refs = {d: {(c['panel'], c['role'], c['template']): c
            for c in json.loads((A / f).read_text())['contexts']}
        for d, f in files.items()}
errors, cast_errors = [], []
for c in r['contexts']:
    old = refs[c['dataset']][c['panel'], c['role'], c['template']]
    for field in ['gradient', 'hessian']:
        actual = np.asarray(c[field], dtype=np.float64)
        actual = actual[:, :, :5] if field == 'gradient' else actual[:, :, :5, :5]
        expected = np.asarray(old[field], dtype=np.float64)
        errors.append(float(np.max(np.abs(actual - expected))))
        cast_errors.append(float(np.max(np.abs(actual - expected.astype(np.float32)))))
assert max(errors) < 1e-8
assert max(cast_errors) == r['max_old_derivative_replay']
assert max(x['absolute'] for x in r['delta_closures']) > 1e-5
out = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
           reference_sha256={d: hashlib.sha256((A/f).read_bytes()).hexdigest() for d,f in files.items()},
           max_float64_derivative_replay=max(errors),
           reproduced_float32_comparison_error=max(cast_errors),
           delta_closure_maxima={k: max(x[k] for x in r['delta_closures'])
                                for k in ['absolute', 'relative']},
           finite_difference_maxima={k: max(x[k] for x in r['finite_checks'])
                                     for k in ['gradient', 'hessian']},
           original_instrument_valid=False,
           verdict='Comparator dtype bug confirmed; independent absolute cast-closure gate still fails. No promotion.')
(P/'SIX_SOURCE_INSTRUMENT_AUDIT_V1.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
