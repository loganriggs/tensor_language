"""Posthoc optimizer redteam on opened six-source response coefficients."""
import json
import hashlib
from pathlib import Path
import numpy as np
from quadratic_edit_upper_bound import upper_bound, controls

P = Path(__file__).resolve().parent
source = P.parent/'bilinear_quotient/circuits/followups/six_source_complement_v1r1_result.json'
r = json.loads(source.read_text())
assert r['predictions']['pred_a_instrument']
lookup = {(c['dataset'],c['panel'],c['role'],c['family'],c['arm']): c for c in r['records']}
cells, bounds = [], []
for c in r['contexts']:
    g,h,a = [np.array(c[k]) for k in ['gradient','hessian','amplitudes']]
    effects = -np.einsum('boi,bi->bo',g,a)-.5*np.einsum('bi,boij,bj->bo',a,h,a)
    pattern = 'SEMANTIC_PORT_FRESH' if c['dataset']=='opened' else 'SOURCE_OOD_V1'
    rows = json.loads((P/f"{pattern}_{c['panel'].upper()}_ROWS.json").read_text())
    rows = [x for x in rows if x['template']==c['template']]
    for family in dict.fromkeys(x['family'] for x in rows):
        ids = [i for i,x in enumerate(rows) if x['family']==family]
        key = (c['dataset'],c['panel'],c['role'],family)
        baseline = np.array(lookup[key+('unitB',)]['target'])[:,0]
        actual = np.array(lookup[key+('quad6',)]['target'])[:,0]
        den = float(baseline@baseline)
        upper = 0.
        for index, weight in zip(ids, baseline):
            b = upper_bound(g[index],h[index],np.array([0.,0.,1.,1.,1.,0.]),multiplier=weight)
            selected = float(weight*effects[index,0])
            assert selected <= b['upper_bound']+1e-7
            bounds.append(dict(**b, selected=selected))
            upper += b['upper_bound']
        cells.append(dict(dataset=c['dataset'],panel=c['panel'],role=c['role'],family=family,
                          upper_retention=upper/den, selected_quadratic_retention=float(effects[ids,0]@baseline/den),
                          native_retention=float(actual@baseline/den)))
out = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), controls=controls(),
           scope='Opened-data diagnostic; bound on 8% per-input quadratic-selector feasible set, not native 10% cell constraints. Floating-point LP, not interval arithmetic.',
           cells=cells, bounds=bounds, lp_calls=len(bounds),
           max_dual_gap=max(b['dual_gap'] for b in bounds),
           max_primal_violation=max(b['primal_violation'] for b in bounds),
           cells_upper_below_80=sum(c['upper_retention']<.8 for c in cells))
(P/'SIX_SOURCE_UPPER_BOUND_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['cells','bounds']},indent=2))
print(json.dumps([c for c in cells if c['native_retention']<.8],indent=2))
