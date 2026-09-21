"""Same-budget fit-metric Pareto audit, separate from native circuit evidence."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;g=json.loads((p/'FULL_QUADRATIC_GEOMETRY_AUDIT_V1.json').read_text());r=json.loads((p/'FULL_CHANNEL_RESPONSE_REFIT_V1.json').read_text());l=json.loads((p/'LEARNED_FULL_QUADRATIC_V1.json').read_text());m=json.loads((p/'FULL_QUADRATIC_MULTIGEOMETRY_FIT_V1.json').read_text());responses={'pruned':r['records'][0]['response_error'],'response_refit':r['records'][2]['response_error'],'learned':next(t['response_error'] for t in l['records'] if t['start']==l['selected'])};rows=[]
for v in g['records']:rows.append(dict(program=v['program'],covariance_error=v['covariance_coefficient_error'],isotropic_error=v['isotropic_coefficient_error'],response_error=responses[v['program']]))
for v in m['records']:rows.append(dict(program=v['arm'],**{k:v[k] for k in ('covariance_error','isotropic_error','response_error')}))
keys=('covariance_error','isotropic_error','response_error')
for row in rows:row['dominated_by']=[other['program'] for other in rows if other is not row and all(other[k]<=row[k] for k in keys) and any(other[k]<row[k] for k in keys)]
result=dict(records=rows,stored_floats_each=14067072,scope='All errors use the same original target, historical coordinates/directions, and affine-corrected program budget. Coefficient comparisons themselves exclude affine terms. Dominance concerns these three fitting measures only, not native/OOD/causal behavior or identifiable circuit units.')
(p/'FULL_QUADRATIC_MULTIGEOMETRY_PARETO_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
