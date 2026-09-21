"""Exact objective tradeoff among already frozen equal-cost programs."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;g=json.loads((p/'FULL_QUADRATIC_GEOMETRY_AUDIT_V1.json').read_text());r=json.loads((p/'FULL_CHANNEL_RESPONSE_REFIT_V1.json').read_text());l=json.loads((p/'LEARNED_FULL_QUADRATIC_V1.json').read_text());response={'pruned':r['records'][0]['response_error'],'response_refit':r['records'][2]['response_error'],'learned':next(v['response_error'] for v in l['records'] if v['start']==l['selected'])};components={x['program']:dict(covariance=x['covariance_coefficient_error']**2,isotropic=x['isotropic_coefficient_error']**2,response=response[x['program']]**2) for x in g['records']};rows=[]
for eta in [0,.01,.02,.1,1.]:
 scores={name:v['covariance']+v['response']+eta*v['isotropic'] for name,v in components.items()};rows.append(dict(isotropic_weight=eta,scores=scores,best=min(scores,key=scores.get)))
a,b=components['learned'],components['response_refit'];threshold=(b['covariance']+b['response']-a['covariance']-a['response'])/(a['isotropic']-b['isotropic']);result=dict(components=components,rows=rows,learned_vs_fixed_response_crossover_weight=threshold,scope='Post-result algebraic comparison of frozen programs, not an optimization or heldout selection rule. Shows relative normalized-loss weighting sufficient to reverse their training ranking; no claim that choosing a weight repairs native behavior.')
(p/'FULL_QUADRATIC_METRIC_TRADEOFF_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
