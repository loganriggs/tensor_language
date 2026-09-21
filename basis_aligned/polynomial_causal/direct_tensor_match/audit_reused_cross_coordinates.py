from pathlib import Path
import json,torch
from reused_cross_problem import problem
P=Path(__file__).parent
items,outside,meta=problem()
scores=torch.stack([x['theta'].square().sum(0)/x['norm'] for x in items]);budget=5600;indices=scores.flatten().topk(budget).indices;mask=torch.zeros_like(scores,dtype=torch.bool).flatten();mask[indices]=True;mask=mask.reshape_as(scores)
error=outside;rows=[]
for j,x in enumerate(items):
 approx=x['left']@(x['theta']*mask[j])@x['right'].T;piece=float(2*(x['residual']-approx).square().sum()/(3*x['norm']));error+=piece;rows.append(dict(pair=j+1,selected_product_positions=int(mask[j].sum()),cross_squared_error=piece,full_coordinate_replay=x['replay']))
out=dict(primary=meta['primary'],positions=budget,total_positions=scores.numel(),outside_span_error=outside**.5,individual_term_threshold_error=error**.5,prospective_source_multiplications=1047648+3*budget,maximum_source_multiplications=.8*1330560,pairs=rows,scope='Prospective products reuse existingshared128andprivate224activations: no newdense projections. Each retainedpairposition costs1product+2readoutmults. Sharedquadraticcore must be updated to cancel induced shared-shared terms and recompiled; not yet executed. Individualtermthreshold is a heuristic, not optimal sparseerror. Dense-atom199vs34bound does not apply to this cheaper family. Next action is fixed-support coefficient refit or sparsity optimization, with full reconstruction checks.')
(P/'REUSED_CROSS_COORDINATE_SCREEN_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
