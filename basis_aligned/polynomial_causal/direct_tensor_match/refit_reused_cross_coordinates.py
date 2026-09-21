from pathlib import Path
import json,time,torch,sys
from reused_cross_problem import problem
from masked_cross_refit import solve
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
version=sys.argv[1] if len(sys.argv)>1 else 'V1'
plan=json.loads((P/f'REUSED_CROSS_REFIT_PLAN_{version}.json').read_text());start=time.monotonic();items,outside,meta=problem();scores=torch.stack([((x['left'].T@x['residual']@x['right']) if plan.get('mask_kind')=='residual_correlation' else x['theta']).square().sum(0)/x['norm'] for x in items]);mask=torch.zeros_like(scores,dtype=torch.bool).flatten();mask[scores.flatten().topk(plan['positions']).indices]=True;mask=mask.reshape_as(scores);error=outside;rows=[]
for j,x in enumerate(items):
 coefficients,diag=solve(x['left'],x['right'],x['residual'],mask[j],ridge=plan['ridge'],steps=plan['steps'],tolerance=plan['tolerance']);residual=x['residual']-x['left']@coefficients@x['right'].T;piece=float(2*residual.square().sum()/(3*x['norm']));error+=piece
 zero=float(2*x['residual'].square().sum()/(3*x['norm']));assert piece<=zero+1e-8
 rows.append(dict(pair=j+1,products=int(mask[j].sum()),cross_squared_error=piece,zero_coefficient_squared_error=zero,**diag))
out=dict(plan=plan,primary=meta['primary'],records=rows,coefficient_error=error**.5,coefficient_guard=meta['plan']['covariance_guard'],coefficient_pass=error**.5<=meta['plan']['covariance_guard'],seconds=time.monotonic()-start,scope='Fixed5600crosspositions using existing activations, coefficients refitted. This is a dense-form diagnostic; shared-core update/recompilation, actual execution, component fidelity and native behavior remain untested. Nonconverged solves are not certified optima.')
(P/f'REUSED_CROSS_REFIT_{version}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
