"""Sign provenance of post-city donor scalar contributions."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh]
a=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);gain=float(torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)['lambdas'][0]);records=[];checks=[]
for group in range(4):
 cells=[];totals=[]
 for i in range(group*24,(group+1)*24):
  n=len(rows[i]['ids']);pos=int(a['cue_positions'][i]);j=rows[i]['donor_id'];sgn=-1 if i%2==0 else 1
  value=(a['modes'][j,pos+1:n]-a['modes'][i,pos+1:n]).sum(-1)*sgn
  route=a['gamma'][i,n-1,pos+1:n]*gain/a['rho9'][i,pos+1:n];con=route*value
  parts=torch.stack([con[(value>=0)&(route>=0)].sum(),con[(value<0)&(route>=0)].sum(),con[(value>=0)&(route<0)].sum(),con[(value<0)&(route<0)].sum()])
  truth=a['final_source_mode_contributions'][i,pos+1:n].sum()*sgn;checks.append(float(abs(parts.sum()-truth)/abs(truth).clamp_min(1e-30)));cells.append(parts);totals.append(truth)
 cells=torch.stack(cells);totals=torch.stack(totals)
 records.append(dict(group=group,mean_contributions=cells.mean(0).tolist(),aligned_fractions=(cells.T@totals/totals.square().sum()).tolist(),mean_postcity_write=float(totals.mean())))
assert max(checks)<1e-9
out=dict(quadrants=['value_positive_route_positive','value_negative_route_positive','value_positive_route_negative','value_negative_route_negative'],max_accounting_error=max(checks),records=records,scope='Exact post-city final-write sign accounting on reused native cache; not an independently validated semantic split.')
(P/'PHI4_SIGN_ORIGIN_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
