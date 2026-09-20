"""Necessary three-root error floors; finite-panel results are not population bounds."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2)
 s={k:v.double() for k,v in torch.load(P/'EXTRACTED_SCALAR_MODES_V1.pt',weights_only=True)['program'].items()}
 qs=[]
 for w in s['quartic_readout']:
  q=s['root_left'].T@torch.diag(w)@s['root_right'];qs.append(((q+q.T)/2).flatten())
 sv=torch.linalg.svdvals(torch.stack(qs))
 out=dict(coefficient_output_unfolding_singular_values=sv.tolist(),rank3_relative_frobenius_lower_bound=float(sv[3]/sv.norm()),scope='Necessary lower bound for three shared quartic functions with linear readout under equal-output coefficient Frobenius metric. Does not assert attainability by product roots or control native behavior. Fixed primitive coordinates only.',panels=[])
 for panel in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  x=panel['rows'].double();p=(x@s['A'].T)*(x@s['B'].T);h=(p@s['root_left'].T)*(p@s['root_right'].T);y=h@s['quartic_readout'].T
  nuisance=torch.cat([torch.ones(len(x),1,dtype=x.dtype),p],1)
  solution=torch.linalg.lstsq(nuisance,y,driver='gelsd').solution;res=y-nuisance@solution
  assert float((nuisance.T@res).norm()/(nuisance.norm()*res.norm()))<1e-10
  ev=torch.linalg.svdvals(res)
  out['panels'].append(dict(context=panel['context'],rows=len(x),quartic_residualized_singular_values=ev.tolist(),rank3_relative_residualized_error_floor=float(ev[3]/ev.norm()),rank3_error_floor_relative_total_scalar=float(ev[3]/(y+p@s['quadratic_readout'].T+s['constant']).norm())))
 (P/'SHARED_ROOT_RANK3_FLOOR_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
