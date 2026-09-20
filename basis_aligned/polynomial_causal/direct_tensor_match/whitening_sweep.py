"""Same physical initial function and covariance objective, different coordinates."""
import json,time,itertools
from pathlib import Path
import torch
from core import Model,metric,inner,covariance_metric
from coordinates import transform_matrix,transform_model_state
from sweep import target_cases,fit

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'WHITENING_SWEEP_V1.json';assert not out.exists();cov=torch.tensor(json.loads((p/'TOY_COVARIANCE_SWEEP_V1.json').read_text())['covariance'],dtype=torch.float64);L=torch.linalg.cholesky(cov);rows=[];start=time.perf_counter()
 for case in target_cases():
  degree=case['degree'];T=transform_matrix(L,degree);Ti=transform_matrix(torch.linalg.inv(L),degree);target=case['target'];M=covariance_metric(6,degree,cov);G=metric(6,degree)
  for opt,variant,seed in itertools.product(['adam','muon'],['matched','wide'],[0,1,2]):
   width=case['width']*(1 if variant=='matched' else 2);torch.manual_seed(seed);initial=Model(6,3,case['kind'],width,degree)
   for coord in ['raw','whitened']:
    initial_state=initial.state_dict() if coord=='raw' else transform_model_state(initial,L)
    row,state=fit(target if coord=='raw' else target@T,6,degree,case['kind'],width,opt,.05,seed,1200,M_override=M if coord=='raw' else G,initial_state=initial_state)
    student=Model(6,3,case['kind'],width,degree);student.load_state_dict(state);coeff=student().detach();coeff=coeff if coord=='raw' else coeff@Ti;delta=coeff-target
    row.update(case=case['name'],variant=variant,coordinates=coord,physical_isotropic_error=float((inner(delta,delta,G)/inner(target,target,G)).sqrt()),physical_covariance_error=float((inner(delta,delta,M)/inner(target,target,M)).sqrt()))
    assert abs(row['physical_covariance_error']-row['relative_error'])<1e-7;rows.append(row)
   assert abs(rows[-1]['history'][0]['error']-rows[-2]['history'][0]['error'])<1e-9
  out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='120paired-coordinate fits; same physical initial functions and covariance objective, no penalties. Coordinate-dependent optimizers and fixed learning rate are the intervention.'),indent=2)+'\n');print(case['name'],flush=True)
if __name__=='__main__':main()
