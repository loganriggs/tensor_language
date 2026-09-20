import json,time,itertools
from pathlib import Path
import torch
from core import covariance_metric
from sweep import fit,target_cases

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);p=Path(__file__).resolve().parent;out=p/'TOY_COVARIANCE_SWEEP_V1.json';assert not out.exists();torch.manual_seed(946)
 q,_=torch.linalg.qr(torch.randn(6,6));population=q@torch.diag(torch.tensor([1.,.5,.2,.08,.02,.005]))@q.T
 calibration=torch.randn(512,6)@torch.linalg.cholesky(population).T;empirical=calibration.T@calibration/512;ridge=float(empirical.trace()/6*.001);cov=empirical+ridge*torch.eye(6)
 records=[];start=time.perf_counter()
 for case in target_cases():
  for objective,optimizer,variant,seed in itertools.product(['isotropic','empirical_covariance_gaussian'],['adam','muon'],['matched','wide'],[0,1]):
   M=None if objective=='isotropic' else covariance_metric(6,case['degree'],cov)
   row,_=fit(case['target'],6,case['degree'],case['kind'],case['width']*(1 if variant=='matched' else 2),optimizer,.05,seed,1200,M_override=M)
   row.update(case=case['name'],objective=objective,variant=variant);records.append(row)
   print(json.dumps({k:row[k] for k in ['case','objective','variant','optimizer','seed','relative_error','isotropic_gaussian_error']}),flush=True)
  out.write_text(json.dumps(dict(records=records,covariance=cov.tolist(),ridge=ridge,calibration_rows=512,population_eigenvalues=[1.,.5,.2,.08,.02,.005],seconds=time.perf_counter()-start,scope='Synthetic calibration covariance; exact Gaussian moment metric built from its second moment. Not empirical fourth/eighth moment matching, and not native activation covariance.'),indent=2)+'\n')
if __name__=='__main__':main()
