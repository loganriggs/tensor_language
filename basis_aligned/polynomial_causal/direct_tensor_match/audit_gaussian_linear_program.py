"""Audit saved programs and necessity of their four quadratic products."""
import json
from pathlib import Path
import torch
from export_centered_dag import build
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);archive=torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True);panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];cov=panels[0]['covariance'].double();rows=[]
 for metric,program in archive['programs'].items():
  s={k:v.double() for k,v in program.items()};dag,out=build(s);cost=dag.cost(out);replay=[];errors=[];ablations=[];M=cov if metric=='centered' else torch.eye(len(cov),dtype=torch.float64)*cov.trace()/len(cov);a,b,c=s['quadratic_left'],s['quadratic_right'],s['quadratic_writer'];qmean=((a@M)*b).sum(1)
  for xpanel,y in zip(panels,targets):
   x=xpanel['rows'].double();pred=evaluate(s,x);replay.append(float((dag.evaluate(out,x[:16])-pred[:16]).norm()/pred[:16].norm()));errors.append(float((pred-y).norm()/y.norm()));delta=x-s['mu'];q=(delta@a.T)*(delta@b.T)-qmean
   removal=[]
   for k in range(4):removal.append(float((pred-q[:,k,None]*c[:,k]-y).norm()/y.norm()))
   ablations.append(dict(individual_mean_preserving_removal_errors=removal,all_products_removed_error=float((pred-q@c.T-y).norm()/y.norm())))
  assert max(replay)<1e-10 and cost['stored_coefficients']==34560 and cost['products']==4
  rows.append(dict(metric=metric,cost=cost,replay=max(replay),errors=errors,ablations=ablations))
 result=dict(records=rows,scope='Independent CPU graph replay; ablations preserve Gaussian component mean without refit. Reconstruction necessity, not selective causal or semantic identification.');(P/'GAUSSIAN_LINEAR_PROGRAM_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
