"""Continue best and least-aligned near-best ALS fits before calling instability."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
from scipy.optimize import linear_sum_assignment
out=p/'MIDPOINT_JOINT_STABILITY_CONVERGENCE_V1.json';assert not out.exists()
e=torch.load(p/'MIDPOINT_JOINT_PRODUCT_ALS_FACTORS_V2.pt',weights_only=True)[0];audit=json.loads((p/'MIDPOINT_JOINT_COMPONENT_STABILITY_V1.json').read_text())['records'][0]
best=audit['best_restart'];worst=min(audit['matches'],key=lambda r:r['min_matched_term_cosine'])['restart'];ids=[best,worst]
a,b,c=[v[ids].clone() for v in e['factors']];core=e['core'];den=core.square().sum();rank=6;history=[]
for step in range(10001):
 if step in [0,100,500,2000,10000]:
  pred=torch.einsum('bir,bjr,bkr->bijk',a,b,c);errors=(pred-core).square().sum((1,2,3))/den
  sim=torch.ones(rank,rank,dtype=a.dtype)
  for v in [a,b,c]:
   vv=v/v.norm(dim=1,keepdim=True);sim*=vv[0].T@vv[1]
  ii,jj=linear_sum_assignment(-sim.numpy());history.append(dict(additional_sweeps=step,relative_squared_errors=errors.tolist(),min_matched_term_cosine=float(sim[ii,jj].min()),mean_matched_term_cosine=float(sim[ii,jj].mean())))
 if step==10000:break
 def solve(rhs,x,y):
  gram=torch.einsum('bdi,bdj->bij',x,x)*torch.einsum('bdi,bdj->bij',y,y);ridge=1e-12*gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-30)
  return torch.linalg.solve(gram+ridge[:,None,None]*torch.eye(rank,dtype=gram.dtype),rhs.transpose(1,2)).transpose(1,2)
 a=solve(torch.einsum('ijk,bjr,bkr->bir',core,b,c),b,c);b=solve(torch.einsum('ijk,bir,bkr->bjr',core,a,c),a,c);c=solve(torch.einsum('ijk,bir,bjr->bkr',core,a,b),a,b)
 norms=torch.stack([v.norm(dim=1).clamp_min(1e-30) for v in [a,b,c]]);target=norms.prod(0).pow(1/3)
 a*=(target/norms[0])[:,None,:];b*=(target/norms[1])[:,None,:];c*=(target/norms[2])[:,None,:]
result=dict(restarts=ids,history=history,scope='Selected worst matched near-best endpoint and best endpoint from prior fixed screen; extra fitting on same core, no new data. Tests whether endpoint component discrepancy persists after more convergence; cannot prove robustness to data/metric changes or semantic meaning.')
out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(factors=[a,b,c],core=core,roots=e['roots'],ids=e['ids']),p/'MIDPOINT_JOINT_STABILITY_REFINED_V1.pt');print(out.read_text())
