"""Permutation/scale-invariant comparison of approximate CP computations.

Agreement across optimizer starts is not held-out semantic identification.
"""
from pathlib import Path
import json,torch
from scipy.optimize import linear_sum_assignment
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_JOINT_COMPONENT_STABILITY_V1.json';assert not out.exists()
data=torch.load(p/'MIDPOINT_JOINT_PRODUCT_ALS_FACTORS_V2.pt',weights_only=True);records=[]
for group,e in data.items():
 a,b,c=e['factors'];errors=e['errors'];best=int(errors.argmin());near=(errors<=1.01*errors[best]).nonzero().flatten().tolist()
 normalized=[v/v.norm(dim=1,keepdim=True).clamp_min(1e-30) for v in [a,b,c]]
 matches=[]
 for j in near:
  if j==best:continue
  similarity=torch.ones(6,6,dtype=a.dtype)
  for v in normalized:similarity*=v[best].T@v[j]
  ii,jj=linear_sum_assignment(-similarity.numpy());score=similarity[ii,jj]
  matches.append(dict(restart=j,relative_loss_to_best=float(errors[j]/errors[best]),min_matched_term_cosine=float(score.min()),mean_matched_term_cosine=float(score.mean()),permutation=jj.tolist()))
 singular=[torch.linalg.svdvals(v[best]) for v in normalized]
 pred=torch.einsum('ir,jr,kr->ijk',a[best],b[best],c[best]);err=float((pred-e['core']).square().sum()/e['core'].square().sum());assert abs(err-float(errors[best]))<1e-12
 records.append(dict(group=group,best_restart=best,near_best_restarts=near,matches=matches,normalized_factor_min_singular_values=[float(s[-1]) for s in singular],normalized_factor_condition_numbers=[float(s[0]/s[-1]) for s in singular],numerical_full_column_rank=[bool(s[-1]>1e-8*s[0]) for s in singular],relative_squared_error=err,all_near_best_terms_cosine_above_point99=all(r['min_matched_term_cosine']>.99 for r in matches)))
result=dict(records=records,scope='Same metric, neighborhood and teacher. Terms matched by signed three-mode tensor cosine with optimal permutation; invariant to compensating factor signs/scales. Near-best defined as <=1% above best loss before checking matches. Numerical full column rank supports but does not rigorously certify the sufficient Kruskal inequality for uniqueness of the represented rank6 tensor; does not guarantee unique approximation optimum or semantic identity.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
