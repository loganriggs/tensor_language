"""Convergence guard and exploratory minimal stable grouping across two metrics.

Grouping criterion is post-screen exploratory: <=10% relative function change
in both shared geometries, denominator min of both group norms. No semantic claim.
"""
from pathlib import Path
import json,torch,itertools
from functools import lru_cache
from scipy.optimize import linear_sum_assignment
from cp_local_als import fit
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_METRIC_STABLE_GROUPS_V1.json';assert not out.exists()
r=json.loads((p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.json').read_text());e=torch.load(p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.pt',weights_only=True)['programs']
convergence=[]
for name,q in e.items():
 winner=int(q['errors'].argmin());original=[v[winner:winner+1] for v in q['factors']];core=torch.einsum('ik,jk,lk->ijl',*q['roots'])
 refined,err,hist=fit(core,original,steps=2000,checkpoints=(0,2000))
 sim=torch.ones(6,6,dtype=torch.float64)
 for a,b in zip(original,refined):sim*=(a[0]/a[0].norm(dim=0)).T@(b[0]/b[0].norm(dim=0))
 ii,jj=linear_sum_assignment(-sim.numpy())
 convergence.append(dict(metric=name,minimum_term_cosine_after_2000=float(sim[ii,jj].min()),old_loss=float(q['errors'][winner]),new_loss=float(err[0])))
permutation=next(x['permutation'] for x in r['comparisons'] if x['left']=='half0' and x['right']=='half1' and x['geometry']=='full')
terms={}
for geom in ['full','isotropic']:
 roots=e[geom]['roots'];pairs=[]
 for which in ['half0','half1']:
  matrices=[root@coeff for root,coeff in zip(roots,e[which]['coefficients'])]
  t=torch.einsum('ir,jr,kr->rijk',*matrices)
  if which=='half1':t=t[permutation]
  pairs.append(t)
 terms[geom]=pairs
subsets=[]
for mask in range(1,64):
 ids=[j for j in range(6) if mask&(1<<j)];checks={}
 for geom,(a,b) in terms.items():
  aa=a[ids].sum(0);bb=b[ids].sum(0);den=min(float(aa.norm()),float(bb.norm()))
  checks[geom]=float((aa-bb).norm())/max(den,1e-30)
 subsets.append(dict(mask=mask,components=ids,relative_function_changes=checks,stable=max(checks.values())<=.1))
by_mask={v['mask']:v for v in subsets}
@lru_cache(None)
def partition(mask):
 if mask==0:return ()
 first=mask&-mask;best=None
 for sub in range(1,64):
  if not sub&first or sub&mask!=sub or not by_mask[sub]['stable']:continue
  rest=partition(mask^sub)
  if rest is None:continue
  candidate=(sub,)+rest
  if best is None or len(candidate)>len(best) or (len(candidate)==len(best) and max(max(by_mask[k]['relative_function_changes'].values()) for k in candidate)<max(max(by_mask[k]['relative_function_changes'].values()) for k in best)):best=candidate
 return best
chosen=partition(63)
result=dict(convergence=convergence,convergence_guard_pass=all(v['minimum_term_cosine_after_2000']>.9999 for v in convergence),alignment_permutation=permutation,singletons=[v for v in subsets if len(v['components'])==1],maximum_stable_partition=[by_mask[k] for k in chosen] if chosen else None,all_subsets=subsets,scope='Exploratory exhaustive partition of six matched CP term functions into stable sums. Labels aligned once in full geometry, same alignment used isotropically. Threshold10% relative change selected for this successor analysis; optimizing grouping on these two splits is discovery, not independent validation. Full graph/teacher selection used all32documents.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='all_subsets'},indent=2))
