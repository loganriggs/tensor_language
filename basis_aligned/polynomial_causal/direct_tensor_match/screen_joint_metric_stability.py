"""Conditional metric/split sensitivity of a fixed local six-product dictionary.

pred_a: Gram truncation energy replay<1e-9 and transformed warmstart replay<1e-8.
pred_b: half0/half1 best-fit component matches >=.95 in both common geometries.
pred_c: each half's fitted function has opposite-half error <=1.2 that half's
own best fit. Check winning-loss relative change last1000sweeps<1e-6; failures
remain unconverged, not structural nulls. Frozen original selection saw all32
calibration documents, so this is conditional sensitivity, not fresh discovery.
"""
from pathlib import Path
import json,itertools,torch,time
from scipy.optimize import linear_sum_assignment
from cp_local_als import fit
from joint_product_refactor import best_subset
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.json';assert not out.exists();start=time.perf_counter()
old=torch.load(p/'MIDPOINT_JOINT_STABILITY_REFINED_V1.pt',weights_only=True)
e=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];ids=old['ids'];A=e['A'][:,ids];B=e['B'][:,ids];W=e['reduced_writers'][:,ids]
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
# Physical reader/writer combinations; truncate only numerically null directions.
warm=[torch.linalg.pinv(root,rtol=1e-6)@v[0] for root,v in zip(old['roots'],old['factors'])]
metrics={};checks=[]
for label,sl in [('full',slice(None)),('half0',slice(0,16)),('half1',slice(16,32)),('isotropic',None)]:
 if sl is None:readers=[A,B,S@W]
 else:
  n=rows['n'][sl].flatten(0,1).double();m=rows['m'][sl].flatten(0,1).double();n-=n.mean(0);m-=m.mean(0)
  readers=[n@A/len(n)**.5,m@B/len(m)**.5,S@W]
 grams=[v.T@v for v in readers];roots=[]
 for g in grams:
  ev,U=torch.linalg.eigh(g);ev=torch.where(ev>1e-12*ev.max(),ev,0);roots.append(ev.sqrt()[:,None]*U.T)
 core=torch.einsum('ik,jk,lk->ijl',*roots);energy=(grams[0]*grams[1]*grams[2]).sum();checks.append(float(abs(core.square().sum()-energy)/energy))
 metrics[label]=dict(roots=roots,core=core,grams=grams)
# Mapping old fit through common roots must preserve its actual output tensor.
oldpred=torch.einsum('ir,jr,kr->ijk',*[v[0] for v in old['factors']])
mapback=torch.einsum('ir,jr,kr->ijk',*[root@q for root,q in zip(old['roots'],warm)])
warm_replay=float((oldpred-mapback).norm()/oldpred.norm());assert warm_replay<1e-8
exports={};records=[]
for num,label in enumerate(['full','half0','half1','isotropic']):
 info=metrics[label];roots=info['roots'];core=info['core'];base=best_subset(core,roots[0].T,roots[1].T,6)
 gen=torch.Generator().manual_seed(261323+num);initial=[torch.randn(8,8,6,dtype=torch.float64,generator=gen) for _ in range(3)]
 for j in range(3):initial[j][0]=roots[j]@warm[j]
 for v,q in zip(initial,[base[2].T,base[3].T,base[4]]):v[1]=q
 factors,errors,history=fit(core,initial)
 best=int(errors.argmin());co=[torch.linalg.pinv(root,rtol=1e-6)@v[best] for root,v in zip(roots,factors)]
 # Compare using original factor-coordinate coefficients, not incompatible whitened bases.
 replay=torch.einsum('ir,jr,kr->ijk',*[root@q for root,q in zip(roots,co)])
 err=float((replay-core).square().sum()/core.square().sum());assert abs(err-float(errors[best]))<1e-9
 before=next(h for h in history if h['step']==11000)['best'][best];convergence=(before-float(errors[best]))/float(errors[best])
 exports[label]=dict(coefficients=co,roots=roots,factors=factors,errors=errors,history=history)
 records.append(dict(metric=label,winner=best,relative_squared_error=err,relative_norm_error=err**.5,winning_loss_improvement_last1000=convergence,converged=convergence<1e-6,restart_relative_squared_errors=errors.tolist()))
 print('FIT',records[-1],flush=True)
comparisons=[]
for left,right in itertools.combinations(exports,2):
 for geometry in ['full','isotropic']:
  roots=metrics[geometry]['roots'];aa=[r@q for r,q in zip(roots,exports[left]['coefficients'])];bb=[r@q for r,q in zip(roots,exports[right]['coefficients'])]
  sim=torch.ones(6,6,dtype=torch.float64)
  for a,b in zip(aa,bb):sim*=(a/a.norm(dim=0)).T@(b/b.norm(dim=0))
  ii,jj=linear_sum_assignment(-sim.numpy());cos=sim[ii,jj]
  pa=torch.einsum('ir,jr,kr->ijk',*aa);pb=torch.einsum('ir,jr,kr->ijk',*bb)
  comparisons.append(dict(left=left,right=right,geometry=geometry,min_term_cosine=float(cos.min()),mean_term_cosine=float(cos.mean()),function_difference_relative_to_target=float((pa-pb).norm()/metrics[geometry]['core'].norm()),permutation=jj.tolist(),term_cosines=cos.tolist()))
transfer=[]
for fitted in exports:
 for evaluated in metrics:
  d=metrics[evaluated];pred=torch.einsum('ir,jr,kr->ijk',*[r@q for r,q in zip(d['roots'],exports[fitted]['coefficients'])]);error=float((pred-d['core']).norm()/d['core'].norm())
  own=next(r['relative_norm_error'] for r in records if r['metric']==evaluated)
  transfer.append(dict(fit=fitted,evaluate=evaluated,relative_norm_error=error,error_ratio_to_own_fit=error/own))
pred=dict(pred_a_instrument=max(checks)<1e-9 and warm_replay<1e-8,pred_b_split_components=all(r['min_term_cosine']>=.95 for r in comparisons if r['left']=='half0' and r['right']=='half1'),pred_c_cross_split_function=all(r['error_ratio_to_own_fit']<=1.2 for r in transfer if {r['fit'],r['evaluate']}=={'half0','half1'}))
result=dict(predictions=pred,records=records,comparisons=comparisons,transfer=transfer,gram_replay=max(checks),warm_replay=warm_replay,seconds=time.perf_counter()-start,scope='Fixed teacher/selected neighborhood, four input metrics, same output metric. Splits only covariance estimation; original upstream feature selection used all documents. Common-geometry component matching and cross-metric loss, not fresh OOD validation, native interventions or semantics.')
out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(programs=exports,ids=ids),p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.pt');print('RESULT',pred,flush=True)
