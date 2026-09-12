"""Residual-derived rank2 candidate on a saved planted stationary miss.

A eigen residual<=1e-10 and initial nonincrease; B final error<=1e-4;
C at most100 recorded iterations. No native/global guarantee.
"""
from pathlib import Path
import json
import torch
from coupled_quartic_writer_v1 import gram
from planted_quartic_objective_v1 import objective
from quartic_manifold_lbfgs_v1 import fit
from quartic_residual_addition_v1 import scores

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
s=torch.load(P/'QUARTIC_LBFGS_STATIONARY_MISS.pt',weights_only=True)
b,n=s['b'],s['n'];ta=torch.tensor([[1.,.3],[-.2,.9]])
evaluate,energy=objective(s['trueb'],s['truen'],ta)
def tensors(b,n):
    q=torch.einsum('kdi,ki,kei->kde',b,n,b)
    return (torch.einsum('kij,klm->kijlm',q,q)+torch.einsum('kil,kjm->kijlm',q,q)+torch.einsum('kim,kjl->kijlm',q,q))/3
target=torch.einsum('jm,jabcd->mabcd',ta,tensors(s['trueb'],s['truen']))
initial,mix=evaluate(b,n,energy);k=gram(b,n)
weak=int((mix.square().sum(-1)/torch.linalg.inv(k).diagonal()).argmin())
retained=[i for i in range(len(b)) if i!=weak]
f=tensors(b[retained],n[retained]);cross=f.flatten(1)@target.flatten(1).T
remaining_mix=torch.linalg.solve(gram(b[retained],n[retained]),cross)
residual=target-torch.einsum('jm,jabcd->mabcd',remaining_mix,f)
output_gram=residual.flatten(1)@residual.flatten(1).T
_,output_vectors=torch.linalg.eigh(output_gram);direction=output_vectors[:,-1]
scalar=torch.einsum('m,mabcd->abcd',direction,residual)
d=b.shape[-2];operator=scalar.reshape(d*d,d*d)
values,vectors=torch.linalg.eigh(operator);chosen=values.abs().argsort()[-4:]
eigenerror=float((operator@vectors[:,chosen]-vectors[:,chosen]*values[chosen]).norm()/operator.norm())
bank=[b[weak]];weights=[n[weak]]
for idx in chosen:
    q=vectors[:,idx].reshape(d,d);q=(q+q.T)/2
    eigen,frame=torch.linalg.eigh(q);ids=eigen.abs().argsort()[-2:]
    bank.append(frame[:,ids]);weights.append(eigen[ids]/eigen[ids].norm())
allb=torch.cat([b[retained],torch.stack(bank)]);alln=torch.cat([n[retained],torch.stack(weights)])
allf=tensors(allb,alln);allk=gram(allb,alln);allc=allf.flatten(1)@target.flatten(1).T
gains,_,_=scores(allk,allc,torch.arange(len(retained)));pick=int(gains.argmax())
newb,newn=b.clone(),n.clone();newb[weak]=allb[pick];newn[weak]=alln[pick]
before,_=evaluate(newb,newn,energy)
newb,newn,newmix,h,reason=fit(newb,newn,evaluate,energy,max_steps=1000,max_seconds=30)
error=max(0.,h[-1]['objective'])**.5
r=dict(pred_a=eigenerror<=1e-10 and before<=initial+1e-12,pred_b=error<=1e-4,pred_c=len(h)<=100,
       weak_node=weak,selected_candidate=pick-len(retained),initial_error=initial**.5,after_candidate_error=max(0.,before)**.5,final_error=error,
       eigen_residual=eigenerror,iterations=len(h),gradient=h[-1]['projected_gradient_norm'],termination=reason,
       candidate_gains=[float(v)/energy for v in gains[len(retained):]],
       scope='One failed planted case; candidate derived from residual tensor, not true factors. Spectral relaxation plus rank truncation is not a global atom solve.')
out=P/'QUARTIC_RESIDUAL_EIGENMATRIX_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['pred_a']
