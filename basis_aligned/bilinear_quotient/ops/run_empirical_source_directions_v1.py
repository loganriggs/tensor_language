#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_values pred_c_coefficient
"""Coefficient-only versus empiricalsource lambda1, same399-product topology.
Two1%perturbationseeds816/1816, Adam.005,1000cosine steps each, full16384
calibrationstates perstep, exact constrainedreadout solves. Primarylambda1.
pred_a_instrument dense/implicit<1e-8,399products896198floats.
pred_b_values all3openederrors<=15% and<=1.10separatebaseline.
pred_c_coefficient primaryoriginalerror<=1.10matched-extra-budgetlambda0.
Null: richer metric overfits or current feature family cannot meet fidelity.
All6source outputs and nativez,h ports retained; no model forwards.
"""
import os,sys,json,math,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from empirical_source_varpro import EmpiricalSourceVarpro
 from global_mixed_source_graph import export,score,source_reads
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 for n,digest in plan['hashes'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==digest
 output=P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json';assert not output.exists();start=time.perf_counter()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);parent=torch.load(P/'COMPACT_GROUP_FROZEN_V1.pt',weights_only=True);expanded=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
 root=torch.linalg.inv(d['inverse_root']);T=d['teacher'].cuda();F=T.flatten(1);e,U=torch.linalg.eigh(F@F.T);records=[];programs={}
 x=(torch.cat([d['z'][:1536],expanded['z'].flatten(0,1).double()])-d['mu'])@d['inverse_root'];x=x.cuda();K=(d['inverse_root']@d['old_covariance']@d['inverse_root']).cuda();Y=torch.einsum('ni,oij,nj->no',x,T,x)-torch.einsum('ij,oji->o',K,T)
 assert len(x)==16384
 truthcores=U.T@F
 for lam in plan['lambdas']:
  k=16;p=parent;init=[root@p[n] for n in ['left_reader','right_reader','square_reader']];metric=EmpiricalSourceVarpro(T,x,K,Y,lam)
  for seed in plan['seeds']:
   g=torch.Generator().manual_seed(seed);params=[]
   for a in init:
    a=a/a.norm(dim=0);noise=torch.randn(a.shape,dtype=a.dtype,generator=g);noise/=noise.norm(dim=0);a=(a+plan['perturbation']*noise).cuda();a/=a.norm(dim=0);a.requires_grad_();params.append(a)
   L,R,V=params;opt=torch.optim.Adam(params,lr=plan['rate']);best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,W,v=metric.loss(L,R,V);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;state=tuple(a.detach().clone() for a in [L,R,V,W,v]);beststep=step
    if step%400==0:history.append(dict(step=step,objective=value));print(lam,seed,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
    with torch.no_grad():
     for a in params:a.div_(a.norm(dim=0))
   with torch.no_grad():
    l,r,s,w,v=state;raw=torch.einsum('ir,or,jr->oij',l,w,r);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(s*v)@s.T;replay=abs(float(metric.explicit(l,r,s,w,v))-best);cores=U.T@hat.flatten(1);cos=((cores*truthcores).sum(1)/(cores.norm(dim=1)*truthcores.norm(dim=1))).tolist()
    extra=torch.zeros((6,len(v)),dtype=T.dtype,device='cuda');extra[5]=v
    big=export(torch.cat([l,s],1).cpu(),torch.cat([r,s],1).cpu(),torch.cat([w,extra],1).cpu(),d);m=l.shape[1];p={name:a.clone() for name,a in big.items()}
    for name in ['left_reader','right_reader']:p[name]=big[name][:,:m].clone()
    p['product_weights']=big['product_weights'][:m].clone();p['square_reader']=big['left_reader'][:,m:].clone();p['square_weights']=big['product_weights'][m:,5].clone();p['square_output']=torch.tensor(5)
    z=d['z'][:32];a=source_reads(z,p);a[:,5]+=(z@p['square_reader']).square()@p['square_weights'];b=source_reads(z,big);execution=float((a-b).norm()/b.norm());floats=sum(a.numel() for a in p.values() if a.is_floating_point());assert floats==896262-4*k
    key=f'{lam}_seed{seed}';programs[key]=p;records.append(dict(key=key,lam=lam,seed=seed,objective=best,best_step=beststep,dense_replay=replay,execution_replay=execution,original_coefficient_error=float((hat-T).norm()/T.norm()),original_core_cosines=cos,source_products=383+k,stored_floats=floats,history=history,**score(big,d)));print('RECORD',json.dumps({key:val for key,val in records[-1].items() if key!='history'}),flush=True)
 winners={str(lam):min((r for r in records if r['lam']==lam),key=lambda r:r['objective'])['key'] for lam in plan['lambdas']};chosen={k:next(r for r in records if r['key']==v) for k,v in winners.items()};selected=chosen['1'];control=chosen['0']
 out=dict(plan=plan,records=records,winners=winners,predictions=dict(pred_a_instrument=max(max(r['dense_replay'],r['execution_replay']) for r in records)<1e-8,pred_b_values=all(x<=.15 and x<=1.1*y for x,y in zip(selected['per_mode_errors'],plan['scalar_baseline'])),pred_c_coefficient=selected['original_coefficient_error']<=1.10*control['original_coefficient_error']),seconds=time.perf_counter()-start)
 torch.save(programs,P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt');output.write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'],flush=True)
if __name__=='__main__':main()
