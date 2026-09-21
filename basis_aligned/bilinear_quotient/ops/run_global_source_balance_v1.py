#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_values pred_c_groups
"""All six source reads in one shared mixed-product graph.
Primary power.5; controls0/1, two random seeds816/1816 each, Adam .02,
2000cosine steps. Analytic readout. Winners by fitted weight objective only.
pred_a_instrument: dense/implicit loss<1e-8;383products896262floats.
pred_b_values: all3opened scalarerrors<=.15 and<=1.10separate baseline.
pred_c_groups: all6original teacher-anchored core cosines>=.99.
Null: reweighting/sharing sacrifices original fidelity or weak groups remain.
No native model forwards; nativez,h explicit; no target truncated.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'GLOBAL_SOURCE_BALANCE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P))
 from source_sobolev import SourceSobolev
 from shared_quadratic_products import materialize_mixed
 from global_mixed_source_graph import export,score
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 output=P/'GLOBAL_SOURCE_BALANCE_V1.json';assert not output.exists()
 start=time.perf_counter();d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 original=d['teacher'].cuda();flat=original.flatten(1);e,U=torch.linalg.eigh(flat@flat.T);e=e.flip(0);U=U.flip(1)
 assert float(e.min()/e.max())>1e-12
 original_core=torch.einsum('oa,oij->aij',U,original)
 records=[];programs={};H=torch.eye(original.shape[-1],dtype=original.dtype,device='cuda')
 for power in plan['powers']:
  outmap=(U*e.pow(-power/2))@U.T;inverse=(U*e.pow(power/2))@U.T
  T=torch.einsum('ab,bij->aij',outmap,original);metric=SourceSobolev(T,H,0)
  for seed in plan['seeds']:
   g=torch.Generator().manual_seed(seed)
   L=torch.randn((T.shape[-1],plan['products']),dtype=T.dtype,generator=g).cuda();R=torch.randn(L.shape,dtype=T.dtype,generator=g).cuda()
   L/=L.norm(dim=0);R/=R.norm(dim=0);L.requires_grad_();R.requires_grad_()
   opt=torch.optim.Adam([L,R],lr=plan['rate']);best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,W=metric.loss(L,R);v=float(loss.detach());assert math.isfinite(v)
    if v<best:best=v;state=(L.detach().clone(),R.detach().clone(),W.detach().clone());beststep=step
    if step%400==0:history.append(dict(step=step,objective=v));print(power,seed,step,v,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
    with torch.no_grad():L.div_(L.norm(dim=0));R.div_(R.norm(dim=0))
   with torch.no_grad():
    l,r,w=state;hat=materialize_mixed(l,r,w);replay=abs(float(metric.explicit(hat,w))-best)
    native_w=inverse@w;native_hat=materialize_mixed(l,r,native_w);cores=torch.einsum('oa,oij->aij',U,native_hat)
    cos=((cores*original_core).sum((-1,-2))/(cores.norm(dim=(-1,-2))*original_core.norm(dim=(-1,-2)))).tolist()
    core_errors=((cores-original_core).norm(dim=(-1,-2))/original_core.norm(dim=(-1,-2))).tolist()
    p=export(l.cpu(),r.cpu(),native_w.cpu(),d);price=sum(v.numel() for v in p.values());assert price==plan['stored_floats']
    key=f'{power}_seed{seed}';programs[key]=p
    records.append(dict(key=key,power=power,seed=seed,objective=best,best_step=beststep,dense_replay=replay,coefficient_error=float((hat-T).norm()/T.norm()),original_coefficient_error=float((native_hat-original).norm()/original.norm()),original_core_cosines=cos,original_core_relative_errors=core_errors,source_products=plan['products'],stored_floats=price,history=history,**score(p,d)))
    print('RECORD',json.dumps({k:v for k,v in records[-1].items() if k!='history'}),flush=True)
 winners={str(power):min((r for r in records if r['power']==power),key=lambda r:r['objective'])['key'] for power in plan['powers']}
 selected=next(r for r in records if r['key']==winners[str(plan['primary_power'])])
 out=dict(plan=plan,records=records,winners=winners,predictions=dict(pred_a_instrument=max(r['dense_replay'] for r in records)<1e-8,pred_b_values=all(a<=.15 and a<=1.1*b for a,b in zip(selected['per_mode_errors'],plan['scalar_baseline'])),pred_c_groups=min(selected['original_core_cosines'])>=.99),seconds=time.perf_counter()-start)
 torch.save(programs,P/'GLOBAL_SOURCE_BALANCE_PROGRAMS_V1.pt');output.write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'],flush=True)
if __name__=='__main__':main()
