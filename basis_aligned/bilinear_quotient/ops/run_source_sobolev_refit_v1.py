#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_sensitivity pred_c_values
"""Source-gradient metric sweep on the existing shared graph, native ports fixed.
pred_a_instrument implicit/dense objective agreement<1e-8 and finite results.
pred_b_sensitivity primarylambda1 reduces BOTH component1/2 Euclidean
conditional Jacobian errors>=10% versus frozen parent.
pred_c_values allthree scalarerrors<=15% and<=1.10same separatebaseline;
512sourceproducts and897804floating coefficients, unchangedprivate3.
Null: response objective trades away valuefidelity or cannotrepair directions.
8fits: lambda0/.1/1/10 x rate.005/.02;2000cosine steps fromfrozenparent.
Perlambda winnerselectedbyitsweightobjective only. No model forwards.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'SOURCE_SOBOLEV_REFIT_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from source_sobolev import SourceSobolev
 from shared_quadratic_products import materialize_mixed
 from source_graph_metrics import export,score
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();output=P/'SOURCE_SOBOLEV_REFIT_V1.json';assert not output.exists()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[plan['parent']]
 root=torch.linalg.inv(data['inverse_root']);s=parent['shared_mixed'];initL=root@s['left_reader'];initR=root@s['right_reader']
 T=data['teacher'][:4].cuda();H=(data['inverse_root']@data['inverse_root']).cuda();records=[];programs={}
 for lam in plan['lambdas']:
  metric=SourceSobolev(T,H,lam)
  for rate in plan['rates']:
   L=initL.cuda().clone();R=initR.cuda().clone();L/=L.norm(dim=0);R/=R.norm(dim=0);L.requires_grad_();R.requires_grad_();opt=torch.optim.Adam([L,R],lr=rate);best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,W=metric.loss(L,R);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;state=(L.detach().clone(),R.detach().clone(),W.detach().clone());beststep=step
    if step%400==0:history.append(dict(step=step,objective=value));print(lam,rate,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=rate*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
    with torch.no_grad():L.div_(L.norm(dim=0));R.div_(R.norm(dim=0))
   with torch.no_grad():
    l,r,w=state;hat=materialize_mixed(l,r,w);replay=abs(float(metric.explicit(hat,w))-best)
    E=hat-T;coef=float(E.norm()/T.norm());grad=float(((E*(H@E)).sum()/(T*(H@T)).sum()).sqrt())
    program=export(l.cpu(),r.cpu(),w.cpu(),data,parent);scores=score(program,data)
    floats=program['residual_writer'].numel()+sum(v.numel() for sub in ['shared_mixed','private_pair'] for v in program[sub].values() if v.is_floating_point())
    assert floats==897804
    key=f'{lam}_{rate}';programs[key]=program;records.append(dict(key=key,lam=lam,rate=rate,objective=best,best_step=beststep,dense_replay=replay,coefficient_error=coef,source_gradient_error=grad,stored_float_scalars=floats,source_products=512,history=history,**scores))
 winners={str(lam):min((r for r in records if r['lam']==lam),key=lambda r:r['objective'])['key'] for lam in plan['lambdas']}
 selected=next(r for r in records if r['key']==winners['1']);baseline=plan['scalar_baseline'];parenterrors=plan['parent_jacobian']
 out=dict(plan=plan,records=records,winners=winners,predictions=dict(pred_a_instrument=max(r['dense_replay'] for r in records)<1e-8,pred_b_sensitivity=all(a<=.9*b for a,b in zip(selected['euclidean_jacobian_errors'],parenterrors)),pred_c_values=all(a<=.15 and a<=1.1*b for a,b in zip(selected['per_mode_errors'],baseline))),seconds=time.perf_counter()-start)
 torch.save(programs,P/'SOURCE_SOBOLEV_REFIT_PROGRAMS_V1.pt');output.write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'],flush=True)
if __name__=='__main__':main()
