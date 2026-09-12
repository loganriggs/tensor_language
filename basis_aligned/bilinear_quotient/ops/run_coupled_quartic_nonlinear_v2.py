#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;180exact coefficient optimization steps,128cached validationvectors.
"""pred_a nativegradientFD<=1e-4 and monotonicobjective; pred_b capturegain>=10%; pred_c projectedgrad<=1e-6.
One seed11511,32rank16quadratics,592704floats. Exactcoefficientvariableprojection, no textfit.
Null: pilotmaynotconverge or improve despite correctgradient; continuationrequired, notabsentstructure.
"""
import os,sys,json,hashlib,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from coupled_quartic_writer_v1 import gram,target_cross,features
from quartic_manifold_cg_v1 import fit
from composed_quartic_contraction_v1 import contract
STEM='COUPLED_QUARTIC_NONLINEAR_V2'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()
def tangent(b,n,gb,gn):
 z=b.transpose(-1,-2)@gb
 return gb-b@((z+z.transpose(-1,-2))/2),gn-(gn*n).sum(-1,keepdim=True)*n
def retract(b,n,db,dn,step):
 q,r=torch.linalg.qr(b+step*db,mode='reduced');sign=r.diagonal(dim1=-2,dim2=-1).sign();q=q*sign[:,None,:]
 newn=n+step*dn;return q,newn/newn.norm(dim=-1,keepdim=True)
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,optimization_steps=180,maximum_line_search=20,fitted_floats=592704)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(1200)
 torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 native=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')];scale=float(state['transformer.h.17.lambdas'][0])
 p=torch.load(P/'COUPLED_QUARTIC_NONLINEAR_V1_PROGRAM.pt',weights_only=True);assert p['seed']==11511
 b=p['input_readers'].cuda();n=p['inner_weights'].cuda();n=n/n.norm(dim=-1,keepdim=True);writer=p['output_writers'].cuda();native[-1]=(metric@writer).T@native[-1]
 def evaluate(b,n,divisor,gradient=False):
  if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
  with torch.set_grad_enabled(gradient):
   k=gram(b,n);c=target_cross(b,n,lambda slots:contract(slots,*native,scale))
   with torch.no_grad():mix=torch.linalg.solve(k,c)
   value=((mix*(k@mix)).sum()-2*(mix*c).sum())/divisor
   if gradient:
    gb,gn=torch.autograd.grad(value,(b,n));gb,gn=tangent(b,n,gb,gn)
    return float(value),mix.detach(),gb.detach(),gn.detach()
   return float(value),mix
 initial,_=evaluate(b,n,1.);divisor=p['divisor'];assert divisor>0
 previous=json.loads((P/'COUPLED_QUARTIC_NONLINEAR_V1_RESULT.json').read_text());assert abs(initial/divisor-previous['history'][-1]['objective'])<=1e-8
 y,mix,gb,gn=evaluate(b,n,divisor,True)
 torch.manual_seed(91603);db,dn=tangent(b,n,torch.randn_like(b),torch.randn_like(n));norm=(db.square().sum()+dn.square().sum()).sqrt();db/=norm;dn/=norm
 eps=1e-5
 bp,np=retract(b,n,db,dn,eps);bm,nm=retract(b,n,db,dn,-eps)
 yp,_=evaluate(bp,np,divisor);ym,_=evaluate(bm,nm,divisor);numerical=(yp-ym)/(2*eps);analytic=float((gb*db).sum()+(gn*dn).sum());fd=abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-10);assert fd<=1e-4
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda();den=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
 reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].cuda()
 initial_native=rel(features(b,n,x)@mix@writer.T/den[:,None],reference);history=[];line_failed=False
 def callback(row,bank,weights,currentmix):
  print(json.dumps(row),flush=True)
  (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(row,indent=2)+'\n')
 b,n,mix,history,reason=fit(b,n,evaluate,divisor,max_steps=180,max_seconds=1000,callback=callback)
 y=history[-1]['objective'];gradnorm=history[-1]['projected_gradient_norm'];line_failed=reason=='line_search_failed'
 native_error=rel(features(b,n,x)@mix@writer.T/den[:,None],reference)
 torch.save(dict(seed=11511,input_readers=b.cpu(),inner_weights=n.cpu(),mixing=mix.cpu(),output_writers=writer.cpu(),divisor=divisor),ap)
 result={'pred_a':fd<=1e-4 and all(z['objective']<=history[i]['objective']+1e-10 for i,z in enumerate(history[1:])),'pred_b':-y>=1.1,'pred_c':gradnorm<=1e-6,'directional_gradient_relative_error':fd,'history':history,'termination':reason,'capture_gain_fraction':-y-1,'line_search_failed':line_failed,'initial_native_error':initial_native,'final_native_error':native_error,'artifact_sha256':digest(ap),'wall_seconds':time.perf_counter()-tic,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'scope':'One-seed exact nonlinear continuation; localstationarity separate from gain, no OOD or selectivity evidence.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert result['pred_a']
if __name__=='__main__':main()
