#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;16384exact coefficient contractions,128cached validationvectors.
"""pred_a normalresidual<=1e-8 and exact coefficient objective nonincrease; pred_b nativeerror halves; pred_c nativeerror<=.1.
Fixed32quadratics shared across2outputs, exactjointwriter solve;592704floats perseed. No textfit.
"""
import os,sys,json,hashlib,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from coupled_quartic_writer_v1 import gram,target_cross,solve,features
from composed_quartic_contraction_v1 import contract
STEM='COUPLED_QUARTIC_WRITER_V1'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/(STEM+'_CONTROL.json')).read_text())['pred_a']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,exact_coefficient_contractions=16384,maximum_batch=256,fitted_floats=592704)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(180)
 torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 native=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')];scale=float(state['transformer.h.17.lambdas'][0])
 programs=torch.load(P/'FULLSOURCE_QUARTIC_SPECTRAL_V1_PROGRAMS.pt',weights_only=True)['programs']
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 den=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
 refs=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'];reports=[];saved=[]
 for idx,p in enumerate(programs):
  b=p['input_readers'].reshape(32,1152,16).cuda();nu=p['inner_weights'].reshape(32,16).cuda();writer=p['output_writers'].cuda()
  w=native[:-1]+[(metric@writer).T@native[-1]]
  k=gram(b,nu);c=target_cross(b,nu,lambda slots:contract(slots,*w,scale));mix,diag=solve(k,c)
  old=torch.zeros(32,2,device='cuda');old[:16,0]=p['outer_weights'][0].cuda();old[16:,1]=p['outer_weights'][1].cuda()
  def objective(z):return float((z*(k@z)).sum()-2*(z*c).sum())
  oldobj,newobj=objective(old),objective(mix);delta=newobj-oldobj
  feat=features(b,nu,x);reference=refs[idx].cuda()
  olderror=rel(feat@old@writer.T/den[:,None],reference);newerror=rel(feat@mix@writer.T/den[:,None],reference)
  reports.append(dict(seed=p['seed'],diagnostics=diag,baseline_native_error=olderror,coupled_native_error=newerror,objective_without_target_constant_before=oldobj,objective_without_target_constant_after=newobj,objective_delta=delta,relative_objective_delta=delta/max(abs(oldobj),1e-30)))
  saved.append(dict(seed=p['seed'],input_readers=b.cpu(),inner_weights=nu.cpu(),mixing=mix.cpu(),output_writers=writer.cpu()))
  print(json.dumps(reports[-1]),flush=True)
 torch.save(dict(programs=saved),ap)
 result={'pred_a':all(r['diagnostics']['normal_residual']<=1e-8 and r['relative_objective_delta']<=1e-10 for r in reports),'pred_b':all(r['coupled_native_error']<=.5*r['baseline_native_error'] for r in reports),'pred_c':all(r['coupled_native_error']<=.1 for r in reports),'reports':reports,'artifact_sha256':digest(ap),'wall_seconds':time.perf_counter()-tic,'scope':'Exact global linear writer optimum in fixed32feature span, not nonlinear optimum or OOD validation.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'}),flush=True);assert result['pred_a']
if __name__=='__main__':main()
