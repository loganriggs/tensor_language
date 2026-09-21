#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_learning pred_c_comparison
"""Exact CP512: two Muon random starts,400steps; price1536products2385920coeff.
Integrity normalresidual<1e-8/export<1e-4. Both explainedgain>=2/queryimprove.
Some start Gaussian<.96355 and same coefficientqueries<.98475; zero100% null.
No fullnormcertificate, native circuit adoption or equalprice superiority claim.
"""
import os,sys,time,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from quartic_cp import directional,cp_entries,cp_gram
 from quartic_cp_profile import native_objective,normalize_factors
 from check_quartic_cp_native_profile import controls
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 def values(factors,coeff,x):
  z=x@factors[0].T
  for f in factors[1:]:z=z*(x@f.T)
  return z@coeff.T
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checked=controls();torch.manual_seed(1001);dtype=torch.float64
  teacher=[torch.randn(16,7,dtype=dtype),torch.randn(7,12,dtype=dtype),torch.randn(7,12,dtype=dtype),torch.randn(12,7,dtype=dtype),torch.randn(7,12,dtype=dtype),torch.randn(7,12,dtype=dtype)]
  p=[torch.randn(512,12,dtype=dtype,requires_grad=True) for _ in range(4)];f=normalize_factors(p);loss,c=native_objective(teacher,f,ridge=1e-6);loss.backward();x=torch.randn(5,12,dtype=dtype)
  assert c.shape==(16,512) and values(f,c,x).shape==(5,16) and all(torch.isfinite(a.grad).all() for a in p)
  assert cp_entries(c,f,torch.randint(12,(9,4))).shape==(9,16)
  print(json.dumps(dict(controls=checked,rank512_shape_smoke=True,steps=400,seeds=[1001,1002],products=1536,coefficients=2385920)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_CP512_NATIVE_V2.json';assert not out.exists();scale=19054614563.464127
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().double()
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 with torch.no_grad():
  indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64)
  query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  gaussian=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[x,x,x,x]) for x in gaussian.split(128)])
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();target=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda()/scale
 def metrics(f,c):
  with torch.no_grad():return dict(sampled_coefficient_error=float((cp_entries(c,f,indices)-query).norm()/query.norm()),gaussian_error=float((values(f,c,gaussian)-truth).norm()/truth.norm()),text_value_error=float((values(f,c,text)-target).norm()/target.norm()))
 rows=[];normals=[];drifts=[]
 for seed in [1001,1002]:
  gen=torch.Generator().manual_seed(seed);params=[torch.nn.Parameter((torch.randn(512,1152,generator=gen,dtype=torch.float64)/math.sqrt(1152)).cuda()) for _ in range(4)]
  rate=.1*math.sqrt(4/1152);opt=torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=None;history=[]
  for step in range(401):
   f=normalize_factors(params);loss,c=native_objective(teacher,f,ridge=1e-6);v=float(loss.detach());assert math.isfinite(v)
   if step==0:normalizer=abs(v);assert normalizer>0;initial=metrics(f,c);initial_score=v
   if best is None or v<best[0]:best=(v,step,[a.detach().clone() for a in f],c.detach().clone())
   if step%50==0:
    row=dict(seed=seed,step=step,objective=v,elapsed=time.monotonic()-start);history.append(row);print(json.dumps(row),flush=True)
   if step==400:break
   opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(a.grad).all() for a in params);opt.step()
   for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/400)))
  with torch.no_grad():
   f,c=best[2],best[3];g=cp_gram(f,f);cross=directional(*teacher,f).T
   normal=float((c@(g+1e-6*torch.eye(512,device='cuda',dtype=torch.float64))-cross).norm()/cross.norm());normals.append(normal)
   final=metrics(f,c);ref=values(f,c,text[:128]);fp=values([a.float() for a in f],(c*scale).float(),text[:128].float()).double()/scale;drift=float((fp-ref).norm()/ref.norm());drifts.append(drift)
   artifact=dict(factors=[a.cpu().float() for a in f],coefficients=(c*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4)
   path=P/f'QUARTIC_CP512_SEED{seed}_V2.pt';torch.save(artifact,path)
   rows.append(dict(seed=seed,initial=initial,final=final,initial_objective=initial_score,selected_objective=best[0],explained_gain=best[0]/initial_score,selected_step=best[1],history=history,normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  del opt,params,best,f,c,g,cross,loss
 pred=dict(pred_a_integrity=max(normals)<1e-8 and max(drifts)<1e-4,pred_b_learning=all(r['explained_gain']>=2 and r['final']['sampled_coefficient_error']<r['initial']['sampled_coefficient_error'] for r in rows),pred_c_comparison=any(r['final']['gaussian_error']<.96355 and r['final']['sampled_coefficient_error']<.98475 for r in rows))
 result=dict(predictions=pred,rows=rows,products=1536,coefficients=2385920,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact coefficient objective,16 fixed data-informed readers,random CP directions; evaluation-only sampled coefficient/Gaussian/text metrics; no convergence or semantic/circuit claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
