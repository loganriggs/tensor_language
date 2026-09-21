#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_gaussian pred_c_preservation
"""FrozenCP512 Gaussianreadout: no probes intraining;2archivedfactor sets.
TraceQ=2mean<1e-9,normal<1e-8,export<1e-4. Both exactlossnonincrease/Gauss<=.9old.
Bothquery/text<=1.1old. Null fixedspan/metrictradeoff;price1536/2385920 unchanged.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from native_quartic_gaussian_projection import project,cp_cross,cp_mean
 from check_native_quartic_gaussian_projection import controls
 from gaussian_cp import gaussian_cp_gram
 from quartic_cp import cp_gram,cp_entries,directional
 from quartic_cp_profile import profile,normalize_factors
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  validation=controls();dtype=torch.float64;torch.manual_seed(10700)
  teacher=[torch.randn(*s,dtype=dtype) for s in [(16,7),(7,12),(7,12),(12,7),(7,12),(7,12)]]
  m,Q=project(teacher);f=normalize_factors([torch.randn(512,12,dtype=dtype) for _ in range(4)]);G=gaussian_cp_gram(f,f);X=cp_cross(teacher,m,Q,f);loss,c=profile(G,X,ridge=1e-6)
  assert c.shape==(16,512) and torch.isfinite(loss)
  print(json.dumps(dict(controls=validation,shape512=True,no_training_probes=True)));return
 out=P/'EXACT_GAUSSIAN_CP_READOUT_V1.json';assert not out.exists();start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False;scale=19054614563.464127
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().double()
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 with torch.no_grad():
  m,Q=project(teacher);trace_error=float((Q.diagonal(dim1=-2,dim2=-1).sum(-1)-2*m).norm()/(2*m).norm());print(json.dumps(dict(projection_seconds=time.monotonic()-start,trace_error=trace_error)),flush=True)
  torch.save(dict(mean=m.cpu(),quadratic=Q.cpu(),scale=scale,writer=writer.cpu()),P/'GAUSSIAN_CP_TEACHER_PROJECTION_V1.pt')
  indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  x=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[z,z,z,z]) for z in x.split(128)])
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();target=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda()/scale
  def values(f,c,x):
   z=x@f[0].T
   for a in f[1:]:z=z*(x@a.T)
   return z@c.T
  rows=[]
  for seed in [1001,1002]:
   source=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt',weights_only=True);f=[a.cuda().double() for a in source['factors']];old=source['coefficients'].cuda().double()/scale
   G=gaussian_cp_gram(f,f);X=cp_cross(teacher,m,Q,f);_,new=profile(G,X,ridge=1e-6);G4=cp_gram(f,f);X4=directional(*teacher,f).T;mu=cp_mean(f);G2=G-24*G4-mu[:,None]*mu[None,:];X2=X-24*X4-m[:,None]*mu[None,:]
   normal=float((new@(G+1e-6*torch.eye(512,device='cuda',dtype=G.dtype))-X).norm()/X.norm());results={}
   for name,c in [('archived',old),('gaussian',new)]:
    pmean=c@mu;candidate2=float((c*(c@G2)).sum());cross2=float((c*X2).sum());e2=float(2*Q.square().sum())+candidate2-2*cross2
    results[name]=dict(gaussian_error=float((values(f,c,x)-truth).norm()/truth.norm()),sampled_coefficient_error=float((cp_entries(c,f,indices)-query).norm()/query.norm()),text_value_error=float((values(f,c,text)-target).norm()/target.norm()),regularized_objective=float((c*(c@G)).sum()-2*(c*X).sum()+1e-6*c.square().sum()),degree0_residual_energy=float((pmean-m).square().sum()),degree2_residual_energy=e2,degree4_explained_energy=float(24*(2*(c*X4).sum()-(c*(c@G4)).sum())))
   ref=values(f,new,text[:128]);fp=values([a.float() for a in f],(new*scale).float(),text[:128].float()).double()/scale;drift=float((fp-ref).norm()/ref.norm())
   path=P/f'GAUSSIAN_CP512_SEED{seed}_V1.pt';torch.save(dict(factors=[a.cpu().float() for a in f],coefficients=(new*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4),path)
   row=dict(seed=seed,results=results,normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
 pred=dict(pred_a_integrity=trace_error<1e-9 and all(r['normal_residual']<1e-8 and r['export_error']<1e-4 for r in rows),pred_b_gaussian=all(r['results']['gaussian']['regularized_objective']<=r['results']['archived']['regularized_objective']+1e-10*abs(r['results']['archived']['regularized_objective']) and r['results']['gaussian']['gaussian_error']<=.9*r['results']['archived']['gaussian_error'] for r in rows),pred_c_preservation=all(r['results']['gaussian'][k]<=1.1*r['results']['archived'][k] for r in rows for k in ['sampled_coefficient_error','text_value_error']))
 result=dict(predictions=pred,rows=rows,trace_error=trace_error,teacher_degree0_energy=float(m.square().sum()),teacher_degree2_energy=float(2*Q.square().sum()),seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact Gaussian fixed-direction readout and Hermite components; no sampled fitting, no exact globalrelativeerrorcertificate, no native circuitadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
