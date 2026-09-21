#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_law pred_c_preservation
"""Three calibration Gaussianlaws,2frozenCP400factor sets;1536products2385920coeff.
Normal<1e-8,Cholesky<1e-12,export<1e-4,exactlossnonincrease.
Alllaws fresherror<=.9old; primaryshifted text<=.9old,iso/query<=1.1old.
No sampledtraining; null covariancefit tradeoff, no semanticadoption.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from noncentral_gaussian_cp import project_shifted,cross,gram
 from native_quartic_gaussian_projection import project
 from check_noncentral_gaussian_cp import controls
 from quartic_cp import directional,cp_entries
 from quartic_cp_profile import profile,normalize_factors
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  validation=controls();dtype=torch.float64;torch.manual_seed(10950);t=[torch.randn(*s,dtype=dtype) for s in [(16,7),(7,12),(7,12),(12,7),(7,12),(7,12)]];location=torch.randn(12,dtype=dtype);projection=project_shifted(t,location);f=normalize_factors([torch.randn(512,12,dtype=dtype) for _ in range(4)]);b=[a@location for a in f];G=gram(f,b,f,b);X=cross(t,location,projection,f,b);loss,c=profile(G,X,ridge=1e-6)
  assert c.shape==(16,512) and torch.isfinite(loss);print(json.dumps(dict(controls=validation,shape512=True,laws=3,training_probes=0)));return
 out=P/'GAUSSIAN_CP_DATA_READOUT_V1.json';assert not out.exists();start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False;scale=19054614563.464127
 stats=torch.load(P/'EXPANDED_INPUT_GEOMETRY_V1.pt',weights_only=True);mean=stats['mean'].cuda();state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().double()
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 with torch.no_grad():
  def truth(x):return torch.cat([directional(*teacher,[z,z,z,z]) for z in x.split(128)])
  def values(f,c,x):
   z=x@f[0].T
   for a in f[1:]:z=z*(x@a.T)
   return z@c.T
  indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  isotropic=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();isotruth=truth(isotropic);zlaw=torch.randn(1024,1152,generator=torch.Generator().manual_seed(10951),dtype=torch.float64).cuda()
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();target=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda()/scale
  cache={};rows=[];cholerrors=[]
  for law in ['covariance_zero','covariance_shifted','second_zero']:
   key='second_moment' if law=='second_zero' else 'covariance';mu=mean if law=='covariance_shifted' else torch.zeros_like(mean)
   if key not in cache:
    M=stats[key].cuda();S=torch.linalg.cholesky(M);cholerrors.append(float((S@S.T-M).norm()/M.norm()));t=teacher[:4]+[teacher[4]@S,teacher[5]@S];zero=project(t);cache[key]=dict(whitener=S,zero_projection=zero)
   S=cache[key]['whitener'];t=teacher[:4]+[teacher[4]@S,teacher[5]@S];location=torch.linalg.solve(S,mu);projection=project_shifted(t,location,cache[key]['zero_projection']);lawx=zlaw@S.T+mu;lawtruth=truth(lawx)
   for seed in [1001,1002]:
    source=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt',weights_only=True);f=[a.cuda().double() for a in source['factors']];fw=[a@S for a in f];bias=[a@mu for a in f];old=source['coefficients'].cuda().double()/scale
    G=gram(fw,bias,fw,bias);X=cross(t,location,projection,fw,bias);_,new=profile(G,X,ridge=1e-6);normal=float((new@(G+1e-6*torch.eye(512,device='cuda',dtype=G.dtype))-X).norm()/X.norm());metrics={}
    for name,c in [('archived',old),('refitted',new)]:
     metrics[name]=dict(law_error=float((values(f,c,lawx)-lawtruth).norm()/lawtruth.norm()),isotropic_error=float((values(f,c,isotropic)-isotruth).norm()/isotruth.norm()),sampled_coefficient_error=float((cp_entries(c,f,indices)-query).norm()/query.norm()),text_value_error=float((values(f,c,text)-target).norm()/target.norm()),regularized_objective=float((c*(c@G)).sum()-2*(c*X).sum()+1e-6*c.square().sum()))
    ref=values(f,new,text[:128]);fp=values([a.float() for a in f],(new*scale).float(),text[:128].float()).double()/scale;drift=float((fp-ref).norm()/ref.norm());path=P/f'GAUSSIAN_CP_DATA_{law.upper()}_SEED{seed}_V1.pt';torch.save(dict(factors=[a.cpu().float() for a in f],coefficients=(new*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4,law=law),path)
    row=dict(law=law,seed=seed,results=metrics,normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
  saved={k:dict(whitener=v['whitener'].cpu(),zero_projection=tuple(a.cpu() for a in v['zero_projection'])) for k,v in cache.items()};torch.save(dict(projections=saved,mean=mean.cpu(),scale=scale,writer=writer.cpu(),calibration_states=6144),P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt')
 pred=dict(pred_a_integrity=max(cholerrors)<1e-12 and all(r['normal_residual']<1e-8 and r['export_error']<1e-4 and r['results']['refitted']['regularized_objective']<=r['results']['archived']['regularized_objective']+1e-10*abs(r['results']['archived']['regularized_objective']) for r in rows),pred_b_law=all(r['results']['refitted']['law_error']<=.9*r['results']['archived']['law_error'] for r in rows),pred_c_preservation=all(r['results']['refitted']['text_value_error']<=.9*r['results']['archived']['text_value_error'] and all(r['results']['refitted'][k]<=1.1*r['results']['archived'][k] for k in ['isotropic_error','sampled_coefficient_error']) for r in rows if r['law']=='covariance_shifted'))
 result=dict(predictions=pred,rows=rows,maximum_cholesky_error=max(cholerrors),seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact calibration-Gaussian fixed-feature readout fit; Gaussian inputs evaluation only; openedtext notOOD; same homogeneousquartic program, no semantic/circuitadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
