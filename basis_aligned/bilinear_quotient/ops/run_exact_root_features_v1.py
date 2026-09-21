#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_coefficients pred_c_function
"""Exact coefficient self/cross feature fit25Adamsteps,32x4/16fixedwriters.
Normalresidual<1e-8/export<1e-4; explainedenergy>=1.1initial andqueryerror<initial.
Gaussian<=.9initial,textvalue<=1.1inherited,price<=384/330240. No globalnormcertificate.
"""
import os,sys,time,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P));from exact_root_tensor_objective import objective,controls
 from shared_quadratic_bank import normalize_bank,bank_entries
 from quartic_cp import directional
 from empirical_quartic_dictionary import features
 from paired_root_compiler import cast,price
 from audit_root_feature_conditions import root_features
 from audit_root_matched_reader import CK
 from run_sensitive_root_fit_v2 import compile_forms
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  rows=controls();print(json.dumps(dict(exact_coefficient_controls=rows,steps=25,input_width=1152,quadratic_features=32,products_per_feature=4,output_roots=16)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'EXACT_ROOT_FEATURE_NATIVE_V1.json';assert not out.exists();scale=19054614563.464127;base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@base['writer'].cuda();readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 params=[torch.nn.Parameter(base[k].cuda().clone()) for k in ['U','V']];rate=.03*math.sqrt(4/1152);opt=torch.optim.Adam(params,lr=rate);history=[];best=None;normal=[];initial=None
 for step in range(26):
  stepstart=time.monotonic();u,v=normalize_bank(*params);loss,c,g,cross=objective(teacher,u,v);value=float(loss.detach())
  if step==0:normalizer=abs(value);assert normalizer>0;initial=dict(U=u.detach().clone(),V=v.detach().clone(),coefficients=c.detach().clone());initial_value=value
  nr=float((c@(g+1e-6*torch.eye(len(g),device='cuda',dtype=g.dtype))-cross).norm().detach()/cross.norm().detach());normal.append(nr)
  if best is None or value<best[0]:best=(value,step,u.detach().clone(),v.detach().clone(),c.detach().clone())
  if step%5==0 or step==25:
   row=dict(step=step,objective=value,relative_explained=-value/normalizer,normal_residual=nr,elapsed=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated());history.append(row);print(json.dumps(row),flush=True)
  if step==25:break
  opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(p.grad).all() for p in params);opt.step()
  for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/25)))
  del loss,c,g,cross,u,v
 learned=dict(U=best[2],V=best[3],coefficients=best[4]);programs=dict(initial=initial,learned=learned);results={};drifts=[];prices={};hashes={}
 with torch.no_grad():
  generator=torch.Generator().manual_seed(951);indices=torch.randint(1152,(4096,4),generator=generator).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  gaussian=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[x,x,x,x]) for x in gaussian.split(128)]);text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();target=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda()/scale
  for name,p in programs.items():
   queryhat=bank_entries(p['U'],p['V'],indices)@p['coefficients'].T;ghat=features(gaussian,p['U'],p['V'])@p['coefficients'].T;that=features(text,p['U'],p['V'])@p['coefficients'].T;results[name]=dict(sampled_coefficient_error=float((queryhat-query).norm()/query.norm()),gaussian_error=float((ghat-truth).norm()/truth.norm()),text_value_error=float((that-target).norm()/target.norm()))
   source=dict(U=p['U'],V=p['V'],writer=base['writer']);program,diagnostic=compile_forms(source,p['coefficients']*scale);program=cast(program,torch.float32);prices[name]=price(program);path=P/f'EXACT_ROOT_FEATURE_{name.upper()}_V1.pt';torch.save(program,path);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
   def move(x):
    if torch.is_tensor(x):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    return x
   actual=move(program);drifts.append(float((root_features(actual,text[:128].float()).double()/scale-that[:128]).norm()/that[:128].norm()))
  old=move(base);oldtext=root_features(old,text)/scale;oldgauss=root_features(old,gaussian)/scale;results['inherited']=dict(text_value_error=float((oldtext-target).norm()/target.norm()),gaussian_error=float((oldgauss-truth).norm()/truth.norm()))
 a,b=results['learned'],results['initial'];pred=dict(pred_a_integrity=max(normal)<1e-8 and max(drifts)<1e-4 and all(math.isfinite(r['objective']) for r in history),pred_b_coefficients=-best[0]>=1.1*(-initial_value) and a['sampled_coefficient_error']<b['sampled_coefficient_error'],pred_c_function=a['gaussian_error']<=.9*b['gaussian_error'] and a['text_value_error']<=1.1*results['inherited']['text_value_error'] and prices['learned']['products']<=384 and prices['learned']['stored_coefficients']<=330240)
 result=dict(predictions=pred,results=results,history=history,selected_step=best[1],initial_objective=initial_value,selected_objective=best[0],maximum_normal_residual=max(normal),maximum_export_replay=max(drifts),prices=prices,program_sha256=hashes,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exactcoefficientoptimizationwithoutteacherconstant;16fixedreaderspurequartic, sampledcoefficient/Gaussian/textonlyevaluation; no semantic/nativeadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
