#!/usr/bin/env python3
# BQGATE: 0 body forwards; four warm-start matrix-free trust-Newton fits; 60sec/64steps each; 420sec wall.
"""pred_a matched coordinates and nonworsening objective; pred_b >=10% capture
improvement and unit-reader gradient<=1e-6 for a secant fit; pred_c held-position
capture nonworsening for best training-objective secant candidate.
A bounded optimizer pilot, not absence-of-structure evidence if unfinished.
"""
import os,sys,time,json,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import numpy as np
import torch
from scipy.optimize import minimize
from cubic_projection_curvature_v1 import Curvature
from cubic_secant_coordinates_v1 import encode,components,normalized_reader_gradient
from cubic_secant_block_v1 import capture as secant_capture
from shared_cubic_source_projection_v1 import capture as raw_capture
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
STEM='FOLDED_CUBIC_TRUST_PILOT_V1'
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
class FitTimeout(Exception):pass

def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/'CUBIC_PROJECTION_CURVATURE_V1_CONTROL.json').read_text())['pred_a']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('Four matched raw/secant warm fits, matrix-free exact Hessian; no model forwards');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(420);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_default_dtype(torch.float64)
 old=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in old if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 readers=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,readers,'cuda')
 weights={}
 for pos in [7,0]:
  r=rotary(8,128).cuda().T@rotary(pos,128).cuda();weights[pos]=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
 saved=torch.load(P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'];scale=max(float(raw_capture(a.cuda(),*weights[7])) for a in saved);rows=[];programs=[];tic=time.perf_counter()
 for arm,pair in [(0,(5,15)),(1,(4,5))]:
  a=saved[arm].cuda();sec,sep=encode(a,pair);initial_reference=float(raw_capture(a,*weights[7]));initial_held=float(raw_capture(a,*weights[0]))
  for mode,start in [('raw',a),('secant',sec)]:
   def cap(z,pos=7):
    if mode=='raw':return raw_capture(z,*weights[pos])
    c,m=components(z,sep);return secant_capture(c,m,weights[pos])
   def objective(z):return -cap(z)/scale
   adapter=Curvature(objective,start.shape,'cuda');x0=start.cpu().numpy().ravel();initial,initial_g=adapter.fun(x0);best={'value':initial,'x':x0.copy()};started=time.perf_counter();calls=[]
   def fun(x):
    if time.perf_counter()-started>60:raise FitTimeout()
    value,g=adapter.fun(x)
    if np.isfinite(value) and value<best['value']:best.update(value=value,x=x.copy())
    return value,g
   def hessp(x,d):
    if time.perf_counter()-started>60:raise FitTimeout()
    return adapter.hessp(x,d)
   try:
    fit=minimize(fun,x0,jac=True,hessp=hessp,method='trust-krylov',options={'maxiter':64,'gtol':1e-7,'initial_trust_radius':.1,'max_trust_radius':10.})
    reason=str(fit.message);iterations=int(fit.nit)
   except FitTimeout:reason='time_limit';iterations=None
   except (torch.linalg.LinAlgError,ValueError,RuntimeError) as e:reason='numerical_failure: '+str(e);iterations=None
   best_tensor=torch.from_numpy(best['x']).reshape(start.shape).cuda();val,g=adapter.fun(best['x']);gg=torch.from_numpy(g).reshape(start.shape).cuda();intrinsic=normalized_reader_gradient(best_tensor,gg,sep if mode=='secant' else None)
   row=dict(arm=arm,coordinates=mode,initial_capture=initial_reference,coordinate_initial_error=abs((-initial*scale)/initial_reference-1),final_capture=-val*scale,gain_ratio=(-val*scale)/initial_reference,initial_held_capture=initial_held,held_capture=float(cap(best_tensor,0)),unit_reader_gradient=float(intrinsic.norm()),stationary=float(intrinsic.norm())<=1e-6,termination=reason,iterations=iterations,evaluations=adapter.evaluations,hessian_products=adapter.hessian_products,seconds=time.perf_counter()-started)
   rows.append(row);programs.append(dict(arm=arm,coordinates=mode,theta=best_tensor.cpu(),separation=sep));print(json.dumps(row),flush=True)
   (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(rows,indent=2)+'\n')
 secrows=[r for r in rows if r['coordinates']=='secant'];bestrow=max(secrows,key=lambda r:r['final_capture'])
 artifact=P/(STEM+'_ARTIFACT.pt');torch.save(dict(programs=programs,scale=scale),artifact)
 result=dict(pred_a=all(r['coordinate_initial_error']<=1e-6 and r['gain_ratio']>=1-1e-7 for r in rows),pred_b=any(r['gain_ratio']>=1.1 and r['stationary'] for r in secrows),pred_c=bestrow['held_capture']>=bestrow['initial_held_capture'],rows=rows,selected_secant_arm=bestrow['arm'],execution_seconds=time.perf_counter()-tic,peak_allocated_bytes=torch.cuda.max_memory_allocated(),artifact_sha=digest(artifact),scope='Four warm-start exact-Hessian trust-region fits of original weights-only producer numerator objective. Best finite evaluated candidate per arm, not necessarily accepted trust iterate. Independent cold starts and behavioral validation not performed by this pilot. Fixed caps do not certify convergence; native gates and generators remain external.')
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
