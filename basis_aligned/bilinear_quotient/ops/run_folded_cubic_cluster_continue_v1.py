#!/usr/bin/env python3
# BQGATE:0 body forwards;4 matched continuation arms;300sec/512steps each;1500sec wall.
"""pred_a exact starts/nonworsening; pred_b cluster improvement>=5% AND
unit-reader stationarity<=1e-6; pred_c selected cluster held capture nonworsening.
Same rank16 cubic family; no native circuit promotion from coefficient gain.
"""
import sys,os,time,json,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import numpy as np
import torch
from scipy.optimize import minimize
from cubic_projection_curvature_v1 import Curvature
from cubic_secant_coordinates_v1 import components as pair_components,normalized_reader_gradient as pair_gradient
from cubic_cluster_coordinates_v1 import components as cluster_components,normalized_reader_gradient as cluster_gradient
from cubic_secant_block_v1 import capture
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
STEM='FOLDED_CUBIC_CLUSTER_CONTINUE_V1'
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
class FitStop(Exception):pass

def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/'CUBIC_CLUSTER_CURVATURE_V1_CONTROL.json').read_text())['pred_a']
 assert json.loads((P/'CUBIC_CLUSTER_NATIVE_V1_AUDIT.json').read_text())['pred_a']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('Four matched secant/cluster continuations;0bodyforwards;300seconds each; explicit stationarity');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(1500);torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
 old=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in old if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,C,'cuda');weights={}
 for pos in [7,0]:
  r=rotary(8,128).cuda().T@rotary(pos,128).cuda();weights[pos]=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
 pairs={z['arm']:z for z in torch.load(P/'FOLDED_CUBIC_TRUST_PILOT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['programs'] if z['coordinates']=='secant'};clusters={z['arm']:z for z in torch.load(P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt',weights_only=True,map_location='cpu')};prior={r['arm']:r for r in json.loads((P/'FOLDED_CUBIC_TRUST_PILOT_V1_RESULT.json').read_text())['rows'] if r['coordinates']=='secant'}
 scale=max(r['final_capture'] for r in prior.values());rows=[];programs=[];tic=time.perf_counter()
 for arm in [0,1]:
  for mode in ['secant','cluster']:
   source=pairs[arm] if mode=='secant' else clusters[arm];start=source['theta'].cuda();chart=source['separation'] if mode=='secant' else {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in source['chart'].items()}
   components=pair_components if mode=='secant' else cluster_components;pullback=pair_gradient if mode=='secant' else cluster_gradient
   def cap(x,pos=7):
    c,m=components(x,chart);return capture(c,m,weights[pos])
   adapter=Curvature(lambda x:-cap(x)/scale,start.shape,'cuda');x0=start.cpu().numpy().ravel();initial,g0=adapter.fun(x0);best={'value':initial,'x':x0.copy()};began=time.perf_counter();history=[]
   def fun(x):
    if time.perf_counter()-began>=300:raise FitStop('time_limit')
    value,g=adapter.fun(x)
    if np.isfinite(value) and value<=best['value']:
     best.update(value=value,x=x.copy());z=adapter.tensor(x);gg=adapter.tensor(g);norm=float(pullback(z,gg,chart).norm());history.append(dict(seconds=time.perf_counter()-began,capture=-value*scale,unit_reader_gradient=norm,evaluations=adapter.evaluations,hvp=adapter.hessian_products))
     if norm<=1e-6:raise FitStop('unit_reader_stationary')
    if adapter.evaluations%20==0:
     print(json.dumps(dict(arm=arm,coordinates=mode,seconds=time.perf_counter()-began,best_capture=-best['value']*scale,evaluations=adapter.evaluations,hvp=adapter.hessian_products)),flush=True)
     torch.save(dict(arm=arm,coordinates=mode,x=torch.from_numpy(best['x']).reshape(start.shape),chart=source.get('chart',source.get('separation')),history=history),P/(STEM+'_CHECKPOINT.pt'))
    return value,g
   def hessp(x,d):
    if time.perf_counter()-began>=300:raise FitStop('time_limit')
    return adapter.hessp(x,d)
   try:
    fit=minimize(fun,x0,jac=True,hessp=hessp,method='trust-krylov',options={'maxiter':512,'gtol':1e-12,'initial_trust_radius':.1,'max_trust_radius':10.});reason=str(fit.message);iterations=int(fit.nit)
   except FitStop as stop_reason:reason=str(stop_reason);iterations=None
   except (torch.linalg.LinAlgError,ValueError,RuntimeError) as numeric_error:reason='numerical_failure: '+str(numeric_error);iterations=None
   z=adapter.tensor(best['x']);value,g=adapter.fun(best['x']);norm=float(pullback(z,adapter.tensor(g),chart).norm());final=-value*scale;held=float(cap(z,0));row=dict(arm=arm,coordinates=mode,initial_capture=prior[arm]['final_capture'],initial_held_capture=prior[arm]['held_capture'],coordinate_initial_error=abs(-initial*scale/prior[arm]['final_capture']-1),final_capture=final,gain_ratio=final/prior[arm]['final_capture'],held_capture=held,unit_reader_gradient=norm,stationary=norm<=1e-6,termination=reason,iterations=iterations,evaluations=adapter.evaluations,hessian_products=adapter.hessian_products,seconds=time.perf_counter()-began);rows.append(row)
   programs.append(dict(arm=arm,coordinates=mode,theta=z.cpu(),chart=source.get('chart',source.get('separation')),history=history));print(json.dumps(row),flush=True);(P/(STEM+'_PROGRESS.json')).write_text(json.dumps(rows,indent=2)+'\n')
 cluster_rows=[r for r in rows if r['coordinates']=='cluster'];bestrow=max(cluster_rows,key=lambda r:r['final_capture']);artifact=P/(STEM+'_ARTIFACT.pt');torch.save(dict(programs=programs,scale=scale),artifact)
 result={'pred_a':all(r['coordinate_initial_error']<=1e-6 and r['gain_ratio']>=1-1e-7 for r in rows),'pred_b':any(r['gain_ratio']>=1.05 and r['stationary'] for r in cluster_rows),'pred_c':bestrow['held_capture']>=bestrow['initial_held_capture']}
 result.update(rows=rows,selected_cluster_arm=bestrow['arm'],execution_seconds=time.perf_counter()-tic,peak_allocated_bytes=torch.cuda.max_memory_allocated(),artifact_sha=digest(artifact),scope='Longer matched warm continuation, same rank16 function family/weights objective. Chart does not add free polynomial partners. Best finite evaluated candidates retained, including trial points. No cold-start recovery or native behavior claimed; time/iteration limits remain unconverged unless independent unit-reader bar holds.')
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
