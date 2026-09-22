#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_learning pred_c_components
"""New96quartic residualatoms, fixedsmalloutputports, four250stepfits.
Integritynormal<1e-8/export<1e-4; >=10%objectivegain allarms;
bothstarts oneoptimizer smallvalue+response<=.85parent. Openedpanel only.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from local_quartic_residual import objective
 from quartic_cp_profile import normalize_factors
 from noncentral_gaussian_cp import project_shifted
 from check_local_quartic_residual import controls
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(25000);dt=torch.float64;d=12;t=[torch.randn(*s,dtype=dt) for s in [(12,3),(3,2),(3,2),(2,4),(4,d),(4,d)]];S=torch.eye(d,dtype=dt);mu=torch.zeros(d,dtype=dt);pr=project_shifted(t,mu);pf=normalize_factors([torch.randn(512,d,dtype=dt) for _ in range(4)]);pc=torch.randn(12,512,dtype=dt);pars=[torch.randn(96,d,dtype=dt,requires_grad=True) for _ in range(4)];loss,C,info=objective(t,t,mu,pr,S,mu,pf,pc,normalize_factors(pars),8);loss.backward();assert C.shape==(12,96) and all(torch.isfinite(a.grad).all() for a in pars)
  f=[torch.randn(96,1152,dtype=dt) for _ in range(4)];x=torch.randn(7,1152,dtype=dt);phi=torch.stack([x@a.T for a in f]).prod(0);packed=torch.stack([C[g,g*8:(g+1)*8] for g in range(12)]);a=(phi.reshape(7,12,8)*packed).sum(2);assert torch.allclose(a,phi@C.T)
  print(json.dumps(dict(controls=len(checks),parent512=True,new96outputs12groups8=True,export1152=True,gradient=True)));return
 manifest=json.loads((P/'LOCAL_QUARTIC_RESIDUAL_NATIVE_INPUTS_V1.json').read_text())
 for name,h in manifest['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'LOCAL_QUARTIC_RESIDUAL_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127;torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].cuda().double();mu=cache['mean'].cuda().double();loc=torch.linalg.solve(S,mu);state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=cache['writer'].cuda().double();vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
  def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].cuda().double()
  t=[readers[:,4:].T@w(17,'Down')/scale,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')];tr=t[:4]+[t[4]@S,t[5]@S];pr=project_shifted(tr,loc,tuple(a[4:].cuda().double() for a in cache['projections']['covariance']['zero_projection']));p=torch.load(P/'MIXED_CP_FEATURES_SEED1001_V1.pt',weights_only=True);pf=[a.cuda().double() for a in p['factors']];parentC=p['coefficients'].cuda().double()/scale;pc=parentC[4:]
  weights=torch.tensor(json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text())['weights'][4:],device='cuda',dtype=torch.float64);weights/=weights.mean()
  x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();y=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda().double()/scale;rec,don=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat'],device='cuda').T;parent=torch.stack([x@a.T for a in pf]).prod(0)@parentC.T
 def score(pred):
  e=pred[:,4:]-y[:,4:];ref=y[don,4:]-y[rec,4:];de=pred[don,4:]-pred[rec,4:]-ref;v=(e.square().sum(0)/y[:,4:].square().sum(0)).sqrt();r=(de.square().sum(0)/ref.square().sum(0)).sqrt();return dict(small_value_rms=float(v.square().mean().sqrt()),small_response_rms=float(r.square().mean().sqrt()),feature_values=v.tolist(),feature_responses=r.tolist())
 baseline=score(parent);rows=[]
 for optim in ['adam','muon']:
  for seed in [25001,25002]:
   torch.manual_seed(seed);params=[torch.randn(96,1152,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(4)];rate=manifest['rates'][optim];opt=(torch.optim.Adam if optim=='adam' else torch.optim.Muon)(params,lr=rate);best=None;history=[];normal=0.
   for step in range(251):
    fs=normalize_factors(params);loss,C,info=objective(t,tr,loc,pr,S,mu,pf,pc,fs,8,weights);value=float(loss.detach());normal=max(normal,info['normal_residual'])
    if step==0:initial=value;normalizer=max(abs(value),1e-10)
    if best is None or value<best[0]:best=(value,step,[a.detach().clone() for a in fs],C.detach().clone())
    if step%25==0:history.append(dict(step=step,objective=value,elapsed=time.monotonic()-start));print(json.dumps(dict(optimizer=optim,seed=seed,**history[-1])),flush=True)
    if step==250:break
    opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(a.grad).all() for a in params);opt.step()
   with torch.no_grad():
    value,selected,fs,C=best;phi=torch.stack([x@a.T for a in fs]).prod(0);pred=parent.clone();pred[:,4:]+=phi@C.T;assert torch.equal(pred[:,:4],parent[:,:4]);metrics=score(pred);packed=torch.stack([C[g,g*8:(g+1)*8] for g in range(12)]);arc=dict(factors=[a.cpu().float() for a in fs],coefficients=(packed*scale).cpu().float(),output_start=4,per_output=8,parent='MIXED_CP_FEATURES_SEED1001_V1.pt',parent_sha256=manifest['files']['MIXED_CP_FEATURES_SEED1001_V1.pt'])
    fp=torch.stack([x.float()@a.float().T for a in fs]).prod(0).reshape(len(x),12,8);fp=(fp*(packed*scale).float()).sum(2).double()/scale;ref=phi@C.T;drift=float((fp-ref).norm()/ref.norm().clamp_min(1e-30));path=P/f'LOCAL_QUARTIC_RESIDUAL_{optim.upper()}_SEED{seed}_V1.pt';torch.save(arc,path)
    rows.append(dict(optimizer=optim,seed=seed,rate=rate,initial_objective=initial,selected_objective=value,selected_step=selected,relative_gain=(initial-value)/max(abs(initial),1e-10),normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics,value_ratio=metrics['small_value_rms']/baseline['small_value_rms'],response_ratio=metrics['small_response_rms']/baseline['small_response_rms'],history=history))
   del opt,params,best,fs,C,loss
 preds=dict(pred_a_integrity=all(r['normal_residual']<1e-8 and r['export_error']<1e-4 for r in rows),pred_b_learning=all(r['relative_gain']>=.1 for r in rows),pred_c_components=any(all(r['value_ratio']<=.85 and r['response_ratio']<=.85 for r in rows if r['optimizer']==o) for o in ['adam','muon']))
 out.write_text(json.dumps(dict(predictions=preds,rows=rows,baseline=baseline,seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),added_products=288,added_coefficients=442464,scope='Newoutputlocalquartic residuallearning; fixedCP1001parent; four250steparms; exactGaussianweights/data-informedmetric; no textlabelsfitted; openedpanel screen, not OOD/semantic/fullmodeladoption.'),indent=2)+'\n')
if __name__=='__main__':main()
