#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_components pred_c_gaussian
"""Hybrid Gaussian/native-sensitivity local quartic learning, eta=.5.
Adam .1, seeds25001/25002,250updates; fixedparent,96atoms/288products/442464coefficients.
pred_a_integrity: normal<1e-8,export<1e-4,protectedports exact.
pred_b_components: both starts larger-panel value AND response RMS <=.85 matchedGaussianAdam.
pred_c_gaussian: both retain >=.90 matchedGaussianAdam unregularized Gaussian improvement.
Null: empirical moments overfit or trade away weight fidelity without repairing components.
Opened evaluation panels; no OOD or semantic adoption. All selection uses calibration hybrid loss.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from hybrid_local_quartic import objective
 from mixed_gaussian_cp import gram_dynamic
 from noncentral_gaussian_cp import cross
 from audit_local_quartic_followup import distribution
 from audit_balanced_shared_followup import scores
 from quartic_cp_profile import normalize_factors
 from noncentral_gaussian_cp import project_shifted
 from check_hybrid_local_quartic import controls
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(25000);dt=torch.float64;d=12;t=[torch.randn(*s,dtype=dt) for s in [(12,3),(3,2),(3,2),(2,4),(4,d),(4,d)]];S=torch.eye(d,dtype=dt);mu=torch.zeros(d,dtype=dt);pr=project_shifted(t,mu);pf=normalize_factors([torch.randn(512,d,dtype=dt) for _ in range(4)]);pc=torch.randn(12,512,dtype=dt);pars=[torch.randn(96,d,dtype=dt,requires_grad=True) for _ in range(4)];loss,C,info=objective(t,t,mu,pr,S,mu,pf,pc,normalize_factors(pars),8,torch.ones(12,dtype=dt),torch.randn(23,d,dtype=dt),torch.randn(23,12,dtype=dt),torch.rand(23,12,dtype=dt)+.1);loss.backward();assert C.shape==(12,96) and all(torch.isfinite(a.grad).all() for a in pars)
  f=[torch.randn(96,1152,dtype=dt) for _ in range(4)];x=torch.randn(7,1152,dtype=dt);phi=torch.stack([x@a.T for a in f]).prod(0);packed=torch.stack([C[g,g*8:(g+1)*8] for g in range(12)]);a=(phi.reshape(7,12,8)*packed).sum(2);assert torch.allclose(a,phi@C.T)
  print(json.dumps(dict(controls=len(checks),parent512=True,new96outputs12groups8=True,export1152=True,gradient=True)));return
 manifest=json.loads((P/'HYBRID_LOCAL_QUARTIC_NATIVE_INPUTS_V1.json').read_text())
 for name,h in manifest['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'HYBRID_LOCAL_QUARTIC_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127;torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].cuda().double();mu=cache['mean'].cuda().double();loc=torch.linalg.solve(S,mu);state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=cache['writer'].cuda().double();vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
  def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].cuda().double()
  t=[readers[:,4:].T@w(17,'Down')/scale,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')];tr=t[:4]+[t[4]@S,t[5]@S];pr=project_shifted(tr,loc,tuple(a[4:].cuda().double() for a in cache['projections']['covariance']['zero_projection']));p=torch.load(P/'MIXED_CP_FEATURES_SEED1001_V1.pt',weights_only=True);pf=[a.cuda().double() for a in p['factors']];parentC=p['coefficients'].cuda().double()/scale;pc=parentC[4:]
  weights=torch.tensor(json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text())['weights'][4:],device='cuda',dtype=torch.float64);weights/=weights.mean()
  x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();y=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].cuda().double()/scale;rec,don=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat'],device='cuda').T;parent=torch.stack([x@a.T for a in pf]).prod(0)@parentC.T
  labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
  cov=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)
  extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)
  tokens=torch.load(ROOT/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:96,:65].contiguous()
  digest=lambda z:hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest()
  assert digest(tokens[:,:64])==labels['token_sha256']['calibration']
  assert digest(tokens[:32])==cov['panels'][0]['token_sha256']
  assert digest(tokens[32:])==extra['token_sha256']
  calx=torch.cat([cov['panels'][0]['rows'],extra['rows']]).cuda().double()
  caly=labels['panels'][0]['target'].cuda().double()/scale
  sensitivity=labels['panels'][0]['weight'][:,4:].cuda().double()
  assert calx.shape==(6144,1152) and caly.shape==(6144,16) and sensitivity.shape==(6144,12)
  calparent=torch.stack([calx@a.T for a in pf]).prod(0)@parentC.T
  residual=caly[:,4:]-calparent[:,4:]
  fresh=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)
  fx=fresh['rows'].cuda().double();fy=fresh['target'].cuda().double()/scale
  pair=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs'],device='cuda').T
  fparent=torch.stack([fx@a.T for a in pf]).prod(0)@parentC.T
 def predict(xx,base,fs,C):
  pred=base.clone();pred[:,4:]+=torch.stack([xx@a.T for a in fs]).prod(0)@C.T
  assert torch.equal(pred[:,:4],base[:,:4]);return pred
 def gaussian_gain(fs,C):
  fw=[a@S for a in fs];fb=[a@mu for a in fs];pw=[a@S for a in pf];pb=[a@mu for a in pf]
  G=gram_dynamic(fw,fb,fw,fb)
  X=cross(tr,loc,pr,fw,fb)-pc@gram_dynamic(pw,pb,fw,fb)
  # Evaluate ACTUAL coefficients: do not reprofile in the Gaussian metric.
  return float((weights*(2*(C*X).sum(1)-((C@G)*C).sum(1))).sum())
 def unpack(artifact):
  fs=[a.cuda().double() for a in artifact['factors']];C=torch.zeros(12,96,device='cuda',dtype=torch.float64)
  for g in range(12):C[g,g*8:(g+1)*8]=artifact['coefficients'][g].cuda().double()/scale
  return fs,C
 def calibration_score(pred):
  e=pred[:,4:]-caly[:,4:];ref=caly[:,4:];w=sensitivity/sensitivity.mean(0)
  return dict(value_errors=(e.square().sum(0)/ref.square().sum(0)).sqrt().tolist(),sensitivity_errors=((w*e.square()).sum(0)/(w*ref.square()).sum(0)).sqrt().tolist())
 def score(pred):
  e=pred[:,4:]-y[:,4:];ref=y[don,4:]-y[rec,4:];de=pred[don,4:]-pred[rec,4:]-ref;v=(e.square().sum(0)/y[:,4:].square().sum(0)).sqrt();r=(de.square().sum(0)/ref.square().sum(0)).sqrt();return dict(small_value_rms=float(v.square().mean().sqrt()),small_response_rms=float(r.square().mean().sqrt()),feature_values=v.tolist(),feature_responses=r.tolist())
 baseline=score(parent);rows=[]
 for optim in ['adam']:
  for seed in [25001,25002]:
   torch.manual_seed(seed);params=[torch.randn(96,1152,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(4)];rate=manifest['rates'][optim];opt=(torch.optim.Adam if optim=='adam' else torch.optim.Muon)(params,lr=rate);best=None;history=[];normal=0.
   for step in range(251):
    fs=normalize_factors(params);loss,C,info=objective(t,tr,loc,pr,S,mu,pf,pc,fs,8,weights,calx,residual,sensitivity,eta=manifest['eta']);value=float(loss.detach());normal=max(normal,info['normal_residual'])
    if step==0:initial=value;normalizer=max(abs(value),1e-10)
    if best is None or value<best[0]:best=(value,step,[a.detach().clone() for a in fs],C.detach().clone())
    if step%25==0:history.append(dict(step=step,objective=value,elapsed=time.monotonic()-start));print(json.dumps(dict(optimizer=optim,seed=seed,**history[-1])),flush=True)
    if step==250:break
    opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(a.grad).all() for a in params);opt.step()
   with torch.no_grad():
    value,selected,fs,C=best;phi=torch.stack([x@a.T for a in fs]).prod(0);pred=parent.clone();pred[:,4:]+=phi@C.T;assert torch.equal(pred[:,:4],parent[:,:4]);metrics=score(pred);packed=torch.stack([C[g,g*8:(g+1)*8] for g in range(12)]);arc=dict(factors=[a.cpu().float() for a in fs],coefficients=(packed*scale).cpu().float(),output_start=4,per_output=8,parent='MIXED_CP_FEATURES_SEED1001_V1.pt',parent_sha256=manifest['files']['MIXED_CP_FEATURES_SEED1001_V1.pt'])
    fp=torch.stack([x.float()@a.float().T for a in fs]).prod(0).reshape(len(x),12,8);fp=(fp*(packed*scale).float()).sum(2).double()/scale;ref=phi@C.T;drift=float((fp-ref).norm()/ref.norm().clamp_min(1e-30));path=P/f'HYBRID_LOCAL_QUARTIC_ADAM_SEED{seed}_V1.pt';torch.save(arc,path)
    exported_fs,exported_C=unpack(arc)
    freshpred=predict(fx,fparent,exported_fs,exported_C);freshmetrics=scores(freshpred,fy,*pair)
    matched=torch.load(P/f'LOCAL_QUARTIC_RESIDUAL_ADAM_SEED{seed}_V1.pt',weights_only=True)
    matched_fs,matched_C=unpack(matched)
    matchedmetrics=scores(predict(fx,fparent,matched_fs,matched_C),fy,*pair)
    gain=gaussian_gain(exported_fs,exported_C);matchedgain=gaussian_gain(matched_fs,matched_C)
    assert matchedgain>0
    docerrors=[float((((pp[:,4:]-yy[:,4:]).square().sum(0)/yy[:,4:].square().sum(0)).mean()).sqrt()) for pp,yy in zip(freshpred.split(64),fy.split(64))]
    rows.append(dict(optimizer=optim,seed=seed,rate=rate,initial_objective=initial,selected_objective=value,selected_step=selected,relative_gain=(initial-value)/max(abs(initial),1e-10),normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics,value_ratio=metrics['small_value_rms']/baseline['small_value_rms'],response_ratio=metrics['small_response_rms']/baseline['small_response_rms'],history=history,calibration=calibration_score(predict(calx,calparent,exported_fs,exported_C)),fresh_metrics=freshmetrics,matched_gaussian_metrics=matchedmetrics,matched_value_ratio=freshmetrics['small_value_rms']/matchedmetrics['small_value_rms'],matched_response_ratio=freshmetrics['small_response_rms']/matchedmetrics['small_response_rms'],gaussian_gain=gain,matched_gaussian_gain=matchedgain,gaussian_retention=gain/matchedgain,document_small_value_rms=docerrors,residual_distribution=distribution(freshpred,fy),small_output_residual_distribution=distribution(freshpred[:,4:],fy[:,4:])))
   del opt,params,best,fs,C,loss
 preds=dict(pred_a_integrity=all(r['normal_residual']<1e-8 and r['export_error']<1e-4 for r in rows),pred_b_components=all(r['matched_value_ratio']<=.85 and r['matched_response_ratio']<=.85 for r in rows),pred_c_gaussian=all(r['gaussian_retention']>=.9 for r in rows))
 out.write_text(json.dumps(dict(predictions=preds,rows=rows,baseline=baseline,seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),added_products=288,added_coefficients=442464,eta=manifest['eta'],calibration_baseline=calibration_score(calparent),scope='Hybrid exact Gaussian and empirical native tangent sensitivity; fixed CP1001; two Adam250 fits. Calibration6144 fitted; original2048/larger16384 opened panels evaluation only. No OOD, semantic or fullmodel adoption.'),indent=2)+'\n')
if __name__=='__main__':main()
