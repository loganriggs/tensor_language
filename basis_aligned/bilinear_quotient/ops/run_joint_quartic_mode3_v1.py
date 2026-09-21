#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_coefficient pred_c_native_probe
"""Joint original-mode3 quartic numerator fitting, no native forwards.
8Adamfits:iso/covariance coefficient frames x lr.01/.05 x spectral/random,
400steps, rank16per-source, fixedexactcenteredaffinebranches.36864trainablevalues.
pred_a frame/kernelreplay<1e-8, initialnativevariationmatches.37710677.
pred_b bestcovariance coefficientnormerror<=.8initial. pred_c bestcoefficient-
selected covariance candidate nativecalibrationmodevariation<=.15 and<=.8initial.
Null: jointnumeratorcoefficient improvement doesnotfix nativefunctionfidelity.
Bothmetricsusecalibrationcentering/preconditioning; covariance frame is secondmoment
weighting, not Gaussian/probeMSE. TwoRMS/inputnative ports explicit. Noadoptionclaim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  x=torch.load(P/'JOINT_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);assert x['frames']['covariance']['A'].shape==(1155,1155);assert x['common']['sources']['a']['initial'].shape==(1152,16);assert json.loads((P/'QUARTIC_LOWRANK_METRIC_CHECK_V1.json').read_text())['passed'];print(json.dumps(dict(fits=8,steps=400,rank=16,forwards=0,shape_checks='PASS')));return
 import torch
 sys.path.insert(0,str(P));from quartic_lowrank_metric import squared_error
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'JOINT_QUARTIC_MODE3_FIT_V1.json';assert not out.exists();raw=torch.load(P/'JOINT_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)
 def move(x):return {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x.cuda().double()
 data=move(raw);common=data['common'];frames=data['frames'];cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);delta=cal['z'].flatten(0,1).cuda().double()-common['mu'];records=[];programs={};initials={}
 def factors(frame,U,V):
  result=[]
  for k,factor,Z in [('a',-.5,U),('b',1.,V)]:
   sign=common['sources'][k]['sign'];X=torch.cat([frame['J']@Z,frame[k+'_fixed_vectors'],frame['unit'][:,None]],1);w=torch.cat([factor*sign,frame[k+'_fixed_values'],(-factor*(Z.square().sum(0)*sign).sum()).reshape(1)]);result.extend([X,w])
  return result
 def evaluate(U,V):
  reads=[]
  for k,Z in [('a',U),('b',V)]:
   s=common['sources'][k];proj=common['source_inverse_root']@Z;reads.append(s['constant']+delta@s['linear']+(((delta@proj).square()-Z.square().sum(0))*s['sign']).sum(1))
  phi=((common['t']-.5*reads[0])/common['scale']-common['alpha'])*(reads[1]/common['scale']-common['beta']);return float((phi-common['true_phi']).norm()/(common['true_phi']-common['true_phi'].mean()).norm())
 def export(U,V):
  e={k:common[k].detach().cpu().clone() for k in ['h_reader','residual_writer','alpha','beta']}
  for k,Z in [('a',U),('b',V)]:
   s=common['sources'][k];proj=common['source_inverse_root']@Z;pm=common['mu']@proj;linear=s['linear']-2*proj@(s['sign']*pm);bias=s['constant']-common['mu']@s['linear']+(s['sign']*(pm.square()-Z.square().sum(0))).sum()
   for suffix,t in [('reader',proj),('eigenvalues',s['sign']),('linear',linear),('bias',bias)]:e[k+'_'+suffix]=t.detach().cpu().clone()
  return e
 with torch.no_grad():
  baseline=evaluate(common['sources']['a']['initial'],common['sources']['b']['initial']);programs['initial']=export(common['sources']['a']['initial'],common['sources']['b']['initial'])
 for metric,frame in frames.items():
  with torch.no_grad():
   X,w,Y,v=factors(frame,common['sources']['a']['initial'],common['sources']['b']['initial']);initials[metric]=float((squared_error(frame['A'],frame['B'],frame['teacher_norm'],X,w,Y,v)/frame['teacher_norm']).clamp_min(0).sqrt())
  for lr in [.01,.05]:
   for init in ['spectral','random']:
    params=[];gen=torch.Generator().manual_seed(2511)
    for k in ['a','b']:
     Z=common['sources'][k]['initial'].clone()
     if init=='random':
      random=torch.randn(Z.shape,dtype=torch.float64,generator=gen).cuda();Z=random/random.norm(dim=0,keepdim=True)*Z.norm(dim=0,keepdim=True)
     params.append(torch.nn.Parameter(Z))
    U,V=params;opt=torch.optim.Adam(params,lr=lr);best=None;history=[];runstart=time.perf_counter()
    for step in range(401):
     X,w,Y,v=factors(frame,U,V);loss=squared_error(frame['A'],frame['B'],frame['teacher_norm'],X,w,Y,v)/frame['teacher_norm'];value=float(loss.detach());assert math.isfinite(value)
     if best is None or value<best[0]:best=(value,U.detach().clone(),V.detach().clone(),step)
     if step%100==0:history.append([step,value]);print(metric,lr,init,step,value,flush=True)
     if step==400:break
     opt.zero_grad();loss.backward();opt.step()
     for group in opt.param_groups:group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/400)))
    with torch.no_grad():
     value,bu,bv,step=best;label=f'{metric}_{lr}_{init}';records.append(dict(label=label,metric=metric,learning_rate=lr,initialization=init,coefficient_relative_error=max(value,0.)**.5,native_calibration_variation_error=evaluate(bu,bv),best_step=step,history=history,seconds=time.perf_counter()-runstart));programs[label]=export(bu,bv)
 selected={m:min([r for r in records if r['metric']==m],key=lambda r:r['coefficient_relative_error'])['label'] for m in frames};chosen=next(r for r in records if r['label']==selected['covariance']);prep=json.loads((P/'JOINT_QUARTIC_MODE3_PREP_V1.json').read_text());pred=dict(pred_a_instrument=prep['frame_replay']<1e-8 and abs(baseline-.3771067718710561)<1e-7,pred_b_coefficient=chosen['coefficient_relative_error']<=.8*initials['covariance'],pred_c_native_probe=chosen['native_calibration_variation_error']<=min(.15,.8*baseline))
 result=dict(predictions=pred,selected_by_coefficient_loss=selected,initial_coefficient_errors=initials,initial_native_calibration_variation_error=baseline,records=records,seconds=time.perf_counter()-start,input_sha256=hashlib.sha256((P/'JOINT_QUARTIC_MODE3_INPUTS_V1.pt').read_bytes()).hexdigest(),scope='Joint full1155-coordinate quartic numerator fit to originalmode3. Nativecalibration probes do not selectwinners. No fresh/nativeintervention evaluation; normalized execution and semanticreuse stillrequired.');torch.save(programs,P/'JOINT_QUARTIC_MODE3_PROGRAMS_V1.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
