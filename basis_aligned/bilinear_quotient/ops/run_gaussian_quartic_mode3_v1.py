#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_gaussian pred_c_native_probe
"""Exact Gaussian numerator fitting:4Adam fits,rates.01/.05,spectral/perturbed.
Training24 jointmean/covariance, fixedconstantcoordinate, rank16,400steps.
Gaussian loss selects winner; Gaussianerror<=.8initial. Nativeopenedseven
distinctrows error<=.15 and<=.8initial. No GaussianinverseRMS expectation.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  torch.set_num_threads(2);sys.path.insert(0,str(P))
  from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
  x=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=x['common'];f=x['frames']['covariance'];teacher=prepare_teacher(f['A'],f['B']);parts=[]
  for k,factor in [('a',-.5),('b',1.)]:
   Z=c['sources'][k]['initial'].clone().requires_grad_();sign=c['sources'][k]['sign'];X=torch.cat([f['J']@Z,f[k+'_fixed_vectors'],f['unit'][:,None]],1);w=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*(Z.square().sum(0)*sign).sum()).reshape(1)]);parts.extend([X,w])
  loss=squared_error_by_degree(teacher,*parts).sum()/teacher['variance'];loss.backward();assert abs(float(loss.detach().sqrt())-.09025675109803488)<1e-8
  assert json.loads((P/'GAUSSIAN_QUARTIC_LOWRANK_CHECK_V1.json').read_text())['passed']
  print(json.dumps(dict(fits=4,steps=400,rank=16,forwards=0,full_initial_gaussian_error=float(loss.detach().sqrt()),shape_and_gradient_check='PASS')));return
 import torch
 sys.path.insert(0,str(P));from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'GAUSSIAN_QUARTIC_MODE3_FIT_V1.json';assert not out.exists();raw=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)
 def move(x):return {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x.cuda().double()
 data=move(raw);common=data['common'];frames=data['frames'];cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);delta=cal['z'].flatten(0,1).cuda().double()-common['mu'];records=[];programs={};initials={}
 teacher=prepare_teacher(frames['covariance']['A'],frames['covariance']['B']);frames['covariance']['teacher_norm']=teacher['variance']
 def squared_error(A,B,norm,X,w,Y,v):return squared_error_by_degree(teacher,X,w,Y,v).sum()
 hashes=[hashlib.sha256(row[:64].numpy().tobytes()).hexdigest() for row in cal['tokens']]
 docs=[i for i in range(24,32) if hashes[i] not in set(hashes[:24])]
 assert docs==[24,25,26,27,29,30,31]
 eval_indices=torch.tensor([j for i in docs for j in range(i*64,(i+1)*64)],device='cuda')

 def factors(frame,U,V):
  result=[]
  for k,factor,Z in [('a',-.5,U),('b',1.,V)]:
   sign=common['sources'][k]['sign'];X=torch.cat([frame['J']@Z,frame[k+'_fixed_vectors'],frame['unit'][:,None]],1);w=torch.cat([factor*sign,frame[k+'_fixed_values'],(-factor*(Z.square().sum(0)*sign).sum()).reshape(1)]);result.extend([X,w])
  return result
 def evaluate(U,V):
  reads=[]
  for k,Z in [('a',U),('b',V)]:
   s=common['sources'][k];proj=common['source_inverse_root']@Z;reads.append(s['constant']+delta@s['linear']+(((delta@proj).square()-Z.square().sum(0))*s['sign']).sum(1))
  phi=((common['t']-.5*reads[0])/common['scale']-common['alpha'])*(reads[1]/common['scale']-common['beta']);truth=common['true_phi'][eval_indices];return float((phi[eval_indices]-truth).norm()/(truth-truth.mean()).norm())
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
   for init in ['spectral','perturbed']:
    params=[];gen=torch.Generator().manual_seed(2511)
    for k in ['a','b']:
     Z=common['sources'][k]['initial'].clone()
     if init=='perturbed':
      random=torch.randn(Z.shape,dtype=torch.float64,generator=gen).cuda();Z=Z+.01*random/random.norm(dim=0,keepdim=True)*Z.norm(dim=0,keepdim=True)
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
     value,bu,bv,step=best;label=f'{metric}_{lr}_{init}';records.append(dict(label=label,metric=metric,learning_rate=lr,initialization=init,gaussian_numerator_variation_error=max(value,0.)**.5,native_opened_evaluation_error=evaluate(bu,bv),best_step=step,history=history,seconds=time.perf_counter()-runstart));programs[label]=export(bu,bv)
 selected={m:min([r for r in records if r['metric']==m],key=lambda r:r['gaussian_numerator_variation_error'])['label'] for m in frames};chosen=next(r for r in records if r['label']==selected['covariance']);prep=json.loads((P/'FUNCTIONAL_QUARTIC_MODE3_PREP_V1.json').read_text());pred=dict(pred_a_instrument=prep['frame_replay']<1e-8 and abs(baseline-.3986787277891258)<1e-7,pred_b_gaussian=chosen['gaussian_numerator_variation_error']<=.8*initials['covariance'],pred_c_native_probe=chosen['native_opened_evaluation_error']<=min(.15,.8*baseline))
 result=dict(predictions=pred,selected_by_gaussian_loss=selected,initial_gaussian_errors=initials,initial_native_opened_evaluation_error=baseline,records=records,seconds=time.perf_counter()-start,input_sha256=hashlib.sha256((P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt').read_bytes()).hexdigest(),evaluation_documents=docs,scope='Exact Gaussian numerator moment objective with constant coordinate fixed1 and training24-document jointmean/covariance. Four fixedrank16 fits; winner chosen by Gaussianobjective only. Native normalized error scored on seven distinct previouslyopened rows, no refitting to them. NoGaussianinverseRMS, nativeintervention or semanticadoptionclaim.');torch.save(programs,P/'GAUSSIAN_QUARTIC_MODE3_PROGRAMS_V1.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
