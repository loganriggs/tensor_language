#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_training pred_c_validation
"""24/8 document adaptation diagnostic, fixed rank16 source capacity.
10 fits: coefficient control at two rates; normalized/numerator empirical
functional loss x coefficient penalty 0/.01 x rates .01/.05. 400 steps.
Select rate within family using training objective only.
Primary normalized_0.01: validation error<=.15 and <=.8 initial/coefficient.
Training error<=.8 initial. Export replay<1e-8.
Frozen mode discovery used all32docs; validation is adaptation-only.
"""
import os, sys, json, time, math, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
TRAIN=24*64

def main():
 import torch
 torch.set_num_threads(2)
 sys.path.insert(0,str(P))
 from quartic_lowrank_metric import squared_error
 from source_interface import source_read
 dry=bool(os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'))
 device='cpu' if dry else 'cuda'
 torch.backends.cuda.matmul.allow_tf32=False
 start=time.perf_counter()
 raw=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)
 def move(x):
  return {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x.to(device=device,dtype=torch.float64)
 data=move(raw);c=data['common'];f=data['frames']['covariance']
 white=data['whitened_delta'];affine=data['affine']
 initial=[c['sources'][k]['initial'] for k in ['a','b']]
 def coefficient(U,V):
  factors=[]
  for k,factor,Z in [('a',-.5,U),('b',1.,V)]:
   sign=c['sources'][k]['sign']
   X=torch.cat([f['J']@Z,f[k+'_fixed_vectors'],f['unit'][:,None]],1)
   weight=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*(Z.square().sum(0)*sign).sum()).reshape(1)])
   factors.extend([X,weight])
  return squared_error(f['A'],f['B'],f['teacher_norm'],*factors)/f['teacher_norm']
 def predict(U,V,sl):
  reads=[]
  for j,(k,Z) in enumerate([('a',U),('b',V)]):
   sign=c['sources'][k]['sign']
   reads.append(affine[sl,j]+(((white[sl]@Z).square()-Z.square().sum(0))*sign).sum(1))
  return ((c['t'][sl]-.5*reads[0])/c['scale'][sl]-c['alpha'])*(reads[1]/c['scale'][sl]-c['beta'])
 def functional(U,V,kind,sl=slice(0,TRAIN)):
  truth=c['true_phi'][sl];pred=predict(U,V,sl)
  if kind=='numerator':
   scale2=c['scale'][sl].square();truth=truth*scale2;pred=pred*scale2
  return (pred-truth).square().sum()/(truth-truth.mean()).square().sum()
 def export(U,V):
  result={k:c[k].detach().cpu().clone() for k in ['h_reader','residual_writer','alpha','beta']}
  for k,Z in [('a',U),('b',V)]:
   source=c['sources'][k];proj=c['source_inverse_root']@Z;pm=c['mu']@proj
   linear=source['linear']-2*proj@(source['sign']*pm)
   bias=source['constant']-c['mu']@source['linear']+(source['sign']*(pm.square()-Z.square().sum(0))).sum()
   for suffix,tensor in [('reader',proj),('eigenvalues',source['sign']),('linear',linear),('bias',bias)]:
    result[k+'_'+suffix]=tensor.detach().cpu().clone()
  return result
 def describe(U,V):
  return dict(coefficient_relative_error=float(coefficient(U,V).clamp_min(0).sqrt()),
   train_normalized_error=float(functional(U,V,'normalized').sqrt()),
   validation_normalized_error=float(functional(U,V,'normalized',slice(TRAIN,None)).sqrt()),
   train_numerator_error=float(functional(U,V,'numerator').sqrt()),
   validation_numerator_error=float(functional(U,V,'numerator',slice(TRAIN,None)).sqrt()))
 def export_replay(U,V):
  program=move(export(U,V))
  z=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)['z'].flatten(0,1).to(device=device,dtype=torch.float64)
  a=source_read(z,program,'a');b=source_read(z,program,'b')
  pred=((c['t']-.5*a)/c['scale']-c['alpha'])*(b/c['scale']-c['beta'])
  expected=predict(U,V,slice(None))
  return float((pred-expected).norm()/expected.norm())
 if dry:
  params=[v.clone().requires_grad_() for v in initial]
  for kind in ['coefficient','normalized','numerator']:
   loss=coefficient(*params) if kind=='coefficient' else functional(*params,kind)
   grads=torch.autograd.grad(loss,params)
   assert all(torch.isfinite(g).all() and g.shape==(1152,16) for g in grads)
  with torch.no_grad():
   replay=export_replay(*initial);summary=describe(*initial)
  assert replay<1e-8
  print(json.dumps(dict(fits=10,steps=400,forwards=0,rank=16,train_documents=24,validation_documents=8,shape_and_gradient_smoke='PASS',export_replay=replay,initial=summary)))
  return
 out=P/'FUNCTIONAL_QUARTIC_MODE3_FIT_V1.json'
 assert not out.exists()
 records=[];programs={}
 with torch.no_grad():
  baseline=describe(*initial);programs['initial']=export(*initial)
 specs=[('coefficient',0.)]+[(kind,penalty) for kind in ['numerator','normalized'] for penalty in [0.,.01]]
 for kind,penalty in specs:
  for lr in [.01,.05]:
   params=[torch.nn.Parameter(v.clone()) for v in initial]
   opt=torch.optim.Adam(params,lr=lr);history=[];best=None
   for step in range(401):
    coef=coefficient(*params) if kind=='coefficient' or penalty else None
    loss=coef if kind=='coefficient' else functional(*params,kind)+(penalty*coef if penalty else 0.)
    value=float(loss.detach());assert math.isfinite(value)
    if best is None or value<best[0]:
     best=(value,step,*[v.detach().clone() for v in params])
    if step%100==0:
     history.append([step,value]);print(kind,penalty,lr,step,value,flush=True)
    if step==400:break
    opt.zero_grad();loss.backward();opt.step()
    for group in opt.param_groups:
     group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/400)))
   value,step,U,V=best;family=f'{kind}_{penalty}';label=f'{family}_lr{lr}'
   with torch.no_grad():
    row=dict(label=label,family=family,learning_rate=lr,training_objective=value,best_step=step,history=history,**describe(U,V),export_replay=export_replay(U,V))
    programs[label]=export(U,V)
   records.append(row)
 selected={f'{kind}_{penalty}':min([r for r in records if r['family']==f'{kind}_{penalty}'],key=lambda r:r['training_objective'])['label'] for kind,penalty in specs}
 primary=next(r for r in records if r['label']==selected['normalized_0.01'])
 control=next(r for r in records if r['label']==selected['coefficient_0.0'])
 prep=json.loads((P/'FUNCTIONAL_QUARTIC_MODE3_PREP_V1.json').read_text())
 predictions=dict(pred_a_instrument=prep['frame_replay']<1e-8 and max(r['export_replay'] for r in records)<1e-8,
  pred_b_training=primary['train_normalized_error']<=.8*baseline['train_normalized_error'],
  pred_c_validation=primary['validation_normalized_error']<=min(.15,.8*baseline['validation_normalized_error'],.8*control['validation_normalized_error']))
 result=dict(predictions=predictions,initial=baseline,selected_by_training_objective=selected,records=records,
  seconds=time.perf_counter()-start,input_sha256=hashlib.sha256((P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt').read_bytes()).hexdigest(),
  scope='24 training/8 adaptation-heldout documents from existing calibration. Native mode identity used all32 before split. Fixed rank16 source signs/affine branches. Primary normalized loss +.01 coefficient penalty; rate selection uses training only. No fresh/native intervention confirmation.')
 torch.save(programs,P/'FUNCTIONAL_QUARTIC_MODE3_PROGRAMS_V1.pt')
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()
