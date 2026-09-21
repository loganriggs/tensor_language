#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_simplicity
"""Joint mixed-product decomposition of six original folded source forms.
Four Adam fits,384products,4000cosine steps; rates.01/.05,two starts.
pred_a_instrument implicit/dense loss agrees<1e-8; finite factors.
pred_b_fidelity selected-by-weight-loss candidate every component<=15% and
<=1.10 separate baseline on seven opened prefixes. pred_c_simplicity float
storage<90% baseline and384products<768. Null: cheaper products lose native
fidelity even if coefficient loss improves. No native model forwards.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'MIXED_PRODUCT_NATIVE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from shared_quadratic_products import mixed_coefficient_loss,materialize_mixed
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();output=P/'MIXED_PRODUCT_NATIVE_FIT_V1.json';assert not output.exists()
 path=P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['input_sha256']
 data=torch.load(path,weights_only=True);teacher=data['teacher'].cuda();records=[];programs={};states={}
 initpath=P/'MIXED_PRODUCT_NATIVE_INITIALIZATION_V1.pt';assert hashlib.sha256(initpath.read_bytes()).hexdigest()==plan['initialization_sha256'];init=torch.load(initpath,weights_only=True)
 for lr in plan['learning_rates']:
  for kind in plan['starts']:
   torch.manual_seed(4015)
   L=init['left'].cuda().clone();R=init['right'].cuda().clone();W=init['weights'].cuda().clone()
   if kind!='spectral':
    L+=.01*L.std()*torch.randn_like(L);R+=.01*R.std()*torch.randn_like(R);W+=.01*W.std()*torch.randn_like(W)
   L.requires_grad_();R.requires_grad_();W.requires_grad_();opt=torch.optim.Adam([L,R,W],lr=lr)
   initial=float(mixed_coefficient_loss(teacher,L,R,W).detach());best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss=mixed_coefficient_loss(teacher,L,R,W);value=float(loss.detach())
    assert math.isfinite(value)
    if value<best:best=value;best_state=(L.detach().clone(),R.detach().clone(),W.detach().clone());best_step=step
    if step%400==0:history.append(dict(step=step,loss=value));print(lr,kind,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=lr*.5*(1+math.cos(math.pi*step/plan['steps']))
    loss.backward();opt.step()
   L,R,W=[v.cpu() for v in best_state];key=f'{lr}_{kind}';states[key]=(L,R,W)
   with torch.no_grad():
    explicit=float((materialize_mixed(L,R,W)-data['teacher']).square().sum()/data['teacher'].square().sum())
    left=data['inverse_root']@L;right=data['inverse_root']@R;readout=W.T*data['scales'][None,:];linears=[];biases=[]
    for j,pair in enumerate(data['pairs']):
     for k,Q in enumerate(pair['Qs']):
      pos=2*j+k;raw=(left*readout[:,pos])@right.T;Qnew=.5*(raw+raw.T)
      linear=2*Q@data['mu'];constant=data['mu']@Q@data['mu']+torch.trace(data['old_covariance']@Q)
      linears.append(linear-2*Qnew@data['mu'])
      biases.append(constant-data['mu']@linear+data['mu']@Qnew@data['mu']-torch.trace(data['old_covariance']@Qnew))
    program=dict(left_reader=left,right_reader=right,product_weights=readout,source_linear=torch.stack(linears,1),source_bias=torch.stack(biases),h_readers=torch.stack([p['a'] for p in data['pairs']],1),alpha=torch.stack([p['alpha'] for p in data['pairs']]),beta=torch.stack([p['beta'] for p in data['pairs']]),residual_writer=data['residual_writer'])
    z=data['z'];reads=((z@left)*(z@right))@readout+z@program['source_linear']+program['source_bias']
    phi=((data['h']@program['h_readers']-.5*reads[:,0::2])/data['scale'][:,None]-program['alpha'])*(reads[:,1::2]/data['scale'][:,None]-program['beta'])
    errors=[]
    for j,pair in enumerate(data['pairs']):
     truth=pair['truth'][data['indices']];errors.append(float((phi[data['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
    records.append(dict(key=key,initial_coefficient_error=math.sqrt(max(initial,0)),coefficient_error=math.sqrt(max(best,0)),dense_loss_replay=abs(best-explicit),best_step=best_step,per_mode_errors=errors,stored_floats=sum(v.numel() for v in program.values()),source_products=left.shape[1],history=history))
    programs[key]=program
 winner=min(records,key=lambda r:r['coefficient_error']);baseline=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text());ratios=[a/b for a,b in zip(winner['per_mode_errors'],baseline['baseline_errors'])]
 result=dict(plan=plan,records=records,winner=winner['key'],winner_ratios_to_baseline=ratios,predictions=dict(pred_a_instrument=max(r['dense_loss_replay'] for r in records)<1e-8,pred_b_fidelity=max(winner['per_mode_errors'])<=.15 and max(ratios)<=1.1,pred_c_simplicity=winner['stored_floats']<=1.01*897804 and winner['source_products']<=384),seconds=time.perf_counter()-start)
 torch.save(programs,P/'MIXED_PRODUCT_NATIVE_PROGRAMS_V1.pt');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['predictions']),flush=True)
if __name__=='__main__':main()
