#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_simplicity
"""Joint shared-square decomposition of six original folded source forms.
Four Adam fits,512products,4000cosine steps; rates.01/.05,two starts.
pred_a_instrument implicit/dense loss agrees<1e-8; finite factors.
pred_b_fidelity selected-by-weight-loss candidate every component<=15% and
<=1.10 separate baseline on seven opened prefixes. pred_c_simplicity float
storage<90% baseline and512products<768. Null: cheaper products lose native
fidelity even if coefficient loss improves. No native model forwards.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'SHARED_PRODUCT_NATIVE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from shared_quadratic_products import coefficient_loss,materialize
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();output=P/'SHARED_PRODUCT_NATIVE_FIT_V1.json';assert not output.exists()
 path=P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['input_sha256']
 data=torch.load(path,weights_only=True);teacher=data['teacher'].cuda();records=[];programs={};states={}
 for lr in plan['learning_rates']:
  for kind in plan['starts']:
   torch.manual_seed(4015)
   F=data['initial_reader'].cuda().clone();W=data['initial_weights'].cuda().clone()
   if kind!='spectral':
    F+=.01*F.std()*torch.randn_like(F);W+=.01*W.std()*torch.randn_like(W)
   F.requires_grad_();W.requires_grad_();opt=torch.optim.Adam([F,W],lr=lr)
   initial=float(coefficient_loss(teacher,F,W));best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss=coefficient_loss(teacher,F,W);value=float(loss.detach())
    assert math.isfinite(value)
    if value<best:best=value;best_state=(F.detach().clone(),W.detach().clone());best_step=step
    if step%400==0:history.append(dict(step=step,loss=value));print(lr,kind,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=lr*.5*(1+math.cos(math.pi*step/plan['steps']))
    loss.backward();opt.step()
   F,W=[v.cpu() for v in best_state];key=f'{lr}_{kind}';states[key]=(F,W)
   with torch.no_grad():
    explicit=float((materialize(F,W)-data['teacher']).square().sum()/data['teacher'].square().sum())
    reader=data['inverse_root']@F;readout=W.T*data['scales'][None,:];linears=[];biases=[]
    for j,pair in enumerate(data['pairs']):
     for k,Q in enumerate(pair['Qs']):
      pos=2*j+k;Qnew=(reader*readout[:,pos])@reader.T
      linear=2*Q@data['mu'];constant=data['mu']@Q@data['mu']+torch.trace(data['old_covariance']@Q)
      linears.append(linear-2*Qnew@data['mu'])
      biases.append(constant-data['mu']@linear+data['mu']@Qnew@data['mu']-torch.trace(data['old_covariance']@Qnew))
    program=dict(shared_reader=reader,product_weights=readout,source_linear=torch.stack(linears,1),source_bias=torch.stack(biases),h_readers=torch.stack([p['a'] for p in data['pairs']],1),alpha=torch.stack([p['alpha'] for p in data['pairs']]),beta=torch.stack([p['beta'] for p in data['pairs']]),residual_writer=data['residual_writer'])
    z=data['z'];reads=(z@reader).square()@readout+z@program['source_linear']+program['source_bias']
    phi=((data['h']@program['h_readers']-.5*reads[:,0::2])/data['scale'][:,None]-program['alpha'])*(reads[:,1::2]/data['scale'][:,None]-program['beta'])
    errors=[]
    for j,pair in enumerate(data['pairs']):
     truth=pair['truth'][data['indices']];errors.append(float((phi[data['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
    records.append(dict(key=key,initial_coefficient_error=math.sqrt(max(initial,0)),coefficient_error=math.sqrt(max(best,0)),dense_loss_replay=abs(best-explicit),best_step=best_step,per_mode_errors=errors,stored_floats=sum(v.numel() for v in program.values()),source_products=reader.shape[1],history=history))
    programs[key]=program
 winner=min(records,key=lambda r:r['coefficient_error']);baseline=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text());ratios=[a/b for a,b in zip(winner['per_mode_errors'],baseline['baseline_errors'])]
 result=dict(plan=plan,records=records,winner=winner['key'],winner_ratios_to_baseline=ratios,predictions=dict(pred_a_instrument=max(r['dense_loss_replay'] for r in records)<1e-8,pred_b_fidelity=max(winner['per_mode_errors'])<=.15 and max(ratios)<=1.1,pred_c_simplicity=winner['stored_floats']<.9*baseline['baseline_stored_floats'] and winner['source_products']<768),seconds=time.perf_counter()-start)
 torch.save(programs,P/'SHARED_PRODUCT_NATIVE_PROGRAMS_V1.pt');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['predictions']),flush=True)
if __name__=='__main__':main()
