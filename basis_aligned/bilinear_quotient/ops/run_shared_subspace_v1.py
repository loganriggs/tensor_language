#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_primary_covariance pred_c_primary_native
"""Dense-core relaxation of shared128/private224 input spans, unchanged6native forms.
pred_a finite gradients, dense loss replay<1e-8, projection residual orthogonality<1e-8.
pred_b primary covariance error<=.07734469220772448.
pred_c primary native error<=.6078558841808147.
Null: moving the tied subspaces still fails the original coefficient guards.
This is a capacity diagnostic; no sparse circuit, cost saving, component or OOD claim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'SHARED_SUBSPACE_NATIVE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from shared_subspace_loss import SharedSubspaceLoss
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;t0=time.monotonic()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/'SHARED_SUBSPACE_NATIVE_V1.json').exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True)
 Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]).cuda();S=torch.linalg.inv(d['inverse_root']).cuda();I=torch.eye(1152,dtype=Q.dtype,device='cuda');rows=[];states={}
 for geometry in plan['metrics']:
  A,inv=(S,d['inverse_root'].cuda()) if geometry=='calibration_shaped' else (I,I)
  source=plan['parents'][geometry];program=programs[source];initial=[A@x.cuda() for x in [program['input_basis']]+[program['pairs'][str(j)]['private_reader'] for j in range(3)]];metric=SharedSubspaceLoss(A@Q@A)
  for seed in plan['seeds']:
   if seed:
    rng=torch.Generator().manual_seed(seed);initial_params=[(torch.randn(x.shape,dtype=x.dtype,generator=rng)/x.shape[0]**.5).cuda() for x in initial]
   else:initial_params=initial
   params=[p.detach().clone().requires_grad_() for p in initial_params];opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss=metric.loss(params);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;saved=[p.detach().clone() for p in params];beststep=step
    if step%300==0:history.append(dict(step=step,loss=value));print(geometry,seed,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();assert all(torch.isfinite(p.grad).all() for p in params);opt.step()
   with torch.no_grad():
    explicit=float(metric.loss(saved,True));replay=abs(best-explicit);assert replay<1e-8
    hats=[];orth=[]
    for U,T in zip(metric.spans(saved),metric.targets):
     H=U@(U.T@T@U)@U.T;orth.append(float((U.T@(T-H)@U).norm()/T.norm()));hats.append(inv@H@inv)
    assert max(orth)<1e-8
    H=torch.cat(hats);scores=[]
    for B in (I,S):
     T=B@Q@B;E=B@(H-Q)@B;scores.append(float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt()))
    key=f'{geometry}_{seed}';states[key]=[p.cpu().clone() for p in saved];row=dict(key=key,geometry=geometry,seed=seed,best_step=beststep,objective=best,native_error=scores[0],covariance_error=scores[1],dense_loss_replay=replay,projection_orthogonality=max(orth),history=history);rows.append(row);print('RECORD',json.dumps(row),flush=True)
 primary=min((r for r in rows if r['geometry']=='calibration_shaped'),key=lambda r:r['objective'])
 pred=dict(pred_a_instrument=all(r['dense_loss_replay']<1e-8 and r['projection_orthogonality']<1e-8 for r in rows),pred_b_primary_covariance=primary['covariance_error']<=plan['covariance_guard'],pred_c_primary_native=primary['native_error']<=plan['native_guard'])
 torch.save(states,P/'SHARED_SUBSPACE_NATIVE_STATES_V1.pt');(P/'SHARED_SUBSPACE_NATIVE_V1.json').write_text(json.dumps(dict(plan=plan,records=rows,primary=primary['key'],predictions=pred,seconds=time.monotonic()-t0,scope='Unpriced dense-core relaxation of shared128/private224 supports. Selected6forms only; not a sparse graph or native behavior test.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
