#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_primary_covariance pred_c_primary_native pred_d_matched_warm_improvement
"""Pairwise versus common dictionary relaxation, unchanged6native forms.
pred_a finite gradients, dense loss replay<1e-8, projection residual orthogonality<1e-8.
pred_b primary covariance error<=.07734469220772448.
pred_c primary native error<=.6078558841808147.
pred_d pairwise warm fit root error<=.99 common warm continuation in both geometries.
Null: pairwise reuse does not improve the matched-budget common continuation.
This is a capacity diagnostic; no sparse circuit, cost saving, component or OOD claim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'PAIRWISE_SUBSPACE_NATIVE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from shared_subspace_loss import SharedSubspaceLoss
 from pairwise_subspace_loss import PairwiseSubspaceLoss,from_common
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;t0=time.monotonic()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);parent_states=torch.load(P/'SHARED_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True)
 Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]).cuda();S=torch.linalg.inv(d['inverse_root']).cuda();I=torch.eye(1152,dtype=Q.dtype,device='cuda');rows=[];states={}
 for geometry in plan['metrics']:
  A,inv=(S,d['inverse_root'].cuda()) if geometry=='calibration_shaped' else (I,I)
  parent=[p.cuda() for p in parent_states[plan['parents'][geometry]]]
  for topology,seed in [('common',0),('pairwise',0),('pairwise',26301)]:
   initial=parent if topology=='common' else from_common(parent)
   metric=(SharedSubspaceLoss if topology=='common' else PairwiseSubspaceLoss)(A@Q@A)
   if seed:
    rng=torch.Generator().manual_seed(seed);initial_params=[(torch.randn(x.shape,dtype=x.dtype,generator=rng)/x.shape[0]**.5).cuda() for x in initial]
   else:initial_params=initial
   params=[p.detach().clone().requires_grad_() for p in initial_params];opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss=metric.loss(params);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;saved=[p.detach().clone() for p in params];beststep=step
    if step%300==0:history.append(dict(step=step,loss=value));print(geometry,topology,seed,step,value,flush=True)
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
    key=f'{geometry}_{topology}_{seed}';states[key]=[p.cpu().clone() for p in saved];row=dict(key=key,geometry=geometry,topology=topology,seed=seed,best_step=beststep,objective=best,native_error=scores[0],covariance_error=scores[1],dense_loss_replay=replay,projection_orthogonality=max(orth),history=history);rows.append(row);print('RECORD',json.dumps(row),flush=True)
 primary=min((r for r in rows if r['geometry']=='calibration_shaped' and r['topology']=='pairwise'),key=lambda r:r['objective'])
 matched=[]
 for geometry in plan['metrics']:
  field='covariance_error' if geometry=='calibration_shaped' else 'native_error'
  arms={r['topology']:r[field] for r in rows if r['geometry']==geometry and r['seed']==0}
  matched.append(dict(geometry=geometry,common=arms['common'],pairwise=arms['pairwise'],passes=arms['pairwise']<=.99*arms['common']))
 pred=dict(pred_d_matched_warm_improvement=all(r['passes'] for r in matched),pred_a_instrument=all(r['dense_loss_replay']<1e-8 and r['projection_orthogonality']<1e-8 for r in rows),pred_b_primary_covariance=primary['covariance_error']<=plan['covariance_guard'],pred_c_primary_native=primary['native_error']<=plan['native_guard'])
 torch.save(states,P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt');(P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').write_text(json.dumps(dict(plan=plan,records=rows,primary=primary['key'],matched_warm_comparisons=matched,predictions=pred,seconds=time.monotonic()-t0,scope='Unpriced dense-core comparison: common128/private224 versus three pairwise64/private224 supports. Equal extra fitting steps for warm controls. Selected6forms only; not a sparse graph or native behavior test.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
