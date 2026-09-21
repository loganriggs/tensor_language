#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_components pred_c_coefficients pred_d_arithmetic pred_e_warm_progress
"""Free private input directions with exact symmetric-core variable projection.
pred_a all fits finite, implicit/full loss and compiled export replay<1e-8.
pred_b primary all3values<=.15and<=1.10baseline; fixed-h zJacobian<=1.10baseline.
pred_c primary native/covariance coefficient errors<=1.10original baseline.
pred_d same1058124floats/1047648source mults, >=20%arithmetic reduction.
pred_e both warm runs improve fitting error by at least 1% versus their identical initial functions.
Null: moving private directions does not rescue faithful cheap reuse.
Shared pairwise dictionaries fixed; conditional nativez/h and old448limitations remain.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'ORTHOGONAL_PRIVATE_NATIVE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from orthogonal_private_varpro import OrthogonalPrivateMetric
 from pairwise_reader_graph import GROUPS
 from export_orthogonal_private import export
 from pairwise_graph_assessment import Assessment
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/'ORTHOGONAL_PRIVATE_NATIVE_V1.json').exists()
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);parents=torch.load(P/'ALTERNATING_COMPLETION_PROGRAMS_V1.pt',weights_only=True);audit=Assessment(data);records=[];programs={};states={}
 for geometry in plan['metrics']:
  template=parents[plan['parents'][geometry]];A,inv=(audit.S,data['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);shared=[A@torch.cat([template['input_bases'][str(a)],template['input_bases'][str(b)]],1) for a,b in GROUPS];private=[A@template['pairs'][str(j)]['private_reader'] for j in range(3)];initial=[torch.linalg.qr(v,mode="reduced").Q for v in private];metric=OrthogonalPrivateMetric((A@audit.Q@A).cuda(),[s.cuda() for s in shared])
  for seed in plan['seeds']:
   rng=torch.Generator().manual_seed(seed);params=[(p.clone() if seed==0 else torch.randn(p.shape,dtype=p.dtype,generator=rng)/p.shape[0]**.5).cuda().requires_grad_() for p in initial];opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw') if plan['optimizer']=='muon' else torch.optim.Adam(params,lr=plan['rate']);best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,_=metric.loss(params);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;saved=[p.detach().clone() for p in params];beststep=step
    if step%300==0:history.append(dict(step=step,loss=value));print(geometry,seed,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();assert all(torch.isfinite(p.grad).all() for p in params);opt.step()
   key=f'{geometry}_{seed}';states[key]=[p.cpu().clone() for p in saved]
   with torch.no_grad():
    loss,components=metric.loss(saved);explicit=metric.loss(saved,dense=True)[0];replay=abs(float(loss-explicit));assert replay<1e-8
    row=dict(key=key,geometry=geometry,seed=seed,objective=best,best_step=beststep,dense_loss_replay=replay,history=history)
    try:
     graph,diag=export(components,metric.scales,template,A,inv);graph=audit.correct(graph);scores=audit.assess(graph);field='covariance_error' if geometry=='calibration_shaped' else 'native_error';export_replay=abs(scores[field]-float(explicit.sqrt()));assert export_replay<1e-8 and scores['stored_floats']==scores['physical_storage_floats']==1058124 and scores['source_total_multiplications']==1047648
     programs[key]=graph;row.update(instrument=True,export_replay=export_replay,compiler=diag,**scores)
    except (ValueError,RuntimeError) as error:row.update(instrument=False,compiler_failure=str(error))
    records.append(row);print('RECORD',json.dumps({k:v for k,v in row.items() if k not in ('compiler','history')}),flush=True)
 primary=min((r for r in records if r['geometry']=='calibration_shaped'),key=lambda r:r['objective']);base=plan['baseline'];valid=primary['instrument']
 pred=dict(pred_e_warm_progress=all(r['objective']<=plan['warm_improvement_factor']*plan['warm_objectives'][r['geometry']] for r in records if r['seed']==0),pred_a_instrument=all(r['instrument'] for r in records),pred_b_components=valid and all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_coefficients=valid and all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')),pred_d_arithmetic=valid and primary['source_total_multiplications']<=.8*base['source_total_multiplications'])
 torch.save(states,P/'ORTHOGONAL_PRIVATE_NATIVE_STATES_V1.pt');torch.save(programs,P/'ORTHOGONAL_PRIVATE_NATIVE_PROGRAMS_V1.pt');(P/'ORTHOGONAL_PRIVATE_NATIVE_V1.json').write_text(json.dumps(dict(plan=plan,records=records,primary=primary['key'],predictions=pred,seconds=time.monotonic()-start,scope='Private directions move in full1152space, shared pairwise dictionaries frozen; exact symmetriccores thencompiled1056products. Opened448diagnostics, nativez/h retained; nofresh/OOD/adoptionclaim.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
