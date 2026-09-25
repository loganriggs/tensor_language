#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_components pred_c_metrics pred_d_arithmetic
"""Learn overlapping shared/private readers at fixed996876float/986496mult budget.
Registered optimizer selected on planted toys before native execution.
Two geometries, inherited and fully random starts,1800cosine steps.
pred_a native export/dense metric replay<1e-8, finite gradients, literal price exact.
pred_b primary all3component values<=.15 and<=1.10pairbaseline; zJacobian<=1.10.
pred_c primary both coefficient errors<=1.10pairbaseline.
pred_d source mults<=.80pairbaseline and floats lower.
Null: even joint input fitting fails faithful reuse at this structure/cost.
No native forwards/new inputs. Supplied z/h and opened448limitations remain.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main(prefix='JOINT_OVERLAP'):
 plan=json.loads((P/f'{prefix}_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from overlap_varpro import OverlapMetric,parameters_from_program
 from overlap_export import export_overlap,assess_overlap
 from pack_reader_graph_artifacts import counts
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/f'{prefix}_V1.json').exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);templates_cpu=torch.load(P/plan.get('input_programs','LOCAL_SHARED_READER_PROGRAMS_V1.pt'),weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);records=[];programs={}
 for geometry in plan['metrics']:
  transform,inverse=(S,d['inverse_root']) if geometry=='calibration_shaped' else (I,I);template=templates_cpu[geometry+'_128_160'];targets=(transform@Q@transform).cuda();templates=[{k:template['pairs'][str(j)][k].cuda() for k in ('shared_indices','private_indices','product_indices')} for j in range(3)];metric=OverlapMetric(targets,templates)
  for seed in plan['seeds']:
   params=parameters_from_program(template,transform)
   if seed:
    g=torch.Generator().manual_seed(seed);params=[torch.randn(p.shape,dtype=p.dtype,generator=g)/p.shape[0]**.5 for p in params]
   params=[p.cuda().detach().requires_grad_() for p in params];opt=torch.optim.Adam(params,lr=plan['rate']) if plan['optimizer']=='adam' else torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,_,_=metric.loss(params);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;state=[p.detach().clone() for p in params];beststep=step
    if step%300==0:history.append(dict(step=step,objective=value));print(geometry,seed,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step()
   with torch.no_grad():
    loss,W,_=metric.loss(state);explicit=metric.loss(state,dense=True)[0];replay=abs(float(loss-explicit));assert replay<1e-8
    program=export_overlap(state,W,template,inverse.cuda(),d);scores=assess_overlap(program,d);storage=counts(program);assert storage['backing_storage_floats']==scores['stored_floats']==996876 and scores['source_total_multiplications']==986496
    key=f'{geometry}_{seed}';programs[key]=program;row=dict(key=key,geometry=geometry,seed=seed,objective=best,best_step=beststep,dense_loss_replay=replay,history=history,**scores);records.append(row);print('RECORD',json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
 winners={name:min((r for r in records if r['geometry']==name),key=lambda r:r['objective'])['key'] for name in plan['metrics']};primary=next(r for r in records if r['key']==winners['calibration_shaped']);base=plan['baseline']
 pred=dict(pred_a_instrument=all(max(r['execution_replay'],r['dense_loss_replay'])<1e-8 for r in records),pred_b_components=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_metrics=all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')),pred_d_arithmetic=primary['source_total_multiplications']<=.8*base['source_total_multiplications'] and primary['stored_floats']<base['stored_floats'])
 torch.save(programs,P/f'{prefix}_PROGRAMS_V1.pt');(P/f'{prefix}_V1.json').write_text(json.dumps(dict(plan=plan,records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
