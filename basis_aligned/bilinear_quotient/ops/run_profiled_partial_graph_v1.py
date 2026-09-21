#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_simplicity
"""Continuous refit after graph edit, keeping component3 private.
Four4000step profiled Adamfits rates.005/.02 x pruned/perturbed,cosine.
pred_a_instrument profiled loss matches dense+ridge<1e-8 andfiniteoutputs.
pred_b_fidelity eachofthree nativeopenederrors<=.15 and<=1.10baseline.
pred_c_simplicity <=1.01*897804floats and512products versus768.
Null: newgraphcannot meetfidelityatmatchedstorageevenafterrefitting.
Zero native forwards; nativeinputdependencies remain.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'PROFILED_PARTIAL_GRAPH_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from profiled_mixed_products import profiled_loss
 from shared_quadratic_products import materialize_mixed
 from shared_mixed_source_graph import component_scalars
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();output=P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json';assert not output.exists()
 paths=[P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',P/'PARTIAL_MIXED_GRAPH_V1.pt']
 for path,key in zip(paths,['input_sha256','graph_sha256']):assert hashlib.sha256(path.read_bytes()).hexdigest()==plan[key]
 data=torch.load(paths[0],weights_only=True);original=torch.load(paths[1],weights_only=True);T=data['teacher'][:4].cuda();metric_root=torch.linalg.inv(data['inverse_root']);initL=metric_root@original['shared_mixed']['left_reader'];initR=metric_root@original['shared_mixed']['right_reader'];records=[];programs={}
 for rate in plan['learning_rates']:
  for kind in plan['starts']:
   torch.manual_seed(5147);L=initL.cuda().clone();R=initR.cuda().clone()
   if kind!='pruned':L+=.01*L.std()*torch.randn_like(L);R+=.01*R.std()*torch.randn_like(R)
   L=L/L.norm(dim=0);R=R/R.norm(dim=0);L.requires_grad_();R.requires_grad_();opt=torch.optim.Adam([L,R],lr=rate);best=float('inf');history=[]
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,W=profiled_loss(T,L,R,plan['ridge']);value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;beststate=(L.detach().clone(),R.detach().clone(),W.detach().clone());beststep=step
    if step%400==0:history.append(dict(step=step,objective=value));print(rate,kind,step,value,flush=True)
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=rate*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
    with torch.no_grad():L.div_(L.norm(dim=0));R.div_(R.norm(dim=0))
   with torch.no_grad():
    l,r,w=[x.cpu() for x in beststate];hat=materialize_mixed(l,r,w);coef=float((hat-data['teacher'][:4]).norm()/data['teacher'][:4].norm());explicit=coef**2+plan['ridge']*float(w.square().sum()/data['teacher'][:4].square().sum())
    left=data['inverse_root']@l;right=data['inverse_root']@r;readout=w.T*data['scales'][None,:4];raw=torch.einsum('ir,ro,jr->oij',left,readout,right);Qs=.5*(raw+raw.transpose(-1,-2));linears=[];biases=[]
    for j,pair in enumerate(data['pairs'][:2]):
     for k,Q in enumerate(pair['Qs']):
      pos=2*j+k;linear=2*Q@data['mu'];constant=data['mu']@Q@data['mu']+torch.trace(data['old_covariance']@Q)
      linears.append(linear-2*Qs[pos]@data['mu']);biases.append(constant-data['mu']@linear+data['mu']@Qs[pos]@data['mu']-torch.trace(data['old_covariance']@Qs[pos]))
    shared={**original['shared_mixed'],**dict(left_reader=left,right_reader=right,product_weights=readout,source_linear=torch.stack(linears,1),source_bias=torch.stack(biases))}
    program=dict(shared_mixed=shared,private_pair=original['private_pair'],residual_writer=original['residual_writer']);phi=component_scalars(data['z'],data['h'],program);errors=[]
    for j,pair in enumerate(data['pairs']):
     truth=pair['truth'][data['indices']];errors.append(float((phi[data['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
    key=f'{rate}_{kind}';floats=program['residual_writer'].numel()+sum(v.numel() for pr in [shared,program['private_pair']] for v in pr.values() if v.is_floating_point());programs[key]=program
    records.append(dict(key=key,selection_objective=best,coefficient_error_first_two=coef,dense_objective_replay=abs(explicit-best),best_step=beststep,per_mode_errors=errors,stored_float_scalars=floats,source_products=512,history=history))
 winner=min(records,key=lambda r:r['selection_objective']);base=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text());ratios=[a/b for a,b in zip(winner['per_mode_errors'],base['baseline_errors'])]
 out=dict(plan=plan,records=records,winner=winner['key'],ratios_to_baseline=ratios,predictions=dict(pred_a_instrument=max(r['dense_objective_replay'] for r in records)<1e-8,pred_b_fidelity=max(winner['per_mode_errors'])<=.15 and max(ratios)<=1.1,pred_c_simplicity=winner['stored_float_scalars']<=1.01*897804 and winner['source_products']==512),seconds=time.perf_counter()-start)
 torch.save(programs,P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt');output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['predictions']),flush=True)
if __name__=='__main__':main()
