#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_components pred_c_coefficients pred_d_arithmetic pred_e_initialization
"""Joint graph refitting from inherited and algebraic input directions.
pred_a all runs finite, implicit/full loss and exported coefficients replay<1e-8.
pred_b primary all3 component values<=.15 and<=1.10 original baseline, zJac<=1.10baseline.
pred_c primary both coefficient errors<=1.10original baseline.
pred_d primary same1058124 floats/1047648source mults, at least20%source saving.
pred_e algebraic initialization beats inherited fit objective by>=1% relative error in both geometries.
Null: algebraic proposals confer no benefit over refitting the inherited graph.
Same graph cost, native z/h interface, opened448diagnostics; no fresh or adoption claim.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'PENCIL_JOINT_REFIT_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from joint_orthogonal_private import JointOrthogonalPrivateMetric
 from pairwise_reader_graph import GROUPS
 from pairwise_graph_assessment import Assessment
 from export_orthogonal_private import export
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/'PENCIL_JOINT_REFIT_V1.json').exists()
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);audit=Assessment(data)
 proposal=torch.load(P/'PENCIL_NATIVE_PROPOSAL_STATES_V1.pt',weights_only=True);oldstates=torch.load(P/'ORTHOGONAL_PRIVATE_NATIVE_STATES_V1.pt',weights_only=True);oldgraphs=torch.load(P/'ORTHOGONAL_PRIVATE_NATIVE_PROGRAMS_V1.pt',weights_only=True);records=[];graphs={};savedstates={}
 for geometry in plan['metrics']:
  A,inv=(audit.S,data['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);parent=oldgraphs[geometry+'_0']
  inherited=[A@parent['input_bases'][str(j)] for j in range(3)]+oldstates[geometry+'_0']
  for init in ('inherited','algebraic'):
   initial=inherited if init=='inherited' else proposal[geometry];params=[torch.nn.Parameter(torch.linalg.qr(v,mode='reduced').Q.contiguous().cuda()) for v in initial]
   metric=JointOrthogonalPrivateMetric((A@audit.Q@A).cuda(),[torch.cat([params[a],params[b]],1).detach() for a,b in GROUPS]);first=float(metric.loss(params)[0].detach());assert abs(first-plan['initial_objectives'][geometry][init])<1e-8
   best=first;saved=[p.detach().clone() for p in params];evaluations=0;best_eval=0;history=[];key=geometry+'_'+init;failure=None
   opt=torch.optim.LBFGS(params,lr=1.,max_iter=200,max_eval=250,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
   def closure():
    nonlocal best,saved,evaluations,best_eval
    opt.zero_grad();loss=metric.loss(params)[0];value=float(loss.detach());assert math.isfinite(value)
    if value<best:best=value;saved=[p.detach().clone() for p in params];best_eval=evaluations
    if evaluations%50==0:history.append(dict(evaluation=evaluations,loss=value));print(key,evaluations,value,flush=True)
    evaluations+=1;scaled=loss/first;scaled.backward()
    for parameter in params:
     assert torch.isfinite(parameter.grad).all();parameter.grad=parameter.grad.contiguous()
    return scaled
   try:opt.step(closure)
   except (ValueError,RuntimeError,AssertionError) as error:failure=str(error)
   savedstates[key]=[p.cpu().clone() for p in saved];row=dict(key=key,geometry=geometry,initialization=init,initial_objective=first,objective=best,evaluations=evaluations,best_evaluation=best_eval,history=history,optimization_failure=failure)
   try:
    with torch.no_grad():
     loss,components=metric.loss(saved);dense=metric.loss(saved,dense=True)[0];replay=abs(float(loss-dense));assert replay<1e-8
     template=dict(parent,input_bases={str(j):inv@saved[j].cpu() for j in range(3)});graph,diag=export(components,metric.scales,template,A,inv);graph=audit.correct(graph);scores=audit.assess(graph);field='covariance_error' if geometry=='calibration_shaped' else 'native_error';assert abs(scores[field]-float(dense.sqrt()))<1e-8
     assert scores['stored_floats']==scores['physical_storage_floats']==1058124 and scores['source_total_multiplications']==1047648
     graphs[key]=graph;row.update(instrument=failure is None,dense_replay=replay,compiler=diag,**scores)
   except (ValueError,RuntimeError,AssertionError) as error:row.update(instrument=False,export_failure=str(error))
   records.append(row);print('RECORD',json.dumps({k:v for k,v in row.items() if k not in ('compiler','history')}),flush=True)
 primary=min((r for r in records if r['geometry']=='calibration_shaped'),key=lambda r:r['objective']);valid=primary['instrument'];baseline=plan['baseline'];bykey={r['key']:r for r in records}
 pred=dict(pred_a_instrument=all(r['instrument'] for r in records),pred_b_components=valid and all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],baseline['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],baseline['euclidean_jacobian_errors'])),pred_c_coefficients=valid and all(primary[k]<=1.1*baseline[k] for k in ('native_error','covariance_error')),pred_d_arithmetic=valid and primary['source_total_multiplications']<=.8*baseline['source_total_multiplications'],pred_e_initialization=all(bykey[g+'_algebraic']['instrument'] and bykey[g+'_algebraic']['objective']<=.9801*bykey[g+'_inherited']['objective'] for g in plan['metrics']))
 torch.save(savedstates,P/'PENCIL_JOINT_REFIT_STATES_V1.pt');torch.save(graphs,P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt');(P/'PENCIL_JOINT_REFIT_V1.json').write_text(json.dumps(dict(plan=plan,records=records,primary=primary['key'],predictions=pred,seconds=time.monotonic()-start),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
