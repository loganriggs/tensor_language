#!/usr/bin/env python3
# BQGATE:360bodyforwards;480finalreadouts;120prefixes<=248tokens;180seconds.
"""pred_a 24oldanchor margins/states/linearports replay<=1e-4relative andmaxabsmargins<=1e-4.
pred_b all cached states, linear parts and margins finite.
pred_c exactly360bodyforwards and120cachedrows.
24anchors+96freshsyntax prefixes; frozen downstream candidate validation occurs after this cache. No compressed readers installed.
Price360fullforwards+360extraMLPevaluations+480finalreadouts;24anchors+96freshprefixes;180seconds;no fitting.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from joint_attention_mixed_ports_v1 import decompose
from joint_attention_three_group_v1 import execute as three_group
from additive_head_raw_ports_v1 import project, EPS
from three_corner_head_interface_v1 import execute as interface_execute
STEM='MINIMAX_FRESH_CACHE_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(ROOT/k)==v for k,v in files.items());rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];validate(rows);assert len(rows)==120
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('360forwards+360extraMLP+480readouts;composedfinalblockpredictor');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={};parent={};spec=importlib.util.spec_from_file_location('freshhierarchy',P/'extracted_circuits/crossfirst_state_executor_v1/hierarchy.py');hier=importlib.util.module_from_spec(spec);spec.loader.exec_module(hier);weights=hier.executor.load_weights(model.state_dict(),'cuda');ctx={};measures=torch.zeros(120,5,2,dtype=torch.float64);statechecks=[];readouts=torch.zeros(120,4,2,dtype=torch.float64);count=0;W=model.transformer.h[17].attn.c_proj.weight.double().reshape(1152,9,128);headwrites=torch.zeros(120,9,1152,dtype=torch.float64);headinputs=torch.zeros(120,9,128,dtype=torch.float64)
 saved_states=torch.zeros(120,4,1152,dtype=torch.float64);saved_linear=torch.zeros_like(saved_states)
 matrices=[model.state_dict()['transformer.h.17.attn.'+k+'.weight'].reshape(9,128,1152)[2].double() for k in ('c_q','c_k','c_q2','c_k2','c_v')];mix=float(model.transformer.h[17].attn.lamb)
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:
   parts=hier.split_fields(ctx['ids'],ctx['x7'],ctx['x8'],args[0],ctx['R8'],ctx['rho9'],weights);child[i]=parts['child'];parent[i]=parts['parent'];return output
  amplitude=child[i] if arm==1 else parent[i]-child[i] if arm==3 else parent[i]
  return output[0]-(amplitude[...,None]*w).to(output[0].dtype),output[1]
 def input7(module,args):
  if ctx['arm']==0:ctx['x7']=args[0]
 def pre8(module,args):
  if ctx['arm']==0:ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def input8(module,args):
  if ctx['arm']==0:ctx['x8']=args[0]
 def after8(module,args,output):
  if ctx['arm']==0:
   z=ctx['raw8']+output[0];ctx['R8']=z.square().mean(-1).double()+torch.finfo(torch.float32).eps
 def pre9(module,args):
  if ctx['arm']==0:
   z=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho9']=(z.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def pre17(module,args):
  ctx['raw17']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
  if ctx['arm'] in (0,1,3):ctx['raw_inputs'][ctx['arm']]=ctx['raw17'].double()
 def first_values(module,args,output):ctx['first_values']=output.reshape(1,-1,9,128)[:,:,2].double()
 def after_attention17(module,args,output):
  ctx['z17'][ctx['arm']]=(ctx['raw17']+output[0])[:,-1].double()
 def after17(module,args,output):
  assert ctx['arm'] in (0,1,3)
  ctx['h17'][ctx['arm']]=output[0][:,-1].double()
  return output

 handles=[model.transformer.h[17].attn.register_forward_hook(after_attention17),model.transformer.h[0].attn.c_v.register_forward_hook(first_values),model.transformer.h[7].mlp.register_forward_pre_hook(input7),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(input8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].register_forward_hook(after17)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,ids=ids,h16={},h17={},z17={},head_inputs={},ports={},raw_inputs={})
   for arm in (0,1,3):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<120:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
   raw=ctx['raw_inputs'];ordered=[raw[k] for k in (0,1,3)];projections=[project(x,matrices) for x in ordered];norms=[x.square().mean(-1)+EPS for x in ordered];cross=((raw[1]-raw[0])*(raw[3]-raw[0])).mean(-1)
   full=interface_execute(projections,norms,cross,ctx['first_values'],mix)@W[:,2,:].T;small=interface_execute(projections,norms,cross,ctx['first_values'],mix,compact=True)@W[:,2,:].T
   final=ctx['h17'];bar=final[1]+final[3]-final[0];zs=ctx['z17'];zbar=(zs[1]+zs[3]-zs[0]).float();mlp=model.transformer.h[17].mlp
   def g(z):
    z=z.float();return z+mlp(F.rms_norm(z,(1152,)))
   ctx['readout_states']=[bar,g(zbar.double()+full),g(zbar.double()+small),g(zbar)]
   linear_parts=[zs[1]+zs[3]-zs[0],(zbar.double()+full).float(),(zbar.double()+small).float(),zbar]
   for j,state in enumerate(ctx['readout_states']):
    saved_states[i,j]=state[0].double().cpu();saved_linear[i,j]=linear_parts[j][0].double().cpu()
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(state.float(),(1152,)))/30))[0]
    if i<120:readouts[i,j,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();readouts[i,j,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:readouts[i,j,0]=-logits.log_softmax(-1)[198].cpu();readouts[i,j,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==360
 old=torch.load(P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt',weights_only=True);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 replay=[rel(measures[:24,[0,1,3]],old['measures'][:24]),rel(readouts[:24],old['readouts'][:24]),rel(saved_states[:24],old['states'][:24]),rel(saved_linear[:24],old['linear_parts'][:24])]
 maxabs=max(float((measures[:24,[0,1,3]]-old['measures'][:24]).abs().max()),float((readouts[:24]-old['readouts'][:24]).abs().max()))
 finite=all(bool(torch.isfinite(x).all()) for x in (measures,readouts,saved_states,saved_linear))
 result={'pred_a':max(replay)<=1e-4 and maxabs<=1e-4,'pred_b':finite,'pred_c':count==360 and len(rows)==120,'anchor_replay':replay,'anchor_margin_maxabs':maxabs,'body_forwards':count,'extra_final_readouts':480,'extra_mlp_evaluations':360,'seconds':time.perf_counter()-tic,'executed_arms':[0,1,3],'scope':'24historicalanchors+96freshsyntax, three-trajectory conditional compact predictor cache. No corpusOOD, compressedbody or native effect fidelity claim yet.'}

 torch.save(dict(measures=measures[:,[0,1,3]],readouts=readouts,states=saved_states,linear_parts=saved_linear),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
