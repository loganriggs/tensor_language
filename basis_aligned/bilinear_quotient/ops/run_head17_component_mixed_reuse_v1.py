#!/usr/bin/env python3
# BQGATE:600bodyforwards;720finalreadouts;120prefixes<=248tokens;180seconds.
"""pred_a previous120native five-arm/readout anchors<=1e-4 and fullportstateerror<=.02.
pred_b old shared component mixed effect predictsfullhead effect<=.20 eachfournewpanelgroups.
pred_c projected-reader effect predicts exactoldcomponent effect<=.10 eachfourgroups.
Null: same head doesnot imply same circuit; old component maynot carry mixedresponse.
Price600fullforwards+720finalreadouts,120 reusedprefixes<=24tokens,180seconds,no fitting.
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
from shared_head_native_ports_v1 import execute as port_execute
from folded_normalized_router_v1 import rotary
STEM='HEAD17_COMPONENT_MIXED_REUSE_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24]+json.loads((P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==120
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('600forwards+720readouts;120reusedrows;oldcomponentmixedreuse');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={};parent={};spec=importlib.util.spec_from_file_location('freshhierarchy',P/'extracted_circuits/crossfirst_state_executor_v1/hierarchy.py');hier=importlib.util.module_from_spec(spec);spec.loader.exec_module(hier);weights=hier.executor.load_weights(model.state_dict(),'cuda');ctx={};measures=torch.zeros(120,5,2,dtype=torch.float64);statechecks=[];readouts=torch.zeros(120,6,2,dtype=torch.float64);count=0;W=model.transformer.h[17].attn.c_proj.weight.double().reshape(1152,9,128);headwrites=torch.zeros(120,9,1152,dtype=torch.float64);headinputs=torch.zeros(120,9,128,dtype=torch.float64)
 oldprogram={k:v.cuda() for k,v in torch.load(P/'COMPILED_SHARED_HEAD2_V1_ARTIFACT.pt',weights_only=True).items()};interface={k:v.cuda() for k,v in torch.load(P/'HEAD17_SOURCE_INTERFACE_V1_PROGRAM.pt',weights_only=True).items()}
 native_attention=model.transformer.h[17].attn.squared_attention;allports=[]
 def capture_attention(q,k,v,q2,k2):
  scores=torch.einsum('bthd,bshd->bhts',q,k)/q.shape[-1];scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/q.shape[-1]
  ctx['ports'][ctx['arm']]=(scores[:,2,-1].double(),scores2[:,2,-1].double(),v[:,:,2].double())
  return native_attention(q,k,v,q2,k2)
 model.transformer.h[17].attn.squared_attention=capture_attention
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
 def after16(module,args,output):
  arm=ctx['arm']
  if arm<4:ctx['h16'][arm]=output[0].double()
  if arm==4:
   s=ctx['h16'];return (s[1]+s[3]-s[0]).to(output[0].dtype),output[1]
  return output
 def pre17(module,args):ctx['raw17']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def first0(module,args):ctx['first']=args[0].double()
 def att17(module,args,output):
  if ctx['arm']<5:ctx['z17'][ctx['arm']]=(ctx['raw17']+output[0]).double()
  if ctx['arm'] in (0,1,3,4):
   current=args[0][0].double();length=len(current);source=torch.cat((current,ctx['first'][0]),-1);query=current[-1:].expand(length,-1)
   qa=query@oldprogram['q1'].T;qb=query@oldprogram['q2'].T;ka,kb,vv=(source@interface['stack'].T).split(128,-1)
   rotations=torch.stack([rotary(length-1,128).T@rotary(pos,128) for pos in range(length)]).cuda()
   exact=port_execute(qa,qb,ka,kb,vv,rotations,oldprogram,interface['coefficients'],source@interface['residual'].T).sum(0,keepdim=True)
   projected=port_execute(qa,qb,ka,kb,vv,rotations,oldprogram,interface['coefficients']).sum(0,keepdim=True)
   ctx['old_writes'][ctx['arm']]=(exact,projected)
 def before_proj(module,args):
  if ctx['arm']<5:ctx['head_inputs'][ctx['arm']]=args[0][0,-1].double().reshape(9,128)
 def after17(module,args,output):
  arm=ctx['arm'];x=output[0]
  if arm<4:ctx['h17'][arm]=x.double();return output
  z=ctx['z17'];s=ctx['h17'];bar=(s[1]+s[3]-s[0])[:,-1];zbar=(z[1]+z[3]-z[0]).to(x.dtype);v=(z[4]-zbar.double())[:,-1]
  y=ctx['head_inputs'];delta=y[4]-y[1]-y[3]+y[0];ref=(delta[2]@W[:,2,:].T)[None];ports=ctx['ports'];parts=decompose(ports[0],ports[1],ports[3],ports[4]);cross=parts['cross']@W[:,2,:].T;defect=parts['defect']@W[:,2,:].T;full=cross+defect
  statechecks.append(dict(error_sq=float((full-ref).square().sum()),reference_sq=float(ref.square().sum())))
  small=three_group(ports[0],ports[1],ports[3],ports[4])@W[:,2,:].T;old=ctx['old_writes'];mixed=[old[4][j]-old[1][j]-old[3][j]+old[0][j] for j in range(2)];ctx['readout_states']=[bar,bar+ref,bar+full,bar+small,bar+mixed[0],bar+mixed[1]];allports.append({key:tuple(v.cpu() for v in ports[key]) for key in (0,1,3,4)});return output

 handles=[model.transformer.h[0].attn.register_forward_pre_hook(first0),model.transformer.h[7].mlp.register_forward_pre_hook(input7),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(input8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[17].attn.c_proj.register_forward_pre_hook(before_proj),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[16].register_forward_hook(after16),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.register_forward_hook(att17),model.transformer.h[17].register_forward_hook(after17)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,ids=ids,h16={},h17={},z17={},head_inputs={},ports={},old_writes={})
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<120:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
   for j,state in enumerate(ctx['readout_states']):
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(state.float(),(1152,)))/30))[0]
    if i<120:readouts[i,j,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();readouts[i,j,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:readouts[i,j,0]=-logits.log_softmax(-1)[198].cpu();readouts[i,j,1]=(logits[198]-logits[11]).cpu()
 finally:
  model.transformer.h[17].attn.squared_attention=native_attention
  for h in handles:h.remove()
 assert count==600 and len(statechecks)==120
 old=torch.load(P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ARTIFACT.pt',weights_only=True);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measures,old['measures']),rel(readouts[:,:4],old['readouts'])];local=[];cells=[]
 for group in range(5):
  lo,hi=24*group,24*(group+1);stats=statechecks[lo:hi];local.append((sum(x['error_sq'] for x in stats)/max(sum(x['reference_sq'] for x in stats),1e-30))**.5);z=readouts[lo:hi,:,0];ref=z[:,2]-z[:,0];pred=z[:,3]-z[:,0];native=measures[lo:hi,0,0];contrasts=native[::2]-native[1::2]
  cells.append(dict(old_component_error=rel(z[:,4]-z[:,0],z[:,1]-z[:,0]),projection_effect_error=rel(z[:,5]-z[:,0],z[:,4]-z[:,0]),old_component_norm_ratio=float((z[:,4]-z[:,0]).norm()/(z[:,1]-z[:,0]).norm()),maxabs_old_component_effect=float((z[:,4]-z[:,0]).abs().max()),group=group,fresh=group>0,effect_error=rel(pred,ref),same_sign=int(((pred*ref)>0).sum()),reference_zeros=int((ref==0).sum()),native_positive_pairs=int((contrasts>0).sum()),native_mean_contrast=float(contrasts.mean()),formula_native_effect_error=rel(ref,z[:,1]-z[:,0]),maxabs_prediction_error=float((pred-ref).abs().max())))
 result={'pred_a':max(replay)<=1e-4 and max(local)<=.02,'pred_b':all(c['old_component_error']<=.2 for c in cells[1:]),'pred_c':all(c['projection_effect_error']<=.1 for c in cells[1:]),'anchor_replay':replay,'joint_port_state_error':local,'cells':cells,'body_forwards':count,'extra_final_readouts':720,'seconds':time.perf_counter()-tic,'scope':'Known old source component mixed response versus fullhead17.2 mixed response, plus projected-reader variant.120 reused regional prefixes, no new OOD or fitting; four native corners and final additive background retained.'}
 torch.save(dict(measures=measures,readouts=readouts,ports=allports,local_stats=statechecks,child={i:x.cpu() for i,x in child.items()},parent={i:x.cpu() for i,x in parent.items()}),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
