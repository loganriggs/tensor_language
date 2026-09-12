#!/usr/bin/env python3
# BQGATE:432bodyforwards+72eightblockprefixes;72reusedprefixes;donors;120seconds.
"""pred_a full directional map interaction replay<=1e-4 and capable families.
pred_b rank64 signed interaction error<=5% in each family.
pred_c rank64 joint removal effect error<=1% in each family.
Null: compact scalar prediction does not survive the native suffix.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from compiled_directional_response_v1 import predict
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='DIRECTIONAL_INTERACTION_DONOR_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'DIRECTIONAL_INTERACTION_LOGIT_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==72
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('432bodyforwards+72eightblockprefixes;72reusedprefixes;donors;120seconds');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('interaction_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};low={k:v.cuda() for k,v in torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True).items()};bridge=torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True);matrix=bridge['mixed_map'].cuda();direction=bridge['direction'].cuda();gain=model.transformer.h[9].lambdas[0].double();context={};margins=torch.zeros(72,6,2,dtype=torch.float64);count=0;field_checks=[]
 donor_fields=[]
 for row in rows:
  ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h[:8]:x,v1=block(x,v1,x0)
  block=model.transformer.h[8];current=F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,));donor_fields.append(runtime.scalar(current,ids,p,0))
 def before8(module,args):
  if context['arm']==0:context['r8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def head8(module,args,output):
  s=runtime.scalar(args[0],context['tokens'],p,0)-context['donor']
  if context['arm']==0:context['z']=context['r8']+output[0];context['a']=s;return output
  return output[0]-(s[...,None]*p['writers'][0]).to(output[0].dtype),output[1]
 def mlp8(module,args,output):
  if context['arm']==0:context['u0']=output-module.Down_bias
 def before9(module,args):
  if context['arm']==0:context['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def head9(module,args,output):
  arm=context['arm']
  if arm==0:
   context['frozen']=runtime.scalar(args[0],context['tokens'],p,1)
   for name,kw in [('rank64',dict(left=low['left'],right=low['right'])),('full',dict(matrix=matrix))]:
    raw=predict(context['z'],context['raw9'],context['u0'],context['a'],direction,gain,**kw);current=F.rms_norm(raw.float(),(1152,));context[name]=runtime.scalar(current,context['tokens'],p,1)
   return output
  if arm==1:return output
  if arm==2:
   s=runtime.scalar(args[0],context['tokens'],p,1);field_checks.append(float((s-context['full']).norm()/s.norm().clamp_min(1e-30)))
  else:s=context[{3:'rank64',4:'frozen',5:'full'}[arm]]
  return output[0]-(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].register_forward_pre_hook(before8),model.transformer.h[8].attn.register_forward_hook(head8),model.transformer.h[8].mlp.register_forward_hook(mlp8),model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');context.clear();context['tokens']=ids;context['donor']=donor_fields[row['donor_id']]
   for arm in range(6):
    context['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;margins[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();margins[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for handle in handles:handle.remove()
 assert count==432
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));records=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];m=margins[ix];actual=m[:,2,0]-m[:,4,0];approx=m[:,3,0]-m[:,4,0];full=m[:,5,0]-m[:,4,0];effect=m[:,2,0]-m[:,0,0];approx_effect=m[:,3,0]-m[:,0,0];paired=m[::2,0,0]-m[1::2,0,0];removed=m[::2,2,0]-m[1::2,2,0];capable=float(paired.mean())>=.2 and int((paired>0).sum())>=10;anchor=float(actual.norm())>1e-6 and rel(full,actual)<=1e-4
  records.append(dict(family=family,name=rows[ix[0]]['family_name'],native_paired_contrast=float(paired.mean()),positive_native_pairs=int((paired>0).sum()),capable=capable,actual_interaction_norm=float(actual.norm()),full_interaction_error=rel(full,actual),rank64_interaction_error=rel(approx,actual),rank64_joint_effect_error=rel(approx_effect,effect),actual_joint_coverage=float((paired-removed).mean()/paired.mean()),control_interaction_error=rel(m[:,3,1]-m[:,4,1],m[:,2,1]-m[:,4,1]),anchor_pass=anchor,interaction_pass=rel(approx,actual)<=.05,joint_effect_pass=rel(approx_effect,effect)<=.01))
 old=torch.load(P/'DIRECTIONAL_INTERACTION_LOGIT_V1_ARTIFACT.pt',weights_only=True)['margins'][:,0];baseline=rel(margins[:,0],old)
 A=baseline<=1e-6 and all(r['capable'] and r['anchor_pass'] for r in records);B=A and all(r['interaction_pass'] for r in records);C=B and all(r['joint_effect_pass'] for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'records':records,'max_full_scalar_relative_error':max(field_checks),'native_baseline_error':baseline,'prefix_forwards_8blocks':72,'body_forwards':count,'seconds':time.perf_counter()-tic,'arms':['native','swap8','swap8_actual9','swap8_predicted64_9','swap8_frozen9','swap8_predicted_full9'],'scope':'Frozen rank64 on reused longer contexts; paired-native head8 donor replacement induces nonuniform tokenwise edits. All six arms retain actual recipient background; head9 is removed, not donor-swapped. No new text OOD. Pristine z8, bias-free MLP8 output, raw9 and head8 amplitude are inputs. Conditional surrogate-removal simulator with native prefix/background/suffix, not autonomous circuit extraction or corpus OOD.'}
 torch.save(dict(margins=margins),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
