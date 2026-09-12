#!/usr/bin/env python3
# BQGATE:30bodyforwards;6countryvariants88tokens;120seconds;no fitting.
"""pred_a Australia CE/margin replay<=1e-4relative, one-token row control.
pred_b countryrange jointCEdamage>=.5originalAustraliadamage.
pred_c EACHvariant newline-margin>=.2,CE<=5,meanheadCEdamage>=.02.
Null: postselected collateral does not depend strongly on this country token.
Price30bodyforwards6rows5arms;120sec;no refit or OOD claim.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
STEM='SCALAR_PRODUCERS_NEWLINE_COUNTRY_AUDIT_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 natural=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];assert len(natural)==6
 assert natural[0]['country']=='Australia' and all(len(r['ids'])==88 for r in natural)
 assert all(all(x==y for i,(x,y) in enumerate(zip(natural[0]['ids'],r['ids'])) if i!=r['changed_position']) for r in natural)
 assert json.loads((P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json').read_text())['pred_a']
 assert json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_RESULT.json').read_text())['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('30bodyforwards6fixedcountryvariants; original failure retained');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists()
 tic=time.perf_counter();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 prior=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True,map_location='cpu')
 mean_head=prior['mean_head'].cuda();context={};checks=[];body_count=0
 def capture(index,args):context[('preov',index)]=args[0]
 def hook(index,head,module,args,output):
  arm=context['arm'];selected=index in context['selected']
  if not selected:return output
  if arm=='physical':
   scalar=head_scalar(args[0],context['tokens'],producer,index)
   removed=(scalar[...,None]*writers[index]).to(output[0].dtype)
   return output[0]-removed,output[1]
  z=context[('preov',index)].reshape(*args[0].shape[:2],9,128)
  O=module.c_proj.weight[:,128*head:128*(head+1)]
  changed=output[0]-F.linear(z[:,:,head],O);edited=z.clone();edited[:,:,head]=0
  if arm=='mean':changed=changed+F.linear(mean_head,O)[None,None,:];edited[:,:,head]=mean_head
  direct=module.c_proj(edited.reshape_as(args[0]));checks.append(float((changed-direct).norm()/direct.norm()))
  return changed,output[1]
 handles=[]
 for index,layer,head in ((0,8,2),(1,9,8)):
  attn=model.transformer.h[layer].attn
  handles.append(attn.c_proj.register_forward_pre_hook(lambda m,a,i=index:capture(i,a)))
  handles.append(attn.register_forward_hook(lambda m,a,o,i=index,h=head:hook(i,h,m,a,o)))
 def forward(tokens,arm,selected):
  nonlocal body_count
  context.update(tokens=tokens,arm=arm,selected=selected)
  x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  body_count+=1
  return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 ce=torch.zeros(6,5,dtype=torch.float64);nl_margin=torch.zeros_like(ce)
 arms=[('native',()),('physical',(0,)),('physical',(1,)),('physical',(0,1)),('zero',(0,)),('zero',(1,)),('zero',(0,1))]
 try:
  for i,row in enumerate(natural):
   tokens=torch.tensor([row['ids']],device='cuda')
   for a,(kind,selected) in enumerate(arms[:4]+[('mean',(0,))]):
    logits=forward(tokens,kind,selected);ce[i,a]=-logits.log_softmax(-1)[198].cpu();nl_margin[i,a]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert body_count==30
 original=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 replay=dict(original_ce=rel(ce[0],original['newline_ce'][30]),original_margin=rel(nl_margin[0],original['newline_margin'][30]),head_subtraction=max(checks))
 delta=ce-ce[:,:1];range_change=float(delta[:,3].max()-delta[:,3].min());original_damage=float(delta[0,3])
 cells=[dict(country=r['country'],native_ce=float(ce[i,0]),native_margin=float(nl_margin[i,0]),ce_changes=delta[i].tolist(),native_capability=float(nl_margin[i,0])>=.2 and float(ce[i,0])<=5,mean_head_control=float(delta[i,4])>=.02) for i,r in enumerate(natural)]
 A=max(replay.values())<=1e-4 and replay['head_subtraction']<=1e-5
 result={'pred_a':A,'pred_b':A and original_damage>0 and range_change>=.5*original_damage,'pred_c':A and all(c['native_capability'] and c['mean_head_control'] for c in cells),'checks':replay,'cells':cells,'joint_country_effect_range':range_change,'range_to_original_damage':range_change/original_damage,'arms':['native','physical8','physical9','physicaljoint','mean8'],'body_forwards':body_count,'seconds':time.perf_counter()-tic,'scope':'Postselected failure audit with one-token country substitutions; original failure retained. No new natural observations, broadOOD, factor fit, or redefinition of preservation.','source_shas':binding}
 torch.save(dict(newline_ce=ce,newline_margin=nl_margin),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
