#!/usr/bin/env python3
# BQGATE:496bodyforwards;80fixedprefixes<=179tokens;300seconds;no fitting.
"""pred_a baseline/cache<=1e-4relative; zero/mean replay<=1e-5relative.
pred_b EACHtemplate: component/wholehead effect<=10%relative eachindividual/joint;
>=10/12cue reductions eacharm,joint>=5%nativecontrast; unrelated<=.5regional.
pred_c EACHnaturalhalf: V2cap/control and physical meanabsCE<=.02,maxabs<=.1.
Null: consumer-specific components fail selective native recursive removal.
Price496bodyforwards48regional7arms+32natural5arms;300seconds,no optimization.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
STEM='SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 regional=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows']
 natural=[r for r in json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text())['rows'] if r['pool']=='fineweb']
 assert len(regional)==48 and len(natural)==32 and max(len(r['ids']) for r in regional+natural)<=179
 assert json.loads((P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json').read_text())['pred_a']
 assert json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_RESULT.json').read_text())['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('496recursivebodyforwards80fixedprefixes; frozen physical components');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists()
 tic=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
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
 reg=torch.zeros(48,7,2,dtype=torch.float64);ce=torch.zeros(32,5,dtype=torch.float64);nl_margin=torch.zeros_like(ce)
 arms=[('native',()),('physical',(0,)),('physical',(1,)),('physical',(0,1)),('zero',(0,)),('zero',(1,)),('zero',(0,1))]
 try:
  for i,row in enumerate(regional):
   tokens=torch.tensor([row['ids']],device='cuda')
   for a,(kind,selected) in enumerate(arms):
    logits=forward(tokens,kind,selected);reg[i,a,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,a,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
  for i,row in enumerate(natural):
   tokens=torch.tensor([row['ids']],device='cuda')
   for a,(kind,selected) in enumerate(arms[:4]+[('mean',(0,))]):
    logits=forward(tokens,kind,selected);ce[i,a]=-logits.log_softmax(-1)[198].cpu();nl_margin[i,a]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert body_count==496
 def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 oldreg=torch.load(P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['baseline_margins'][:,:2]
 replay=dict(regional_baseline=relative(reg[:,0],oldreg),newline_baseline=relative(ce[:,0],prior['ce'][16:,0]),newline_mean=relative(ce[:,4],prior['ce'][16:,5]),newline_mean_effect=relative(ce[:,4]-ce[:,0],prior['ce'][16:,5]-prior['ce'][16:,0]),head_subtraction=max(checks))
 rcells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(regional) if r['family']==family];m=reg[ix,:,0];ctrl=reg[ix,:,1];effect=m-m[:,:1];contrast=m[::2]-m[1::2];reduction=contrast[:,:1]-contrast
  cap=float(contrast[:,0].mean())>=.2 and int((contrast[:,0]>0).sum())>=10
  fidelity=[relative(effect[:,a],effect[:,a+3]) for a in (1,2,3)]
  directions=[int((reduction[:,a]>0).sum()) for a in (1,2,3)]
  specificity=[float((ctrl[:,a]-ctrl[:,0]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in (1,2,3)]
  fraction=float(reduction[:,3].mean()/contrast[:,0].mean())
  passed=cap and max(fidelity)<=.1 and min(directions)>=10 and max(specificity)<=.5 and fraction>=.05
  rcells.append(dict(family=family,passed=passed,native_capability=cap,native_mean_contrast=float(contrast[:,0].mean()),physical_vs_wholehead_errors=fidelity,positive_reduction_pairs=directions,joint_native_contrast_fraction=fraction,unrelated_to_regional=specificity,mean_cue_reduction_by_arm=reduction.mean(0).tolist(),physical_nonadditivity=relative(effect[:,1]+effect[:,2],effect[:,3]),wholehead_nonadditivity=relative(effect[:,4]+effect[:,5],effect[:,6])))
 ncells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(natural) if r['family']==family];delta=ce[ix]-ce[ix,:1]
  cap=float(nl_margin[ix,0].mean())>=.2 and int((nl_margin[ix,0]>0).sum())>=12 and float(ce[ix,0].mean())<=5
  pos=float(delta[:,4].mean())>=.02 and int((delta[:,4]>0).sum())>=8
  preserve=all(float(delta[:,a].abs().mean())<=.02 and float(delta[:,a].abs().max())<=.1 for a in (1,2,3))
  ncells.append(dict(family=family,native_capability=cap,positive_control=pos,preservation=preserve,mean_ce_change=delta.mean(0).tolist(),meanabs_ce_change=delta.abs().mean(0).tolist(),maxabs_ce_change=delta.abs().amax(0).tolist()))
 A=max(replay.values())<=1e-4 and replay['head_subtraction']<=1e-5
 result=dict(pred_a=A,pred_b=A and all(c['passed'] for c in rcells),pred_c=A and all(c['native_capability'] and c['positive_control'] and c['preservation'] for c in ncells),checks=replay,regional_cells=rcells,newline_cells=ncells,regional_arms=['native','physical8','physical9','physicaljoint','zero8','zero9','zerojoint'],newline_arms=['native','physical8','physical9','physicaljoint','mean8'],body_forwards=body_count,seconds=time.perf_counter()-tic,scope='Unmerged frozen source components removed at their physical native producer output; subsequent states recompute. Reused regional/FineWeb rows, no new OOD or factor fit.',source_shas=binding)
 torch.save(dict(regional=reg,newline_ce=ce,newline_margin=nl_margin),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
