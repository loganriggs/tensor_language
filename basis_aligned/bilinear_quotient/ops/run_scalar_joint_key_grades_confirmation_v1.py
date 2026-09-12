#!/usr/bin/env python3
# BQGATE:632bodyforwards;104prefixes<=247tokens;300seconds;no fitting.
"""pred_a native capability, fullsum/fullsector<=1e-5relative, selfdonor<=1e-4.
pred_b EACH3regionalfamilies: removal>=.5coverage,>=10/12positive;
donor>=.5transfer,>=20/24positive; bothunrelated<=.5.
pred_c BOTHnaturalhalves meanabsCE<=.02,maxabs<=.1 withcap/meanheadcontrol.
Null: selected reused-panel grade mask45 fails prospective confirmation.
Price632bodyforwards72regional7arms+32natural4arms,300seconds;no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
from scalar_value_sectors_v1 import head_scalar_sectors
from scalar_joint_key_paths_v1 import paths
STEM='SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1'
SELECTED=[0,4,5,6,7,8,9]

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/(STEM+'_SELECTION.json')).read_text())['selected']['mask']==45
 panel=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text());regional=panel['regional'];natural=panel['natural'];assert len(regional)==72 and len(natural)==32
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('632bodyforwards72regional7arms32natural4arms; frozenmask45prospectiveconfirmation');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'].cuda();writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda();mean_head=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True,map_location='cpu')['mean_head'].cuda()
 context={};native={};checks=[];meanchecks=[];count=0
 def hook(index,head,module,args,output):
  arm=context['arm'];i=context['row'];natural_mode=context['pool']=='natural'
  if natural_mode and arm==3:
   if index==1:return output
   z=context['preov'].reshape(*args[0].shape[:2],9,128);O=module.c_proj.weight[:,128*head:128*(head+1)];changed=output[0]-F.linear(z[:,:,head],O)+F.linear(mean_head,O)[None,None];edited=z.clone();edited[:,:,head]=mean_head;direct=module.c_proj(edited.reshape_as(args[0]));meanchecks.append(float((changed-direct).norm()/direct.norm()));return changed,output[1]
  pieces=paths(args[0],context['tokens'],p,index,bands[index]);selected=pieces[...,SELECTED,1 if index==0 else 0].sum(-1)
  if arm==0:
   truth=head_scalar(args[0],context['tokens'],p,index);checks.append(float((pieces.sum((-1,-2))-truth).norm()/truth.norm().clamp_min(1e-30)));native[i,index]=selected
   return output
  fullsector=(natural_mode and arm==2) or (not natural_mode and arm==4)
  if fullsector:
   truth=head_scalar_sectors(args[0],context['tokens'],p,index)[...,1 if index==0 else 0];checks.append(float((pieces[...,1 if index==0 else 0].sum(-1)-truth).norm()/truth.norm().clamp_min(1e-30)));delta=truth
  elif not natural_mode and arm in (5,6):delta=selected-native[(i^1 if arm==5 else i),index]
  elif natural_mode or arm==3 or arm==index+1:delta=selected
  else:return output
  return output[0]-(delta[...,None]*writers[index]).to(output[0].dtype),output[1]
 handles=[]
 for index,layer,head in ((0,8,2),(1,9,8)):
  attn=model.transformer.h[layer].attn
  if index==0:handles.append(attn.c_proj.register_forward_pre_hook(lambda m,a:context.update(preov=a[0])))
  handles.append(attn.register_forward_hook(lambda m,a,o,i=index,h=head:hook(i,h,m,a,o)))
 def forward(row,i,arm,pool):
  nonlocal count
  tokens=torch.tensor([row['ids']],device='cuda');context.update(tokens=tokens,row=i,arm=arm,pool=pool);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 reg=torch.zeros(72,7,2,dtype=torch.float64);ce=torch.zeros(32,4,dtype=torch.float64);margin=torch.zeros_like(ce)
 def score_reg(i,a):
  row=regional[i];logits=forward(row,i,a,'regional');reg[i,a,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,a,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 try:
  for pair in range(0,72,2):
   native.clear()
   for i in (pair,pair+1):score_reg(i,0)
   for i in (pair,pair+1):
    for a in range(1,7):score_reg(i,a)
  native.clear()
  for i,row in enumerate(natural):
   for a in range(4):
    logits=forward(row,i,a,'natural');ce[i,a]=-logits.log_softmax(-1)[198].cpu();margin[i,a]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==632
 cells=[]
 for family in range(3):
  ix=[i for i,r in enumerate(regional) if r['family']==family];m=reg[ix,:,0];contrast=m[::2,0]-m[1::2,0];effect=m-m[:,:1];reduction=effect[1::2,3]-effect[::2,3];sign=torch.tensor([-1 if regional[i]['cue']=='British' else 1 for i in ix],dtype=torch.float64);directed=effect[:,5]*sign;coverage=float(reduction.mean()/contrast.mean());transfer=float(directed.mean()/contrast.mean());ratios=[float((reg[ix,a,1]-reg[ix,0,1]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in (3,5)];selfdiff=reg[ix,6]-reg[ix,0];selfnorm=float(reg[ix,0].norm());selferr=float(selfdiff.norm()/max(selfnorm,1e-30));selfok=selferr<=1e-4 if selfnorm>1e-8 else float(selfdiff.abs().max())<=1e-6
  cells.append(dict(family=family,native_mean_contrast=float(contrast.mean()),native_positive_pairs=int((contrast>0).sum()),native_capability=float(contrast.mean())>=.2 and int((contrast>0).sum())>=10,coverage=coverage,positive_removal_pairs=int((reduction>0).sum()),donor_transfer=transfer,positive_donor_rows=int((directed>0).sum()),unrelated_ratios=ratios,self_error=selferr,self_maxabs=float(selfdiff.abs().max()),self_pass=selfok,passed=coverage>=.5 and int((reduction>0).sum())>=10 and transfer>=.5 and int((directed>0).sum())>=20 and max(ratios)<=.5,mean_arm_effects=effect.mean(0).tolist()))
 ncells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(natural) if r['family']==family];delta=ce[ix,1]-ce[ix,0];control=ce[ix,3]-ce[ix,0];ncells.append(dict(family=family,native_mean_ce=float(ce[ix,0].mean()),native_mean_margin=float(margin[ix,0].mean()),native_positive=int((margin[ix,0]>0).sum()),capability=float(ce[ix,0].mean())<=5 and float(margin[ix,0].mean())>=.2 and int((margin[ix,0]>0).sum())>=12,control_mean=float(control.mean()),control_positive=int((control>0).sum()),control_pass=float(control.mean())>=.02 and int((control>0).sum())>=8,meanabs_change=float(delta.abs().mean()),maxabs_change=float(delta.abs().max()),passed=float(delta.abs().mean())<=.02 and float(delta.abs().max())<=.1,fullsector_maxabs=float((ce[ix,2]-ce[ix,0]).abs().max())))
 instrument=max(checks+meanchecks)<=1e-5 and all(c['self_pass'] for c in cells);cap=all(c['native_capability'] for c in cells) and all(c['capability'] and c['control_pass'] for c in ncells);A=instrument and cap
 result={'pred_a':A,'pred_b':A and all(c['passed'] for c in cells),'pred_c':A and all(c['passed'] for c in ncells),'instrument':instrument,'capability':cap,'max_sum_error':max(checks),'max_meanhead_error':max(meanchecks),'regional':cells,'newline':ncells,'body_forwards':count,'seconds':time.perf_counter()-tic,'arms_regional':['native','remove8','remove9','removejoint','fullsector','donorjoint','selfdonorjoint'],'arms_natural':['native','removejoint','fullsector','meanhead8'],'scope':'Frozen selectedmask45 on prospective newtemplates/rolechallenge and unusednewlinecacheindices. No alternativecandidate/refit; externalnative input/background/normalizers remain; not broadcorpusOOD or closedextraction.','source_shas':binding}
 torch.save(dict(regional=reg,newline_ce=ce,newline_margin=margin),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
