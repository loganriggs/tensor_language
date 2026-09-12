#!/usr/bin/env python3
# BQGATE:1312bodyforwards;80prefixes<=179tokens;300seconds;no fitting.
"""pred_a sector/native sum<=1e-5, native/fullmask/mean effect replay<=1e-4relative.
pred_b EXISTS fixednonempty mask: EACHtemplate >=10/12cue reductions,>=50%native,
unrelated<=.5regional. Reportall15masks.
pred_c SAME Bmask preserves BOTHnatural halves meanabsCE<=.02,maxabs<=.1,
with actual native capability and meanhead positive controls.
Null: current/first source sectors do not yield selective regional units.
Price1312bodyforwards80rows16masks+32meancontrols,300seconds,no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
from scalar_value_sectors_v1 import head_scalar_sectors
STEM='SCALAR_VALUE_SECTOR_FACTORIAL_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 regional=json.loads((P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json').read_text())['rows']
 natural=[r for r in json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text())['rows'] if r['pool']=='fineweb']
 assert len(regional)==48 and len(natural)==32 and max(len(r['ids']) for r in regional+natural)<=179
 assert json.loads((P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json').read_text())['pred_a']
 assert json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_RESULT.json').read_text())['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('1312bodyforwards80rows; all16 current-first source-sector masks plus mean controls');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists()
 tic=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'].cuda()
 prior=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True,map_location='cpu')
 mean_head=prior['mean_head'].cuda();context={};checks=[];sector_replay=[];body_count=0
 def capture(index,args):context[('preov',index)]=args[0]
 def hook(index,head,module,args,output):
  mask=context['mask'];bits=(mask>>(2*index))&3
  if mask==16 and index==0:
   z=context[('preov',index)].reshape(*args[0].shape[:2],9,128);O=module.c_proj.weight[:,128*head:128*(head+1)]
   changed=output[0]-F.linear(z[:,:,head],O)+F.linear(mean_head,O)[None,None,:];edited=z.clone();edited[:,:,head]=mean_head
   direct=module.c_proj(edited.reshape_as(args[0]));checks.append(float((changed-direct).norm()/direct.norm()));return changed,output[1]
  if mask==0 or bits:
   parts=head_scalar_sectors(args[0],context['tokens'],producer,index)
   if mask==0:
    original=head_scalar(args[0],context['tokens'],producer,index);sector_replay.append(float((parts.sum(-1)-original).norm()/original.norm().clamp_min(1e-30)))
   if bits:
    scalar=sum(parts[...,j] for j in range(2) if bits&(1<<j));return output[0]-(scalar[...,None]*writers[index]).to(output[0].dtype),output[1]
  return output
 handles=[]
 for index,layer,head in ((0,8,2),(1,9,8)):
  attn=model.transformer.h[layer].attn
  handles.append(attn.c_proj.register_forward_pre_hook(lambda m,a,i=index:capture(i,a)))
  handles.append(attn.register_forward_hook(lambda m,a,o,i=index,h=head:hook(i,h,m,a,o)))
 def forward(tokens,mask):
  nonlocal body_count
  context.update(tokens=tokens,mask=mask)
  x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  body_count+=1
  return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 reg=torch.zeros(48,16,2,dtype=torch.float64);ce=torch.zeros(32,17,dtype=torch.float64);nl_margin=torch.zeros_like(ce)
 try:
  for i,row in enumerate(regional):
   tokens=torch.tensor([row['ids']],device='cuda')
   for a in range(16):
    logits=forward(tokens,a);reg[i,a,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,a,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
  for i,row in enumerate(natural):
   tokens=torch.tensor([row['ids']],device='cuda')
   for a in range(17):
    logits=forward(tokens,a);ce[i,a]=-logits.log_softmax(-1)[198].cpu();nl_margin[i,a]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert body_count==1312
 def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 oldreg=torch.load(P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['margins'];oldnl=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['newline_ce']
 replay=dict(regional_baseline=relative(reg[:,0],oldreg[:,0]),regional_fullmask_effect=relative(reg[:,15]-reg[:,0],oldreg[:,3]-oldreg[:,0]),newline_baseline=relative(ce[:,0],oldnl[:,0]),newline_fullmask_effect=relative(ce[:,15]-ce[:,0],oldnl[:,3]-oldnl[:,0]),newline_mean_effect=relative(ce[:,16]-ce[:,0],oldnl[:,4]-oldnl[:,0]),meanhead_subtraction=max(checks),sector_sum=max(sector_replay))
 capability=[];records=[]
 for family in range(2):
  ri=[i for i,r in enumerate(regional) if r['family']==family];ni=[i for i,r in enumerate(natural) if r['family']==family];contrast=reg[ri,0,0][::2]-reg[ri,0,0][1::2];control=ce[ni,16]-ce[ni,0]
  capability.append(dict(family=family,regional=float(contrast.mean())>=.2 and int((contrast>0).sum())>=10,newline=float(nl_margin[ni,0].mean())>=.2 and int((nl_margin[ni,0]>0).sum())>=12 and float(ce[ni,0].mean())<=5,meanhead=float(control.mean())>=.02 and int((control>0).sum())>=8))
 for mask in range(1,16):
  rcells=[];ncells=[]
  for family in range(2):
   ri=[i for i,r in enumerate(regional) if r['family']==family];ni=[i for i,r in enumerate(natural) if r['family']==family];contrast=reg[ri,0,0][::2]-reg[ri,0,0][1::2];effect=reg[ri,mask,0]-reg[ri,0,0];reduction=effect[1::2]-effect[::2];fraction=float(reduction.mean()/contrast.mean());positive=int((reduction>0).sum());ratio=float((reg[ri,mask,1]-reg[ri,0,1]).abs().mean()/effect.abs().mean().clamp_min(1e-30));delta=ce[ni,mask]-ce[ni,0]
   rcells.append(dict(family=family,coverage=fraction,positive_pairs=positive,unrelated_ratio=ratio,passed=fraction>=.5 and positive>=10 and ratio<=.5))
   ncells.append(dict(family=family,mean_change=float(delta.mean()),meanabs_change=float(delta.abs().mean()),maxabs_change=float(delta.abs().max()),passed=float(delta.abs().mean())<=.02 and float(delta.abs().max())<=.1))
  records.append(dict(mask=mask,elements=[['8current','8first','9current','9first'][j] for j in range(4) if mask&(1<<j)],regional=rcells,newline=ncells,regional_pass=all(c['passed'] for c in rcells),newline_pass=all(c['passed'] for c in ncells)))
 A=max(replay.values())<=1e-4 and replay['sector_sum']<=1e-5 and replay['meanhead_subtraction']<=1e-5
 cap=all(c['regional'] and c['newline'] and c['meanhead'] for c in capability);bpass=[r['mask'] for r in records if r['regional_pass']];cpass=[r['mask'] for r in records if r['regional_pass'] and r['newline_pass']]
 result={'pred_a':A,'pred_b':A and cap and bool(bpass),'pred_c':A and cap and bool(cpass),'checks':replay,'capability':capability,'records':records,'regional_passers':bpass,'joint_passers':cpass,'body_forwards':body_count,'seconds':time.perf_counter()-tic,'scope':'All15 fixed current/first value-sector masks; exploratory reused-panel screen with native recursive interactions. Candidate selection requires fresh confirmation; fullmask prior failure preserved.','source_shas':binding}
 torch.save(dict(regional=reg,newline_ce=ce,newline_margin=nl_margin),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','records')}),flush=True)
if __name__=='__main__':main()
