#!/usr/bin/env python3
# BQGATE:288bodyforwards;96prefixes<=26tokens;180seconds;frozen4modes.
"""pred_a old24 native/removal/donor replay<=1e-4, selfdonor<=1e-8, newnativecapability.
pred_b eachnewfamily removal>=.1,10/12positive,unrelated<=.5.
pred_c B and eachnewfamily donor>=.1,20/24positive,unrelated<=.5.
Null: four-mode value path doesnot generalize to these new cue constructions.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP8_VALUE_FRESH_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());oldrows=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=oldrows+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96

 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('288bodyforwards96regional;24oldanchor72fresh; frozenphi donor andremoval');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen={k:v.cuda() for k,v in torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True).items()};reader=gen['reader'];gain=gen['lambdas'][0];mean=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True)['mean_head'].cuda();context={};donors={};selfchecks=[];count=0;reg=torch.zeros(96,3,2,dtype=torch.float64);ce=torch.zeros(64,4,dtype=torch.float64);nlmargin=torch.zeros_like(ce)
 def capture8(module,args):context['preov8']=args[0]
 def head8(module,args,output):
  if context['pool']=='nl' and context['arm']==3:
   z=context['preov8'].reshape(*args[0].shape[:2],9,128);O=module.c_proj.weight[:,256:384];return output[0]-F.linear(z[:,:,2],O)+F.linear(mean,O)[None,None,:],output[1]
  return output
 def pre8(module,args):context['phi']=((args[0].double()@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]).sum(-1)
 def post8(module,args,output):context['qfull']=(output-module.Down_bias).double()@reader
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];context['rho9']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  arm=context['arm'];pool=context['pool'];i=context['row']
  if arm==0:
   if pool=='reg':donors[i]=context['phi'].clone()
   return output
  if pool=='nl' and arm==3:return output
  gamma=routing(args[0],p,1);phi=context['phi'];scale=gain/context['rho9']
  if pool=='reg':
   selfchecks.append(float((phi-donors[i]).norm()/donors[i].norm().clamp_min(1e-30)))
   dv=-phi*scale if arm==1 else (donors[rows[i]['donor_id']]-phi)*scale
  else:dv=-(phi if arm==1 else context['qfull'])*scale
  s=(gamma@dv[...,None])[...,0];return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].attn.c_proj.register_forward_pre_hook(capture8),model.transformer.h[8].attn.register_forward_hook(head8),model.transformer.h[8].mlp.register_forward_pre_hook(pre8),model.transformer.h[8].mlp.register_forward_hook(post8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 def forward(row,pool,i,arm):
  nonlocal count
  ids=torch.tensor([row['ids']],device='cuda');context.update(pool=pool,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 try:
  for arm in range(3):
   for i,row in enumerate(rows):
    logits=forward(row,'reg',i,arm);reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==288
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 oldreg=torch.load(P/'MLP8_VALUE_TRANSFER_V1_ARTIFACT.pt',weights_only=True)['regional'][:24]
 replay=dict(anchor=rel(reg[:24],oldreg),selfdonor=max(selfchecks));records=[]
 for family in range(3):
  ix=[i+24 for i,r in enumerate(fresh) if r['family']==family];z=reg[ix]
  c=z[::2,0,0]-z[1::2,0,0];removed=c-(z[::2,1,0]-z[1::2,1,0]);e=z[:,2,0]-z[:,0,0];directed=e.clone();directed[::2]*=-1
  rec=dict(family=family,native_contrast=float(c.mean()),native_positive=int((c>0).sum()),capable=float(c.mean())>=.2 and int((c>0).sum())>=10,removal_coverage=float(removed.mean()/c.mean()),removal_positive=int((removed>0).sum()),removal_unrelated=float((z[:,1,1]-z[:,0,1]).abs().mean()/(z[:,1,0]-z[:,0,0]).abs().mean().clamp_min(1e-30)),donor_transfer=float(directed.mean()/c.mean()),donor_positive=int((directed>0).sum()),donor_unrelated=float((z[:,2,1]-z[:,0,1]).abs().mean()/e.abs().mean().clamp_min(1e-30)))
  rec['removal_pass']=rec['capable'] and rec['removal_coverage']>=.1 and rec['removal_positive']>=10 and rec['removal_unrelated']<=.5
  rec['donor_pass']=rec['capable'] and rec['donor_transfer']>=.1 and rec['donor_positive']>=20 and rec['donor_unrelated']<=.5
  records.append(rec)
 A=replay['anchor']<=1e-4 and replay['selfdonor']<=1e-8 and all(r['capable'] for r in records);B=A and all(r['removal_pass'] for r in records);C=B and all(r['donor_pass'] for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay':replay,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'72 fresh full-prefix city contexts, 24 reused replay anchors; frozen phi4 donor with recipient QK/norm; native prefix/background retained. Controlled context shift, not corpus OOD.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
