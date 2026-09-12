#!/usr/bin/env python3
# BQGATE:432bodyforwards;72prefixes<=26tokens;180seconds;frozen4modes and fullquadratic.
"""pred_a oldfresh native/phi4removal/donor replay<=1e-4 and nativecapability.
pred_b fullquad-versus-phi4 effects<=5%nativecontrastnorm for removal anddonor eachfamily.
pred_c B plus nearquote fullquad donor<0 and fullselected9removal>=.1.
Null: rank truncation is responsible for the observed context-dependent reversal.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP8_VALUE_FRESH_FULL_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==72

 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('432bodyforwards72fresh sixarms;fullquad discriminator');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen={k:v.cuda() for k,v in torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True).items()};reader=gen['reader'];gain=gen['lambdas'][0];context={};donors={};selfchecks=[];count=0;reg=torch.zeros(72,6,2,dtype=torch.float64)
 def pre8(module,args):context['phi']=((args[0].double()@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]).sum(-1)
 def post8(module,args,output):context['qfull']=(output-module.Down_bias).double()@reader
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];context['rho9']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  arm=context['arm'];pool=context['pool'];i=context['row']
  if arm==0:
   if pool=='reg':donors[i]=(context['phi'].clone(),context['qfull'].clone())
   return output
  gamma=routing(args[0],p,1);phi=context['phi'];scale=gain/context['rho9']
  selfchecks.append(float((phi-donors[i][0]).norm()/donors[i][0].norm().clamp_min(1e-30)))
  if arm==1:dv=-phi*scale
  elif arm==2:dv=-context['qfull']*scale
  elif arm==3:dv=(donors[rows[i]['donor_id']][0]-phi)*scale
  elif arm==4:dv=(donors[rows[i]['donor_id']][1]-context['qfull'])*scale
  elif arm==5:dv=-(args[0].double()@reader)
  else:raise AssertionError(arm)
  s=(gamma@dv[...,None])[...,0];return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].mlp.register_forward_pre_hook(pre8),model.transformer.h[8].mlp.register_forward_hook(post8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 def forward(row,pool,i,arm):
  nonlocal count
  ids=torch.tensor([row['ids']],device='cuda');context.update(pool=pool,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 try:
  for arm in range(6):
   for i,row in enumerate(rows):
    logits=forward(row,'reg',i,arm);reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==432
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 oldreg=torch.load(P/'MLP8_VALUE_FRESH_V1_ARTIFACT.pt',weights_only=True)['regional'][24:]
 replay=dict(anchor=rel(reg[:,[0,1,3]],oldreg),selfdonor=max(selfchecks));records=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];z=reg[ix];c=z[::2,0,0]-z[1::2,0,0]
  arms=[];effects={}
  for arm in range(1,6):
   e=z[:,arm,0]-z[:,0,0]
   if arm in [1,2,5]:effect=e[1::2]-e[::2];den=c
   else:effect=e.clone();effect[::2]*=-1;den=c.repeat_interleave(2)
   effects[arm]=effect
   arms.append(dict(arm=arm,coverage=float(effect.mean()/den.mean()),positive=int((effect>0).sum()),count=len(effect),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/e.abs().mean().clamp_min(1e-30))))
  records.append(dict(family=family,capable=float(c.mean())>=.2 and int((c>0).sum())>=10,removal_error_over_native=float((effects[1]-effects[2]).norm()/c.norm()),donor_error_over_native=float((effects[3]-effects[4]).norm()/c.repeat_interleave(2).norm()),arms=arms))
 A=replay['anchor']<=1e-4 and replay['selfdonor']<=1e-8 and all(r['capable'] for r in records);B=A and all(max(r['removal_error_over_native'],r['donor_error_over_native'])<=.05 for r in records);q=records[1]['arms'];C=B and q[3]['coverage']<0 and q[4]['coverage']>=.1
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay':replay,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','phi4_remove','fullquad_remove','phi4_donor','fullquad_donor','fullselected9_remove'],'scope':'Reused fresh panel, full-quadratic discriminator of rank4 failure; donor numerator with recipient QK/rho, native context retained.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
