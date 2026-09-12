#!/usr/bin/env python3
# BQGATE:480bodyforwards;96prefixes<=26tokens;120seconds;no fitting.
"""pred_a native/full replay<=1e-4 andcity+post scalarfieldsum<=1e-10.
pred_b postcity-vs-full donor branch-relativeerror<=.5 each4groups.
pred_c A andcityonlypositive>=20/24eachgroup, postcitynearquote negative>=20/24.
Null: computational source partition doesnot predict the claimed physical source effects.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='PHI4_SOURCE_DONATION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items())
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('480bodyforwards96prefixes fivearms; physicalsource donation andfinalonly control');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen={k:v.cuda() for k,v in torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True).items()};gain=gen['lambdas'][0];ctx={};donors={};nmax=max(len(r['ids']) for r in rows)
 modes=torch.zeros(96,nmax,4,dtype=torch.float64);rho=torch.zeros(96,nmax,dtype=torch.float64);gam=torch.zeros(96,nmax,nmax,dtype=torch.float64);ds=torch.zeros_like(rho);reg=torch.zeros(96,5,2,dtype=torch.float64);count=0;fields=torch.zeros(96,4,nmax,dtype=torch.float64)
 def pre8(module,args):ctx['modes']=(args[0].double()@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  i=ctx['i'];n=args[0].shape[1];gamma=routing(args[0],p,1)
  if ctx['arm']==0:
   donors[i]=ctx['modes'].sum(-1).clone();modes[i,:n]=ctx['modes'][0].cpu();rho[i,:n]=ctx['rho'][0].cpu();gam[i,:n,:n]=gamma[0].cpu();return output
  delta=(donors[rows[i]['donor_id']]-ctx['modes'].sum(-1))*gain/ctx['rho']
  positions=[k for k,(aa,bb) in enumerate(zip(rows[i]['ids'],rows[rows[i]['donor_id']]['ids'])) if aa!=bb];assert len(positions)==1;pos=positions[0]
  if ctx['arm']==2:
   mask=torch.zeros_like(delta);mask[:,pos]=1;delta=delta*mask
  elif ctx['arm']==3:
   mask=torch.zeros_like(delta);mask[:,pos+1:]=1;delta=delta*mask
  s=(gamma@delta[...,None])[...,0]
  if ctx['arm']==4:
   last=s[:,-1].clone();s=torch.zeros_like(s);s[:,-1]=last
  fields[i,ctx['arm']-1,:n]=s[0].cpu()
  return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].mlp.register_forward_pre_hook(pre8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for arm in range(5):
   for i,row in enumerate(rows):
    ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,arm=arm);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==480
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 ref=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=dict(old=rel(reg[:24,:2],ref[:24]),fresh=rel(reg[24:,:2],ref[24:]));sumerror=rel(fields[:,1]+fields[:,2],fields[:,0]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];c=z[::2,0,0]-z[1::2,0,0];arms=[];effects=[]
  for arm in range(1,5):
   e=z[:,arm,0]-z[:,0,0];directed=e.clone();directed[::2]*=-1;effects.append(directed)
   arms.append(dict(arm=arm,transfer=float(directed.mean()/c.mean()),positive=int((directed>0).sum()),negative=int((directed<0).sum()),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/e.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,post_relative_to_full=rel(effects[2],effects[0]),post_error_over_native=float((effects[2]-effects[0]).norm()/c.repeat_interleave(2).norm()),finalonly_relative_to_full=rel(effects[3],effects[0]),sum_effect_nonadditivity=rel(effects[1]+effects[2],effects[0])))
 A=max(replay.values())<=1e-4 and sumerror<=1e-10;B=A and all(r['post_relative_to_full']<=.5 for r in records);C=A and all(r['arms'][1]['positive']>=20 for r in records) and records[2]['arms'][2]['negative']>=20
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay':replay,'field_sum_error':sumerror,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','fullphi_donor','citysource_donor','postcitysource_donor','finaltarget_only_fullphi_donor'],'scope':'Reused96prefixes; physical source-edge donation partitions, native suffix recomputed. Finaltargetonly descriptive sufficiency control. No newtext confirmation.'}
 torch.save(dict(regional=reg,donor_fields=fields),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
