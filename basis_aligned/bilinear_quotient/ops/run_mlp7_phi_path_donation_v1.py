#!/usr/bin/env python3
# BQGATE:672bodyforwards;96prefixes<=26tokens;180seconds;frozen generated path values.
"""pred_a replay<=1e-4 and generatedpath fieldsum<=1e-4relative.
pred_b nearpostQcont negative>=20/24 and <=.5relativeerror to fullposteffect.
pred_c A and separate Qcont/Qfree effect sum<=.1relative fullsource each8cells.
Null: generated path groups fail to account for and compose as their native source effects.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP7_PHI_PATH_DONATION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('672bodyforwards96rows7arms; generatedpath valuedonation');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gain=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)['lambdas'][0].cuda();cache=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);nmax=cache['phi_native'].shape[1];paths=cache['phi_paths'].cuda();values=torch.stack([cache['phi_native'].cuda(),paths[:,:,[1,3,5]].sum(-1),paths[:,:,[0,2,4]].sum(-1)],-1);ctx={};reg=torch.zeros(96,7,2,dtype=torch.float64);fields=torch.zeros(96,6,nmax,dtype=torch.float64);count=0
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  # 1/2 full city/post, 3/4 city Q-containing/free, 5/6 post Q-containing/free.
  component=0 if arm<=2 else (1 if arm in [3,5] else 2);city=arm in [1,3,4];pos=int(prov['cue_positions'][i]);j=rows[i]['donor_id'];dv=(values[j,:n,component]-values[i,:n,component])[None,:]*gain/ctx['rho'];mask=torch.zeros_like(dv)
  if city:mask[:,pos]=1
  else:mask[:,pos+1:]=1
  s=(routing(args[0],p,1)@(dv*mask)[...,None])[...,0];fields[i,arm-1,:n]=s[0].cpu();return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(7):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==672
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'PHI4_SOURCE_DONATION_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,:3],ref[:,[0,2,3]]);fielderr=max(rel(fields[:,2]+fields[:,3],fields[:,0]),rel(fields[:,4]+fields[:,5],fields[:,1]));records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];c=z[::2,0,0]-z[1::2,0,0];arms=[];e=[]
  for arm in range(1,7):
   delta=z[:,arm,0]-z[:,0,0];d=delta.clone();d[::2]*=-1;e.append(d);arms.append(dict(arm=arm,transfer=float(d.mean()/c.mean()),positive=int((d>0).sum()),negative=int((d<0).sum()),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/delta.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,city_composition_error=rel(e[2]+e[3],e[0]),post_composition_error=rel(e[4]+e[5],e[1]),post_Qcont_relative_to_full=rel(e[4],e[1])))
 A=replay<=1e-4 and fielderr<=1e-4;B=A and records[2]['arms'][4]['negative']>=20 and records[2]['post_Qcont_relative_to_full']<=.5;C=A and all(max(r['city_composition_error'],r['post_composition_error'])<=.1 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'field_sum_error':fielderr,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','full_city','full_post','city_Qcont','city_Qfree','post_Qcont','post_Qfree'],'scope':'Generated composite path-value donation with cached native parents, recipientQK/rho9; not donation ofQnode alone or wholeMLP7 removal. Reused96contexts; conditionalcausal/composition screen.'}
 torch.save(dict(regional=reg,donor_fields=fields),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
