#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes<=26tokens;120seconds;knownhead8.2 andrest.
"""pred_a fullH replay<=1e-4 andfieldsum<=1e-10relative.
pred_b nearhead8.2 effecterror<=.1fullH andnegative>=20/24.
pred_c A andseparate head8.2/rest effectssumtofullH<=.1each4groups.
Null: computational head8.2 dominance failsphysicalpath intervention.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='ATTENTION8_PHI_HEAD_DONATION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('384bodyforwards96rows4arms; head8.2 H-input donor');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);gain=gen['lambdas'][0].cuda();eig=gen['eigenvalues'][:4].cuda();cache=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);heads=torch.load(P/'ATTENTION8_PHI_HEADS_V1_ARTIFACT.pt',weights_only=True)['head_readings'].cuda();q=cache['parents'][:,:,1].cuda();r8=cache['rho8_squared'].cuda();nmax=q.shape[1];values=torch.zeros(96,nmax,3,dtype=torch.float64,device='cuda');ctx={};reg=torch.zeros(96,4,2,dtype=torch.float64);fields=torch.zeros(96,3,nmax,dtype=torch.float64);count=0
 for i,row in enumerate(rows):
  j=row['donor_id'];n=len(row['ids']);dh=heads[j,:n]-heads[i,:n];each=2*(dh*q[i,:n,None,:]*eig).sum(-1)/r8[i,:n,None];values[i,:n,0]=each.sum(-1);values[i,:n,1]=each[:,2];values[i,:n,2]=each[:,[0,1,3,4,5,6,7,8]].sum(-1)
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  pos=int(prov['cue_positions'][i]);dv=values[i,:n,arm-1][None,:]*gain/ctx['rho'];dv[:,:pos+1]=0;s=(routing(args[0],p,1)@dv[...,None])[...,0];fields[i,arm-1,:n]=s[0].cpu();return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(4):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==384
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'MLP7_QPATH_PORT_DONATION_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,:2],ref[:,[0,4]]);sumerr=rel(fields[:,1]+fields[:,2],fields[:,0]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];c=z[::2,0,0]-z[1::2,0,0];arms=[];effects=[]
  for arm in range(1,4):
   delta=z[:,arm,0]-z[:,0,0];e=delta.clone();e[::2]*=-1;effects.append(e);arms.append(dict(arm=arm,transfer=float(e.mean()/c.mean()),positive=int((e>0).sum()),negative=int((e<0).sum()),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/delta.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,head8_2_relative_error=rel(effects[1],effects[0]),composition_error=rel(effects[1]+effects[2],effects[0])))
 A=replay<=1e-4 and sumerr<=1e-10;B=A and records[2]['head8_2_relative_error']<=.1 and records[2]['arms'][1]['negative']>=20;C=A and all(r['composition_error']<=.1 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'field_sum_error':sumerr,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','fullH_only','head8_2_H_only','restH_only'],'scope':'Head-specific H-reading donation through Q-containing path, recipientQ/R8/head9QK/rho andnativebackground retained. Reusedcontexts; notwholehead8.2 donation.'}
 torch.save(dict(regional=reg,donor_fields=fields),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
