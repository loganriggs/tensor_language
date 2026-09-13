#!/usr/bin/env python3
# BQGATE:480bodyforwards;96prefixes<=29tokens;180seconds;Q7-token interaction.
"""pred_a F-only replay<=1e-4 andfieldQF=Q+F+mixed<=1e-10relative.
pred_b previoussix Baltimoremisses FafterQpositive>=5/6.
pred_c isolatedmixed effect predicts physicaldifference-of-differences<=.15 eachgroup.
Null: Q7 modulation failsphysical signprediction ornonlinear suffix destroyscomposition.
Price480forwards180sec; cachednativeconditionalinputs, background charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='FIRST_TOKEN_Q_INTERACTION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('480bodyforwards96rows5arms; Q7-by-token crosspath');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();writer=torch.load(FOLDER/'program.pt',weights_only=True)['writers'][1].cuda();fields=torch.load(P/'FIRST_TOKEN_Q_INTERACTION_V1_FIELDS.pt',weights_only=True)['fields'].cuda();ctx={};reg=torch.zeros(96,5,2,dtype=torch.float64);count=0
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  return output[0]+(fields[i,arm-1,:n][None,:,None]*writer).to(output[0].dtype),output[1]
 handle=model.transformer.h[9].attn.register_forward_hook(head9)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:handle.remove()
 assert count==480
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'FIRST_TOKEN_PATH_FRESH_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,:2],ref[:,[0,2]]);sumerr=rel(fields[:,0]+fields[:,1]+fields[:,3],fields[:,2]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];native=z[::2,0,0]-z[1::2,0,0];effects=[];arms=[]
  for arm in range(1,5):
   e=z[:,arm,0]-z[:,0,0];e=e.clone();e[::2]*=-1;effects.append(e);arms.append(dict(arm=arm,transfer=float(e.mean()/native.mean()),positive=int((e>0).sum()),negative=int((e<0).sum())))
  measured_mixed=effects[2]-effects[1]-effects[0];afterq=effects[2]-effects[1];records.append(dict(group=group,arms=arms,F_afterQ_positive=int((afterq>0).sum()),F_afterQ_negative=int((afterq<0).sum()),mixed_prediction_error=rel(effects[3],measured_mixed)))
 problematic=torch.arange(85,96,2);f_after=reg[problematic,3,0]-reg[problematic,2,0];f_native=reg[problematic,1,0]-reg[problematic,0,0];A=replay<=1e-4 and sumerr<=1e-10;B=A and int((f_after>0).sum())>=5;C=A and all(r['mixed_prediction_error']<=.15 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'field_sum_error':sumerr,'Baltimore_F_native':f_native.tolist(),'Baltimore_F_afterQ':f_after.tolist(),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','F_only','Q_only','Q_and_F','isolated_mixed'],'scope':'Physical Q7/token-value input corners in crossfirst path only; nativejoint routing/norm/background. Reused96prefixes, no repair of failed fresh signcriterion orwholemodule claim.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
