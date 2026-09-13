#!/usr/bin/env python3
# BQGATE:864bodyforwards;96prefixes<=26tokens;180seconds;value/routing path interventions.
"""pred_a native/full head8.2 margin replay<=1e-4relative.
pred_b nearfirst-value-only effecterror<=.4full andnegative>=20/24.
pred_c separatelydonated generatedcurrent/first effectssumtofull<=.1eachgroup.
Null: first-only failsphysicaleffect and/or sectorcomposition fails.
Price864forwards180sec; cachednativeconditionalfields, fullbackground charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='ATTENTION8_PHI_VALUE_PORTS_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('864bodyforwards96rows9arms; conditional joint-routing/value input changes');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();writer=torch.load(FOLDER/'program.pt',weights_only=True)['writers'][1].cuda();sector=torch.load(P/'ATTENTION8_PHI_VALUE_ROUTING_V1_ARTIFACT.pt',weights_only=True)['sector_fields'];terms=torch.load(P/'ATTENTION8_PHI_ROUTING_VALUE_DELTA_V1_ARTIFACT.pt',weights_only=True)['fields'];fields=torch.stack([sector.sum(-1),sector[:,:,0],sector[:,:,1],terms[:,:,0],terms[:,:,1],terms[:,:,2],terms[:,:,1]+terms[:,:,2],terms[:,:,3]+terms[:,:,4]],1).cuda();ctx={};reg=torch.zeros(96,9,2,dtype=torch.float64);count=0
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  return output[0]+(fields[i,arm-1,:n][None,:,None]*writer).to(output[0].dtype),output[1]
 handle=model.transformer.h[9].attn.register_forward_hook(head9)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(9):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:handle.remove()
 assert count==864
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'ATTENTION8_PHI_HEAD_DONATION_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,:2],ref[:,[0,2]]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];c=z[::2,0,0]-z[1::2,0,0];arms=[];effects=[]
  for arm in range(1,9):
   delta=z[:,arm,0]-z[:,0,0];e=delta.clone();e[::2]*=-1;effects.append(e);arms.append(dict(arm=arm,transfer=float(e.mean()/c.mean()),positive=int((e>0).sum()),negative=int((e<0).sum()),relative_error_to_full=rel(e,effects[0]),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/delta.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,sector_composition_error=rel(effects[1]+effects[2],effects[0]),five_term_group_composition_error=rel(effects[3]+effects[6]+effects[7],effects[0]),without_mixed_effect_error=rel(effects[3]+effects[6],effects[0])))
 A=replay<=1e-4;B=A and records[2]['arms'][5]['relative_error_to_full']<=.4 and records[2]['arms'][5]['negative']>=20;C=A and all(r['sector_composition_error']<=.1 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','fullhead8_2','generatedcurrent','generatedfirst','routingonly','currentonly','firstonly','bothvalues','routing_value_mixed'],'scope':'Physical cached path-field injection athead9.8 with native suffix recomputed; equivalent declared head8.2 input-port changes under retained recipientQ7/norm/head9routing. Not whole-head swap or OOD confirmation.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
