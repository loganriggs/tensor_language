#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes<=29tokens;180seconds;absolute path removal.
"""pred_a native marginreplay<=1e-4relative.
pred_b cityKO near1/2 contrastincrease>=1% and>=10/12pairs; distant
contrastdecrease>=.3% and>=10/12pairs.
pred_c allsourceabsolutecontrastchange>=.5% eachfreshfamily andhalflinearity<=.1eachgroup.
Null: donor-sensitive path doesnot support donor-free interpretable removal.
Price384forwards180sec; nativebackground andcontextports remain charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='FIRST_TOKEN_ABSOLUTE_REMOVAL_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('384bodyforwards96rows4arms; donor-free removal');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();writer=torch.load(FOLDER/'program.pt',weights_only=True)['writers'][1].cuda();base=torch.load(P/'FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_FIELDS.pt',weights_only=True)['fields'];fields=-torch.stack([base[:,0],base[:,1],.5*base[:,1]],1).cuda();ctx={};reg=torch.zeros(96,4,2,dtype=torch.float64);count=0
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  return output[0]+(fields[i,arm-1,:n][None,:,None]*writer).to(output[0].dtype),output[1]
 handle=model.transformer.h[9].attn.register_forward_hook(head9)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(4):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:handle.remove()
 assert count==384
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'FIRST_TOKEN_PATH_FRESH_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,0],ref[:,0]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];native=z[::2,0,0]-z[1::2,0,0];effects=[];arms=[]
  for arm in range(1,4):
   effect=z[:,arm,0]-z[:,0,0];effects.append(effect);damage=native-(z[::2,arm,0]-z[1::2,arm,0]);arms.append(dict(arm=arm,mean_fractional_contrast_damage=float(damage.mean()/native.mean()),pairs_contrast_decreased=int((damage>0).sum()),pairs_contrast_increased=int((damage<0).sum()),unrelated_margin_effect_ratio=float((z[:,arm,1]-z[:,0,1]).abs().mean()/effect.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,half_linearity_error=rel(2*effects[2],effects[1])))
 A=replay<=1e-4;B=A and all(records[k]['arms'][0]['pairs_contrast_increased']>=10 and records[k]['arms'][0]['mean_fractional_contrast_damage']<=-.01 for k in (1,2)) and records[3]['arms'][0]['pairs_contrast_decreased']>=10 and records[3]['arms'][0]['mean_fractional_contrast_damage']>=.003;C=A and all(abs(r['arms'][1]['mean_fractional_contrast_damage'])>=.005 for r in records[1:]) and all(r['half_linearity_error']<=.1 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','citysource_KO','allsource_KO','half_allsource_KO'],'scope':'Donor-free crossfirst value removal atselectedhead9edge; citysourcehasexternalcueposition,allsourcehasnone. Reusedregionalcontexts; unrelatedFineWebcontrols pending. Positivecontrastdamage means weaker regionalbehavior; no broadselectivityclaim.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
