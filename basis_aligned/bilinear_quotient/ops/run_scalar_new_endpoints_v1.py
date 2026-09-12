#!/usr/bin/env python3
# BQGATE:216bodyforwards;72prefixes;3arms;120seconds;frozenrank64;no fitting.
"""pred_a noedit<=1e-6relative, matched donor IDs/lengths, nonzero fields.
pred_b all3families capable andremoval coverage>=.5, signs>=10/12, unrelated<=.5target.
pred_c B plus donor transfer>=.5, signs>=20/24, unrelated<=.5target eachfamily.
Null: regionalcomponent doesnotgeneralize to new output spellings.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='SCALAR_NEW_ENDPOINTS_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];validate(rows);assert len(rows)==72
 for i,r in enumerate(rows):assert r['donor_id']==i^1 and len(r['ids'])==len(rows[i^1]['ids'])
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('216bodyforwards72newoutput-endpointrows;frozenrank64');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('cue_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};context={};cache={};checks=[];norms=[];count=0
 def hook(index,module,args,output):
  own=runtime.scalar(args[0],context['tokens'],p,index);arm=context['arm'];row=context['row'];write=own[...,None]*p['writers'][index]
  if arm==0:
   cache[row,index]=own.clone();norms.append(float(own.norm()));noedit=(output[0].double()-write+write).to(output[0].dtype);checks.append(float((noedit-output[0]).norm()/output[0].norm().clamp_min(1e-30)));return output
  if arm==1:return output[0]-write.to(output[0].dtype),output[1]
  donor=rows[row]['donor_id'];incoming=cache[donor,index];assert incoming.shape==own.shape
  return output[0]+((incoming-own)[...,None]*p['writers'][index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[l].attn.register_forward_hook(lambda m,a,o,i=i:hook(i,m,a,o)) for i,l in enumerate([8,9])];reg=torch.zeros(72,3,2,dtype=torch.float64)
 try:
  # All native donor fields exist before any donor intervention. Suffix recomputed.
  for arm in range(3):
   for i,row in enumerate(rows):
    ids=torch.tensor([row['ids']],device='cuda');context.update(arm=arm,row=i,tokens=ids);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for handle in handles:handle.remove()
 assert count==216 and len(cache)==144
 records=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];native=reg[ix,0,0];contrast=native[::2]-native[1::2];mean=float(contrast.mean());cap=mean>=.2 and int((contrast>0).sum())>=10;rem=reg[ix,1,0]-native;reduction=rem[1::2]-rem[::2];swap=reg[ix,2,0]-native;directed=swap.clone();directed[::2]*=-1
  remcov=float(reduction.mean()/contrast.mean()) if abs(mean)>1e-12 else None;swcov=float(directed.mean()/contrast.mean()) if abs(mean)>1e-12 else None;rr=float((reg[ix,1,1]-reg[ix,0,1]).abs().mean()/rem.abs().mean().clamp_min(1e-30));sr=float((reg[ix,2,1]-reg[ix,0,1]).abs().mean()/swap.abs().mean().clamp_min(1e-30))
  records.append(dict(family=family,name=rows[ix[0]]['family_name'],native_mean_contrast=mean,native_positive_pairs=int((contrast>0).sum()),capable=cap,removal_coverage=remcov,removal_positive_pairs=int((reduction>0).sum()),removal_unrelated_ratio=rr,donor_transfer=swcov,donor_positive_directions=int((directed>0).sum()),donor_unrelated_ratio=sr,removal_pass=cap and remcov>=.5 and int((reduction>0).sum())>=10 and rr<=.5,donor_pass=cap and swcov>=.5 and int((directed>0).sum())>=20 and sr<=.5))
 A=max(checks)<=1e-6 and min(norms)>1e-8;B=A and all(r['removal_pass'] for r in records);result={'pred_a':A,'pred_b':B,'pred_c':B and all(r['donor_pass'] for r in records),'max_noedit_relative_error':max(checks),'min_native_scalar_norm':min(norms),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Frozenrank64 on72score-free newcueprompts. Removal/pairednative-donorfields with actual sequentialrecipient states; hybrid conditionalintervention, not full donor model. Weak nativecapability reportedwithoutdroppingrows. Controlleddistribution shift, notcorpus/pretraining OOD; new endpoints; cue templates/controlcontrast reused.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
