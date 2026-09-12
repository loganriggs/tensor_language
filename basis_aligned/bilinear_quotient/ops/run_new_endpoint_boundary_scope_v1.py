#!/usr/bin/env python3
# BQGATE:216bodyforwards;72prefixes;3arms;120seconds;frozenrank64;no fitting.
"""pred_a noedit<=1e-6relative, matched donor IDs/lengths, nonzero fields.
pred_b fullsector removal coverage>=.5, signs>=10/12, unrelated<=.5target eachfamily.
pred_c B plus fullsector coveragegain>=.05 eachfamily.
Null: missed newoutput coverage is not explained by the selected QK boundary.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='NEW_ENDPOINT_BOUNDARY_SCOPE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==72
 for i,r in enumerate(rows):assert r['donor_id']==i^1 and len(r['ids'])==len(rows[i^1]['ids'])
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('216bodyforwards72newoutput-endpointrows;frozenrank64');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('cue_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};context={};cache={};checks=[];norms=[];count=0;fullp=dict(p);fullp['key_basis']=torch.zeros_like(p['key_basis'])
 def hook(index,module,args,output):
  own=runtime.scalar(args[0],context['tokens'],p,index);arm=context['arm'];row=context['row'];write=own[...,None]*p['writers'][index]
  if arm==0:
   cache[row,index]=own.clone();norms.append(float(own.norm()));noedit=(output[0].double()-write+write).to(output[0].dtype);checks.append(float((noedit-output[0]).norm()/output[0].norm().clamp_min(1e-30)));return output
  if arm==1:return output[0]-write.to(output[0].dtype),output[1]
  whole=runtime.scalar(args[0],context['tokens'],fullp,index);return output[0]-(whole[...,None]*p['writers'][index]).to(output[0].dtype),output[1]
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
 old=torch.load(P/'SCALAR_NEW_ENDPOINTS_V1_ARTIFACT.pt',weights_only=True)['regional'];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=rel(reg[:,:2],old[:,:2]);records=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];z=reg[ix];c=z[::2,0,0]-z[1::2,0,0];means=[]
  for arm in [1,2]:
   edit=z[:,arm,0]-z[:,0,0];reduction=edit[1::2]-edit[::2];means.append(dict(coverage=float(reduction.mean()/c.mean()),positive_pairs=int((reduction>0).sum()),unrelated_ratio=float((z[:,arm,1]-z[:,0,1]).abs().mean()/edit.abs().mean())))
  records.append(dict(family=family,name=rows[ix[0]]['family_name'],selected=means[0],full=means[1],coverage_gain=means[1]['coverage']-means[0]['coverage']))
 A=max(checks)<=1e-6 and min(norms)>1e-8 and replay<=1e-6;B=A and all(r['full']['coverage']>=.5 and r['full']['positive_pairs']>=10 and r['full']['unrelated_ratio']<=.5 for r in records);C=B and all(r['coverage_gain']>=.05 for r in records);result={'pred_a':A,'pred_b':B,'pred_c':C,'primary_replay_error':replay,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Same newendpoint panel. Fullvalue-sector reference uses zero reflection basis, hence original fulljointkey numerator. Existing fullsector newlinefailure is retained. This is a scope diagnostic, not primarycomponent promotion or newOOD confirmation.'}
 torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
