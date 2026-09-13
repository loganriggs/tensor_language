#!/usr/bin/env python3
# BQGATE:800bodyforwards;160prefixes<=248tokens;180seconds;hierarchy composition.
"""pred_a native/child removal anchors<=1e-4relative.
pred_b parent/jointphysicaloutcomes<=1e-5 and C+R=P field<=1e-10.
pred_c separatechild+remainder effects predictparent<=.1relativeCHILD each8cells.
Null: native suffix changes smallchild response underparentbackground.
Price800forwards180sec; cachedvalidatedchildfield, nativebackground retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from regional_even_routing_v1 import routing
from sparse_path_stability_atlas_v1 import digest
STEM='CROSSFIRST_HIERARCHY_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regrows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regrows);rows=regrows+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800forwards160rows5arms:child,parent,remainder,joint');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};prior=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True);child={i:v.cuda() for i,v in prior['fields'].items()};parents={};ctx={};measures=torch.zeros(160,5,2,dtype=torch.float64);field_errors=[];count=0
 def hook(module,args,output):
  i=ctx['i'];arm=ctx['arm']
  if arm==0:
   parents[i]=(routing(args[0],p,1)@(args[0].double()@p['current_value_reader'])[...,None])[...,0];return output
  c=child[i];parent=parents[i];remainder=parent-c;field=[c,parent,remainder,c+remainder][arm-1];field_errors.append(float((c+remainder-parent).norm()/parent.norm().clamp_min(1e-30)));return output[0]-(field[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handle=model.transformer.h[9].attn.register_forward_hook(hook)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:handle.remove()
 assert count==800
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measures[:96,:2],prior['measures'][:96]),rel(measures[96:,:2],prior['measures'][96:])];joint=[rel(measures[:96,4],measures[:96,2]),rel(measures[96:,4],measures[96:,2])];cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];ec=z[:,1]-z[:,0];ep=z[:,2]-z[:,0];er=z[:,3]-z[:,0];discrepancy=ec+er-ep;cell=dict(cell=label,error_relative_child=float(discrepancy.norm()/ec.norm().clamp_min(1e-30)),error_relative_parent=float(discrepancy.norm()/ep.norm().clamp_min(1e-30)),child_effect_norm=float(ec.norm()),parent_effect_norm=float(ep.norm()),remainder_effect_norm=float(er.norm()),child_effect_given_remainder_cosine=float(torch.nn.functional.cosine_similarity(ec,ep-er,dim=0)),maxabs_parent_effect=float(ep.abs().max()),maxabs_child_effect=float(ec.abs().max()))
  if lo<96:
   native=z[::2,0]-z[1::2,0];cell['parent_contrast_damage']=float(((z[::2,0]-z[1::2,0])-(z[::2,2]-z[1::2,2])).mean()/native.mean())
  cells.append(cell)
 A=max(replay)<=1e-4;B=A and max(joint)<=1e-5 and max(field_errors)<=1e-10;C=B and all(c['error_relative_child']<=.1 for c in cells);result={'pred_a':A,'pred_b':B,'pred_c':C,'anchor_replay':replay,'joint_parent_replay':joint,'max_field_identity_error':max(field_errors),'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Parent=currentvalue selectedhead9;child=allsourcecrossfirst;remainder=parent-child. Physicalsame-output hierarchicalremovals onreused160prefixes. Small-child denominator forcomposition. No semanticremainderidentification orautonomousmodel.'}
 torch.save(dict(measures=measures,parent_fields={i:v.cpu() for i,v in parents.items()}),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
