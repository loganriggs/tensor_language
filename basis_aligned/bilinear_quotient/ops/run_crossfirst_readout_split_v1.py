#!/usr/bin/env python3
# BQGATE:640bodyforwards;160prefixes<=248tokens;180seconds;readout interaction split.
"""pred_a prior4armoutcomes<=1e-4, exactinteractionpartition<=1e-10relative.
pred_b internalstateinteractioneffect<=.2totalDOD each8cells.
pred_c additivefinalstate with exactreadout predictsparent<=.1child effecteachcell.
Null: suffix interactions require morethan finalreadout correction.
Price640forwards180sec+160finalreadouts; no datafit ornewOOD.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from regional_even_routing_v1 import routing
from sparse_path_stability_atlas_v1 import digest
STEM='CROSSFIRST_READOUT_SPLIT_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regrows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regrows);rows=regrows+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('640forwards160rows4arms; final-state/readout interaction split');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};prior=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True);child={i:v.cuda() for i,v in prior['fields'].items()};parents={};ctx={};measures=torch.zeros(160,4,2,dtype=torch.float64);final_states=torch.zeros(160,4,1152,dtype=torch.float32,device='cuda');field_errors=[];count=0
 def hook(module,args,output):
  i=ctx['i'];arm=ctx['arm']
  if arm==0:
   parents[i]=(routing(args[0],p,1)@(args[0].double()@p['current_value_reader'])[...,None])[...,0];return output
  c=child[i];parent=parents[i];remainder=parent-c;field=[c,parent,remainder,c+remainder][arm-1];field_errors.append(float((c+remainder-parent).norm()/parent.norm().clamp_min(1e-30)));return output[0]-(field[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handle=model.transformer.h[9].attn.register_forward_hook(hook)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(4):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    final_states[i,arm]=x[0,-1];logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:handle.remove()
 assert count==640
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True)['measures'][:,:4];replay=[rel(measures[:96],old[:96]),rel(measures[96:],old[96:])];additive=(final_states[:,1].double()+final_states[:,3].double()-final_states[:,0].double()).float();add_measures=torch.zeros(160,2,dtype=torch.float64)
 for i,row in enumerate(rows):
  logits=30*torch.tanh(model.lm_head(F.rms_norm(additive[i:i+1],(1152,)))/30);logits=logits[0]
  if i<96:add_measures[i,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();add_measures[i,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
  else:add_measures[i,0]=-logits.log_softmax(-1)[198].cpu();add_measures[i,1]=(logits[198]-logits[11]).cpu()
 total=measures[:,2]-measures[:,1]-measures[:,3]+measures[:,0];internal=measures[:,2]-add_measures;readout=add_measures-measures[:,1]-measures[:,3]+measures[:,0];partition=rel(internal+readout,total);cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  t=total[lo:hi,0];inter=internal[lo:hi,0];read=readout[lo:hi,0];child=measures[lo:hi,1,0]-measures[lo:hi,0,0];h=final_states[lo:hi].double();hc=h[:,1]-h[:,0];hm=h[:,2]-h[:,1]-h[:,3]+h[:,0];cells.append(dict(cell=label,total_interaction_norm=float(t.norm()),internal_over_total=float(inter.norm()/t.norm().clamp_min(1e-30)),readout_over_total=float(read.norm()/t.norm().clamp_min(1e-30)),readout_aligned_fraction=float((read*t).sum()/t.square().sum().clamp_min(1e-30)),corrected_error_over_child=float(inter.norm()/child.norm().clamp_min(1e-30)),uncorrected_error_over_child=float(t.norm()/child.norm().clamp_min(1e-30)),final_state_mixed_over_child=float(hm.norm()/hc.norm().clamp_min(1e-30))))
 A=max(replay)<=1e-4 and partition<=1e-10;B=A and all(c['internal_over_total']<=.2 for c in cells);C=A and all(c['corrected_error_over_child']<=.1 for c in cells);result={'pred_a':A,'pred_b':B,'pred_c':C,'anchor_replay':replay,'interaction_partition_error':partition,'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Exactreadout onadditivefinal-state hypothesis, splitting totalDOD into internalstateinteraction andfinalreadoutnonlinearity. Nativeinterventionstates borrowed; no fit,newOOD orautonomouspredictor.'}
 torch.save(dict(measures=measures,additive_readout_measures=add_measures,final_states=final_states.cpu()),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
