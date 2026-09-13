#!/usr/bin/env python3
# BQGATE:800bodyforwards;160prefixes<=248tokens;180seconds;MLP9 mixed-state mediation.
"""pred_a nativefourarm outcomes<=1e-4priorrelative.
pred_b individualpostMLP9response aggregateerror<=.01 eachpanel.
pred_c removingpredictedMLP9mixedstate leaves<=.5totalfinalDOD each8cells.
Null: subsequentlayers generate mostnonadditivity despiteexactlocalresponse.
Price800forwards180sec; nativebaselineMLP9output/stateports, no datafit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='MLP9_CROSSFIRST_RESPONSE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800forwards160rows5arms;nativeMLP9 response andmixedstate mediation');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();prog={k:v.cuda() for k,v in torch.load(P/(STEM+'_PROGRAM.pt'),weights_only=True).items()};w=prog['direction'];J=prog['mixed_map'];Jw=J@w;child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,5,2,dtype=torch.float64);stats=[];count=0
 def pre9(module,args):
  if ctx['arm']==0:ctx['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:ctx['z']=ctx['raw9']+output[0];return output
  amplitude=child[i] if arm==1 else parent[i]-child[i] if arm==3 else parent[i]
  return output[0]-(amplitude[...,None]*w).to(output[0].dtype),output[1]
 def mlp9(module,args,output):
  if ctx['arm']==0:
   z=ctx['z'].double();base=(output-module.Down_bias).double();rho=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;Jz=z@J.T;pred={}
   for arm,a in [(1,child[ctx['i']]),(2,parent[ctx['i']]),(3,parent[ctx['i']]-child[ctx['i']])]:
    a=a[...,None];rh=(z-a*w).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;pred[arm]=-a*w+(rho/rh-1)*base-a/rh*Jz+a.square()/(2*rh)*Jw
   ctx['pred']=pred;ctx['mix']=pred[2]-pred[1]-pred[3]
 def after9(module,args,output):
  arm=ctx['arm'];x=output[0]
  if arm==0:ctx['post0']=x.double();ctx['actual']={};return output
  if arm in (1,2,3):ctx['actual'][arm]=x.double()-ctx['post0'];return output
  actual=ctx['actual'];pred=ctx['pred'];mix=actual[2]-actual[1]-actual[3];numer=sum(float((pred[k]-actual[k]).square().sum()) for k in (1,2,3));denom=sum(float(actual[k].square().sum()) for k in (1,2,3));stats.append(dict(row=ctx['i'],response_error_squared=numer,response_norm_squared=denom,mixed_error_squared=float((ctx['mix']-mix).square().sum()),mixed_norm_squared=float(mix.square().sum()),child_norm_squared=float(actual[1].square().sum())))
  return (x.double()-ctx['mix']).to(x.dtype),output[1]
 handles=[model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[9].mlp.register_forward_hook(mlp9),model.transformer.h[9].register_forward_hook(after9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==800 and len(stats)==160
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measures[:96,:4],old['measures'][:96,:4]),rel(measures[96:,:4],old['measures'][96:,:4])];panelerrors=[(sum(s['response_error_squared'] for s in stats[lo:hi])/max(sum(s['response_norm_squared'] for s in stats[lo:hi]),1e-30))**.5 for lo,hi in [(0,96),(96,160)]];cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];total=z[:,2]-z[:,1]-z[:,3]+z[:,0];left=z[:,4]-z[:,1]-z[:,3]+z[:,0];mediated=z[:,2]-z[:,4];st=stats[lo:hi];num=sum(x['mixed_error_squared'] for x in st);cells.append(dict(cell=label,remaining_over_total=float(left.norm()/total.norm().clamp_min(1e-30)),mediated_aligned_fraction=float((mediated*total).sum()/total.square().sum().clamp_min(1e-30)),remaining_over_child=float(left.norm()/(z[:,1]-z[:,0]).norm().clamp_min(1e-30)),mixed_state_prediction_error=(num/max(sum(x['mixed_norm_squared'] for x in st),1e-30))**.5,mixed_state_error_over_child=(num/max(sum(x['child_norm_squared'] for x in st),1e-30))**.5))
 A=max(replay)<=1e-4;B=A and max(panelerrors)<=.01;C=B and all(c['remaining_over_total']<=.5 for c in cells);result={'pred_a':A,'pred_b':B,'pred_c':C,'anchor_replay':replay,'local_response_errors':panelerrors,'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'ExactconditionalMLP9fixedwriter response checkedagainstnativeFP32; onlypredictedMLP9mixedstate removedfromcombinedparenttrajectory beforelaternativeblocks. Contextandbaselineoutputsupplied; reusedprefixes,nofit.'}
 torch.save(dict(measures=measures,local_stats=stats),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
