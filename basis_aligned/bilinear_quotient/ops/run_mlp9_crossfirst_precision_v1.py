#!/usr/bin/env python3
# BQGATE:960bodyforwards;160prefixes<=248tokens;180seconds;MLP9 precision discriminator.
"""pred_a prior five-arm replay <=1e-4 relative each panel.
pred_b exact-native versus predicted mixed correction discrepancy <=.01 of original final interaction each8cells.
pred_c native mixed correction leaves >.5 original interaction norm each8cells.
Null: local rounding errors amplify enough to invalidate the prior mediation miss.
Price960fullforwards160existingprefixes180seconds. Six frozen arms, no fitting.
Descriptive block10-17 attention/MLP mixed-state census is not causal localization.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='MLP9_CROSSFIRST_PRECISION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('960forwards160rows6arms;predicted versus native mixed correction; suffix census');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();prog={k:v.cuda() for k,v in torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True).items()};w=prog['direction'];J=prog['mixed_map'];Jw=J@w;child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,6,2,dtype=torch.float64);stats=[];census=[];count=0
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
  return (x.double()-(ctx['mix'] if arm==4 else mix)).to(x.dtype),output[1]
 handles=[model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[9].mlp.register_forward_hook(mlp9),model.transformer.h[9].register_forward_hook(after9)]
 def capture(label):
  def hook(module,args,output):
   arm=ctx['arm']
   if arm>3:return
   x=(output[0] if isinstance(output,tuple) else output).double()
   ctx['states'].setdefault(label,{})[arm]=x
  return hook
 for layer in range(9,18):
  handles.append(model.transformer.h[layer].register_forward_hook(capture(('block',layer))))
  if layer>9:
   handles.append(model.transformer.h[layer].attn.register_forward_hook(capture(('attn',layer))))
   handles.append(model.transformer.h[layer].mlp.register_forward_hook(capture(('mlp',layer))))
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i;ctx['states']={}
   for arm in range(6):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
   def mixed(label):
    z=ctx['states'][label];return z[2]-z[1]-z[3]+z[0]
   rowstats=[]
   for layer in range(10,18):
    before=mixed(('block',layer-1));after=mixed(('block',layer));att=mixed(('attn',layer));mlp=mixed(('mlp',layer));transport=model.transformer.h[layer].lambdas[0].double()*before
    rowstats.append(dict(layer=layer,before_sq=float(before.square().sum()),after_sq=float(after.square().sum()),attn_sq=float(att.square().sum()),mlp_sq=float(mlp.square().sum()),increment_sq=float((att+mlp).square().sum()),accounting_error_sq=float((after-transport-att-mlp).square().sum()),attn_mlp_dot=float((att*mlp).sum())))
   census.append(rowstats)
 finally:
  for h in handles:h.remove()
 assert count==960 and len(stats)==320
 old_response=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_ARTIFACT.pt',weights_only=True)['measures']
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 replay=[rel(measures[lo:hi,:5],old_response[lo:hi]) for lo,hi in [(0,96),(96,160)]];cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];total=z[:,2]-z[:,1]-z[:,3]+z[:,0];left=z[:,5]-z[:,1]-z[:,3]+z[:,0]
  cells.append(dict(cell=label,precision_effect_over_total=float((z[:,5]-z[:,4]).norm()/total.norm().clamp_min(1e-30)),native_remaining_over_total=float(left.norm()/total.norm().clamp_min(1e-30))))
 result={'pred_a':max(replay)<=1e-4,'pred_b':all(c['precision_effect_over_total']<=.01 for c in cells),'pred_c':all(c['native_remaining_over_total']>.5 for c in cells),'anchor_replay':replay,'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Native-versus-formula local mixed-state correction; reused160rows, descriptive suffix census, no fitting or causal site promotion.'}
 torch.save(dict(measures=measures,local_stats=stats,census=census),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
