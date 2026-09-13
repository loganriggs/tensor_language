#!/usr/bin/env python3
# BQGATE:800bodyforwards;1920finalreadouts;160prefixes<=248tokens;180seconds.
"""pred_a priorfive/base/direct anchors<=1e-4 and headsumstate<=.02 eachpanel.
pred_b fixedknownhead17.2 direct effect<=.20 relative full direct effect eachregionalcell.
pred_c separate nine-head effects sum<=.05 relative jointheadsum eachregionalcell.
Null: distributed heads or a different source path rather than the known head17.2 component.
Price800fullforwards+1920finalreadouts;160existingprefixes;180seconds;nofit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='CROSSFIRST_ATTENTION17_HEADS_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800forwards+1920finalreadouts; native attention17 mixed write byhead');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,5,2,dtype=torch.float64);statechecks=[];readouts=torch.zeros(160,12,2,dtype=torch.float64);count=0;W=model.transformer.h[17].attn.c_proj.weight.double().reshape(1152,9,128);headwrites=torch.zeros(160,9,1152,dtype=torch.float64);headinputs=torch.zeros(160,9,128,dtype=torch.float64)
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:return output
  amplitude=child[i] if arm==1 else parent[i]-child[i] if arm==3 else parent[i]
  return output[0]-(amplitude[...,None]*w).to(output[0].dtype),output[1]
 def after16(module,args,output):
  arm=ctx['arm']
  if arm<4:ctx['h16'][arm]=output[0].double()
  if arm==4:
   s=ctx['h16'];return (s[1]+s[3]-s[0]).to(output[0].dtype),output[1]
  return output
 def pre17(module,args):ctx['raw17']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def att17(module,args,output):
  if ctx['arm']<5:ctx['z17'][ctx['arm']]=(ctx['raw17']+output[0]).double()
 def before_proj(module,args):
  if ctx['arm']<5:ctx['head_inputs'][ctx['arm']]=args[0][0,-1].double().reshape(9,128)
 def after17(module,args,output):
  arm=ctx['arm'];x=output[0]
  if arm<4:ctx['h17'][arm]=x.double();return output
  z=ctx['z17'];s=ctx['h17'];bar=(s[1]+s[3]-s[0])[:,-1];zbar=(z[1]+z[3]-z[0]).to(x.dtype);v=(z[4]-zbar.double())[:,-1]
  y=ctx['head_inputs'];delta=y[4]-y[1]-y[3]+y[0];writes=torch.einsum('hi,ohi->ho',delta,W);summed=writes.sum(0,keepdim=True);headwrites[ctx['i']]=writes.cpu();headinputs[ctx['i']]=delta.cpu()
  statechecks.append(dict(error_sq=float((summed-v).square().sum()),reference_sq=float(v.square().sum())))
  ctx['readout_states']=[bar,bar+v,bar+summed]+[bar+writes[j:j+1] for j in range(9)];return output
 handles=[model.transformer.h[17].attn.c_proj.register_forward_pre_hook(before_proj),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[16].register_forward_hook(after16),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.register_forward_hook(att17),model.transformer.h[17].register_forward_hook(after17)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,h16={},h17={},z17={},head_inputs={})
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
   for j,state in enumerate(ctx['readout_states']):
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(state.float(),(1152,)))/30))[0]
    if i<96:readouts[i,j,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();readouts[i,j,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:readouts[i,j,0]=-logits.log_softmax(-1)[198].cpu();readouts[i,j,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==800 and len(statechecks)==160
 prior=torch.load(P/'CROSSFIRST_LAST_ATTENTION_TERMS_V1_ARTIFACT.pt',weights_only=True);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[];local=[]
 for lo,hi in [(0,96),(96,160)]:
  replay.extend([rel(measures[lo:hi],prior['measures'][lo:hi]),rel(readouts[lo:hi,0],prior['readouts'][lo:hi,0]),rel(readouts[lo:hi,1],prior['readouts'][lo:hi,3])]);s=statechecks[lo:hi];local.append((sum(x['error_sq'] for x in s)/max(sum(x['reference_sq'] for x in s),1e-30))**.5)
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=readouts[lo:hi,:,0];e=z-z[:,0:1];ref=e[:,1];joint=e[:,2];heads=e[:,3:];comp=heads.sum(1)-joint
  cells.append(dict(cell=label,head2_error=rel(heads[:,2],ref),sumwrite_effect_error=rel(joint,ref),composition_error=float(comp.norm()/joint.norm().clamp_min(1e-30)),head_errors=[rel(heads[:,h],ref) for h in range(9)],head_aligned=[float((heads[:,h]*ref).sum()/ref.square().sum().clamp_min(1e-30)) for h in range(9)],reference_norm=float(ref.norm()),maxabs_composition=float(comp.abs().max())))
 result={'pred_a':max(replay)<=1e-4 and max(local)<=.02,'pred_b':all(c['head2_error']<=.2 for c in cells[:4]),'pred_c':all(c['composition_error']<=.05 for c in cells[:4]),'anchor_replay':replay,'headsum_state_error':local,'cells':cells,'body_forwards':count,'extra_final_readouts':1920,'seconds':time.perf_counter()-tic,'scope':'Native headwise direct mixed attention write at additive background. Fixedhead17.2 hypothesis fromprior dossier; otherhead ranking descriptive, reusedrows,no fit. MixedQK/value inputs remainnative.'}
 torch.save(dict(measures=measures,readouts=readouts,headwrites=headwrites,headinputs=headinputs,local_stats=statechecks),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
