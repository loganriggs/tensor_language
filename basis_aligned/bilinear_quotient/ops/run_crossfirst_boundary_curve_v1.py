#!/usr/bin/env python3
# BQGATE:2080bodyforwards;160prefixes<=248tokens;180seconds;causal boundary curve.
"""pred_a prior firstfour and boundary9/boundary17 outcomes <=1e-4 relative.
pred_b some boundary<=12 leaves<=.5 original final interaction in allfourregionalcells.
pred_c regional remaining-norm boundary curves nonincreasing with .01 original-interaction slack.
Null: interactions remain distributed or cancellation prevents monotone causal boundary interpretation.
Price2080forwards180seconds; existing160prefixes; oracle states fromN/C/P/R, no datafit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='CROSSFIRST_BOUNDARY_CURVE_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2080forwards160rows13arms; native four plus postblock9-17 additive state resets');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,13,2,dtype=torch.float64);count=0
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:return output
  amplitude=child[i] if arm==1 else parent[i]-child[i] if arm==3 else parent[i]
  return output[0]-(amplitude[...,None]*w).to(output[0].dtype),output[1]
 def boundary(layer):
  def hook(module,args,output):
   arm=ctx['arm'];x=output[0]
   if arm<4:ctx['states'].setdefault(layer,{})[arm]=x.double();return output
   if layer==arm+5:
    s=ctx['states'][layer];return (s[1]+s[3]-s[0]).to(x.dtype),output[1]
   return output
  return hook
 handles=[model.transformer.h[9].attn.register_forward_hook(write9)]+[model.transformer.h[k].register_forward_hook(boundary(k)) for k in range(9,18)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i;ctx['states']={}
   for arm in range(13):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==2080
 precision=torch.load(P/'MLP9_CROSSFIRST_PRECISION_V1_ARTIFACT.pt',weights_only=True)['measures'];readout=torch.load(P/'CROSSFIRST_READOUT_SPLIT_V1_ARTIFACT.pt',weights_only=True)['additive_readout_measures']
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[]
 for lo,hi in [(0,96),(96,160)]:replay.extend([rel(measures[lo:hi,:4],old['measures'][lo:hi,:4]),rel(measures[lo:hi,4],precision[lo:hi,5]),rel(measures[lo:hi,12],readout[lo:hi])])
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];total=z[:,2]-z[:,1]-z[:,3]+z[:,0];base=z[:,1]+z[:,3]-z[:,0];left=z[:,4:]-base[:,None]
  cells.append(dict(cell=label,remaining_over_total=[float(x.norm()/total.norm().clamp_min(1e-30)) for x in left.T],remaining_over_child=[float(x.norm()/(z[:,1]-z[:,0]).norm().clamp_min(1e-30)) for x in left.T],aligned_remaining=[float((x*total).sum()/total.square().sum().clamp_min(1e-30)) for x in left.T]))
 early=[k+9 for k in range(4) if all(c['remaining_over_total'][k]<=.5 for c in cells[:4])];monotone=all(all(b<=a+.01 for a,b in zip(c['remaining_over_total'],c['remaining_over_total'][1:])) for c in cells[:4])
 result={'pred_a':max(replay)<=1e-4,'pred_b':bool(early),'pred_c':monotone,'anchor_replay':replay,'early_half_boundaries':early,'boundaries':list(range(9,18)),'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Causal reset of accumulated mixed state at each boundary. Not decomposition of locally generated interaction; borrowed native states and suffix, reused rows, no fitting.'}
 torch.save(dict(measures=measures),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
