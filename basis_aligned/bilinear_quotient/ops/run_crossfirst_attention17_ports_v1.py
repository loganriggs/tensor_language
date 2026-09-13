#!/usr/bin/env python3
# BQGATE:800bodyforwards;800finalreadouts;160prefixes<=248tokens;180seconds.
"""pred_a priorfive/base/head2 anchors<=1e-4 and jointportmixedwrite<=.02 eachpanel.
pred_b cross-edit productgroup effect<=.20 relative head17.2 effect eachregionalcell.
pred_c crossgroup+defectgroup effects sum<=.05 fullcompiled effect eachregionalcell.
Null: inherited nonlinear port changes dominate rather than products between additive changes.
Price800fullforwards+800finalreadouts;160existingprefixes;180seconds;nofit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from joint_attention_mixed_ports_v1 import decompose
STEM='CROSSFIRST_ATTENTION17_PORTS_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800forwards+800finalreadouts; joint score1/score2/value cross-versus-defect');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,5,2,dtype=torch.float64);statechecks=[];readouts=torch.zeros(160,5,2,dtype=torch.float64);count=0;W=model.transformer.h[17].attn.c_proj.weight.double().reshape(1152,9,128);headwrites=torch.zeros(160,9,1152,dtype=torch.float64);headinputs=torch.zeros(160,9,128,dtype=torch.float64)
 native_attention=model.transformer.h[17].attn.squared_attention;allports=[]
 def capture_attention(q,k,v,q2,k2):
  scores=torch.einsum('bthd,bshd->bhts',q,k)/q.shape[-1];scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/q.shape[-1]
  ctx['ports'][ctx['arm']]=(scores[:,2,-1].double(),scores2[:,2,-1].double(),v[:,:,2].double())
  return native_attention(q,k,v,q2,k2)
 model.transformer.h[17].attn.squared_attention=capture_attention
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
  y=ctx['head_inputs'];delta=y[4]-y[1]-y[3]+y[0];ref=(delta[2]@W[:,2,:].T)[None];ports=ctx['ports'];parts=decompose(ports[0],ports[1],ports[3],ports[4]);cross=parts['cross']@W[:,2,:].T;defect=parts['defect']@W[:,2,:].T;full=cross+defect
  statechecks.append(dict(error_sq=float((full-ref).square().sum()),reference_sq=float(ref.square().sum())))
  ctx['readout_states']=[bar,bar+ref,bar+full,bar+cross,bar+defect];allports.append({key:tuple(v.cpu() for v in ports[key]) for key in (0,1,3,4)});return output

 handles=[model.transformer.h[17].attn.c_proj.register_forward_pre_hook(before_proj),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[16].register_forward_hook(after16),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.register_forward_hook(att17),model.transformer.h[17].register_forward_hook(after17)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,h16={},h17={},z17={},head_inputs={},ports={})
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
  model.transformer.h[17].attn.squared_attention=native_attention
  for h in handles:h.remove()
 assert count==800 and len(statechecks)==160
 prior=torch.load(P/'CROSSFIRST_ATTENTION17_HEADS_V1_ARTIFACT.pt',weights_only=True);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[];local=[]
 for lo,hi in [(0,96),(96,160)]:
  replay.extend([rel(measures[lo:hi],prior['measures'][lo:hi]),rel(readouts[lo:hi,0],prior['readouts'][lo:hi,0]),rel(readouts[lo:hi,1],prior['readouts'][lo:hi,5])]);s=statechecks[lo:hi];local.append((sum(x['error_sq'] for x in s)/max(sum(x['reference_sq'] for x in s),1e-30))**.5)
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=readouts[lo:hi,:,0];e=z-z[:,0:1];ref=e[:,1];full=e[:,2];cross=e[:,3];defect=e[:,4]
  cells.append(dict(cell=label,formula_error=rel(full,ref),cross_error=rel(cross,ref),defect_error=rel(defect,ref),composition_error=rel(cross+defect,full),cross_aligned=float((cross*ref).sum()/ref.square().sum().clamp_min(1e-30)),defect_aligned=float((defect*ref).sum()/ref.square().sum().clamp_min(1e-30)),reference_norm=float(ref.norm()),formula_maxabs=float((full-ref).abs().max())))
 result={'pred_a':max(replay)<=1e-4 and max(local)<=.02,'pred_b':all(c['cross_error']<=.2 for c in cells[:4]),'pred_c':all(c['composition_error']<=.05 for c in cells[:4]),'anchor_replay':replay,'joint_port_state_error':local,'cells':cells,'body_forwards':count,'extra_final_readouts':800,'seconds':time.perf_counter()-tic,'scope':'Nativehead17.2 source-summed QK1*QK2*value finite mixed expansion. Cross-edit versus inherited portmixed inputs; conditional borrowedports/background, no fitting or OODclaim.'}
 torch.save(dict(measures=measures,readouts=readouts,ports=allports,local_stats=statechecks),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
