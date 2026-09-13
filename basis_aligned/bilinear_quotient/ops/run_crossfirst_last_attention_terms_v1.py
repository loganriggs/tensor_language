#!/usr/bin/env python3
# BQGATE:800bodyforwards;160extraMLP17;1280finalreadouts;160prefixes<=248tokens;180seconds.
"""pred_a nativefive/base/refattention anchors<=1e-4 and localformularesponse<=.01 eachpanel.
pred_b direct+cross final effect error<=.20 relative native attentionbranch eachregionalcell.
pred_c direct+cross+normalizer final effect error<=.10 eachregionalcell.
Null: normalization/square response or output nonlinearity prevents a simple cross-term program.
Price800fullforwards+160MLP17calls+1280finalreadouts;160existingprefixes;180seconds;nofit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='CROSSFIRST_LAST_ATTENTION_TERMS_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800forwards+160MLP17+1280finalreadouts; exact attention response term comparison');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,5,2,dtype=torch.float64);statechecks=[];readouts=torch.zeros(160,8,2,dtype=torch.float64);count=0;ml=model.transformer.h[17].mlp;L=ml.Left.weight.double();R=ml.Right.weight.double();D=ml.Down.weight.double()
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
 def after17(module,args,output):
  arm=ctx['arm'];x=output[0]
  if arm<4:ctx['h17'][arm]=x.double();return output
  if arm==4:
   z=ctx['z17'];s=ctx['h17'];bar=s[1]+s[3]-s[0];zbar=(z[1]+z[3]-z[0]).to(x.dtype)
   gbar=(zbar+module.mlp(F.rms_norm(zbar,(1152,)))).double();gm=gbar-bar;ga=x.double()-gbar;ctx['bar']=bar;ctx['gm']=gm;ctx['ga']=ga
   zz=zbar.double();v=z[4]-zz;rho=zz.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;newrho=(zz+v).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
   lz=zz@L.T;rz=zz@R.T;lv=v@L.T;rv=v@R.T;base=(lz*rz)@D.T/rho;cross=(lz*rv+lv*rz)@D.T/newrho;square=(lv*rv)@D.T/newrho;norm=(rho/newrho-1)*base;full=v+cross+square+norm
   statechecks.append(dict(error_sq=float((full-ga).square().sum()),reference_sq=float(ga.square().sum())))
   ctx['readout_states']=[bar,bar+ga,bar+full,bar+v,bar+v+cross,bar+v+cross+norm,bar+norm,bar+square];return output
  target=ctx['bar']+(ctx['gm'] if arm==6 else ctx['ga'] if arm==7 else 0)
  return target.to(x.dtype),output[1]
 handles=[model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[16].register_forward_hook(after16),model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.register_forward_hook(att17),model.transformer.h[17].register_forward_hook(after17)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,h16={},h17={},z17={})
   for arm in range(5):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
   for j,state in enumerate(ctx['readout_states']):
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1].float(),(1152,)))/30))[0]
    if i<96:readouts[i,j,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();readouts[i,j,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:readouts[i,j,0]=-logits.log_softmax(-1)[198].cpu();readouts[i,j,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==800 and len(statechecks)==160
 prior=torch.load(P/'CROSSFIRST_LAST_BLOCK_SPLIT_V1_ARTIFACT.pt',weights_only=True)['measures'];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[];local=[]
 for lo,hi in [(0,96),(96,160)]:
  replay.extend([rel(measures[lo:hi,:5],prior[lo:hi,:5]),rel(readouts[lo:hi,0],prior[lo:hi,5]),rel(readouts[lo:hi,1],prior[lo:hi,7])]);s=statechecks[lo:hi];local.append((sum(x['error_sq'] for x in s)/max(sum(x['reference_sq'] for x in s),1e-30))**.5)
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=readouts[lo:hi,:,0];effects=z-z[:,0:1];ref=effects[:,1];errors={name:rel(effects[:,j],ref) for j,name in enumerate(['base','reference','formula','direct','direct_cross','direct_cross_norm','norm','square']) if j>1};aligned={name:float((effects[:,j]*ref).sum()/ref.square().sum().clamp_min(1e-30)) for j,name in [(3,'direct'),(4,'direct_cross'),(5,'direct_cross_norm'),(6,'norm'),(7,'square')]}
  cells.append(dict(cell=label,errors=errors,aligned=aligned,reference_norm=float(ref.norm()),formula_maxabs=float((effects[:,2]-ref).abs().max())))
 result={'pred_a':max(replay)<=1e-4 and max(local)<=.01,'pred_b':all(c['errors']['direct_cross']<=.2 for c in cells[:4]),'pred_c':all(c['errors']['direct_cross_norm']<=.1 for c in cells[:4]),'anchor_replay':replay,'local_formula_error':local,'cells':cells,'body_forwards':count,'extra_MLP17_evaluations':160,'extra_final_readouts':1280,'seconds':time.perf_counter()-tic,'scope':'Native conditional attention response terms at additive finalstate background. No fitting, OOD or standalone predictor; full MLP weights and native states charged.'}
 torch.save(dict(measures=measures,readouts=readouts,local_stats=statechecks),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
