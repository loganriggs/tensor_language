#!/usr/bin/env python3
# BQGATE:320bodyforwards;160prefixes<=248tokens;180seconds;executor native replay.
"""pred_a native regionalmargin/FineWebCE+margin replay<=1e-4relative.
pred_b assembledfields versus nativehook fields<=1e-4relative eachpanel.
pred_c assembledremoval outcomes versuspriorallsource removal<=1e-4relative.
Null: normalizedstate/rotary/mixing/interface mistake in assembledexecutor.
Price320forwards180sec;three normalizedstates,two norms,nativeprefix/suffix.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from regional_even_routing_v1 import routing
from sparse_path_stability_atlas_v1 import digest
STEM='CROSSFIRST_STATE_EXECUTOR_V1';PACKAGE=P/'extracted_circuits/crossfirst_state_executor_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);natural=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];rows=regional+natural;assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('320forwards160rows; explicit-state field andphysicalremoval replay');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();spec=importlib.util.spec_from_file_location('assembled',PACKAGE/'execute.py');exe=importlib.util.module_from_spec(spec);spec.loader.exec_module(exe);w=exe.load_weights(model.state_dict(),'cuda');parent=torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt',weights_only=True);u=parent['readers'].cuda();C=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True)['head_output_readers'][2].cuda();ctx={};fields={};native_fields={};measures=torch.zeros(160,2,2,dtype=torch.float64);checks=[];att8=model.transformer.h[8].attn;original=att8.squared_attention
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 def mlp7(module,args,output):
  if ctx['arm']==0:ctx['x7']=args[0];ctx['Qref']=(output-module.Down_bias).double()@u*w['lambda8']
 def pre8(module,args):
  if ctx['arm']==0:ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def input8(module,args):
  if ctx['arm']==0:
   x,v1=args;n=x.shape[1];ctx['x8']=x;ctx['Fref']=((module.lamb*v1.view(1,n,9,128))[:,:,2].double()@C.T)
 def joint(q,k,v,q2,k2):
  if ctx['arm']==0:
   n=q.shape[1];a=torch.einsum('btd,bsd->bts',q[:,:,2],k[:,:,2])/128;b=torch.einsum('btd,bsd->bts',q2[:,:,2],k2[:,:,2])/128;ctx['G8ref']=(a*b).masked_fill(~torch.ones(n,n,device=q.device,dtype=torch.bool).tril(),0).double();ctx['Href']=ctx['G8ref']@ctx['Fref']
  return original(q,k,v,q2,k2)
 def out8(module,args,output):
  if ctx['arm']==0:
   z=ctx['raw8']+output[0];ctx['R8']=z.square().mean(-1).double()+torch.finfo(torch.float32).eps
 def pre9(module,args):
  if ctx['arm']==0:
   z=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho9']=(z.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def write9(module,args,output):
  i=ctx['i'];token=w['token'];p=w['routing']
  if ctx['arm']==0:
   candidate=exe.field(ctx['ids'],ctx['x7'],ctx['x8'],args[0],ctx['R8'],ctx['rho9'],w);v=2*token['head9_gain']*(ctx['Qref']*token['eigenvalues']*ctx['Href']).sum(-1)/(ctx['R8']*ctx['rho9']);reference=(routing(args[0],p,1)@v[...,None])[...,0];fields[i]=candidate;native_fields[i]=reference.cpu();x=ctx['x7'].double();q=((x@w['left7'].double().T)*(x@w['right7'].double().T))@w['product_coefficients'].T*w['lambda8'];checks.append(dict(q=rel(q,ctx['Qref']),token=rel(exe._token.token_readings(ctx['ids'],w['embedding'],token),ctx['Fref']),gamma8=rel(exe.full_joint_routing8(ctx['x8'],p),ctx['G8ref']),field=rel(candidate,reference)));return output
  return output[0]-(fields[i][...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[7].mlp.register_forward_hook(mlp7),model.transformer.h[8].register_forward_pre_hook(pre8),att8.register_forward_pre_hook(input8),att8.register_forward_hook(out8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9)];att8.squared_attention=joint;count=0
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,ids=ids)
   for arm in range(2):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:
     measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
  att8.squared_attention=original
 assert count==320
 reg=torch.load(P/'FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_ARTIFACT.pt',weights_only=True)['regional'][:,[0,2]];nl=torch.load(P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ARTIFACT.pt',weights_only=True);nat=torch.stack([nl['ce'][:,:2],nl['margins'][:,:2]],-1);reference=torch.cat([reg,nat],0);replays=[rel(measures[:96,0],reference[:96,0]),rel(measures[96:,0],reference[96:,0])];removal=[rel(measures[:96,1],reference[:96,1]),rel(measures[96:,1],reference[96:,1])];field_errors=[]
 for lo,hi in [(0,96),(96,160)]:field_errors.append(rel(torch.cat([fields[i].flatten().cpu() for i in range(lo,hi)]),torch.cat([native_fields[i].flatten() for i in range(lo,hi)])))
 A=max(replays)<=1e-4;B=A and max(field_errors)<=1e-4;Cpass=B and max(removal)<=1e-4;result={'pred_a':A,'pred_b':B,'pred_c':Cpass,'native_replay_errors':replays,'field_replay_errors':field_errors,'removal_replay_errors':removal,'max_individual_checks':{k:max(c[k] for c in checks) for k in checks[0]},'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Explicit-state assembledexecutor replay160existingprefixes; no cachedQ/routing inputs. Nativeprefix/suffix/stateports retained; head9selectedrouting sharedprimitive. Implementationvalidation, notnewOOD orautonomousextraction.'}
 torch.save(dict(measures=measures,checks=checks,fields={i:v.cpu() for i,v in fields.items()}),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
