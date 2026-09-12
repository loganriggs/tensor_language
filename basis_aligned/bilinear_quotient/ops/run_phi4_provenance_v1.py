#!/usr/bin/env python3
# BQGATE:192bodyforwards;96prefixes<=26tokens;120seconds;no fitting.
"""pred_a old/fresh native/donor replay<=1e-4relative.
pred_b mode/source sum reproduces actual donor scalarfield<=1e-10relative.
pred_c cue-token-only final donorwrite error<=.2 each4groups.
Null: source city token alone doesnot localize the computed value donation.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='PHI4_PROVENANCE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items())
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('192bodyforwards96prefixes;small mode/routing cache anddonor replay');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen={k:v.cuda() for k,v in torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True).items()};gain=gen['lambdas'][0];ctx={};donors={};nmax=max(len(r['ids']) for r in rows)
 modes=torch.zeros(96,nmax,4,dtype=torch.float64);rho=torch.zeros(96,nmax,dtype=torch.float64);gam=torch.zeros(96,nmax,nmax,dtype=torch.float64);ds=torch.zeros_like(rho);reg=torch.zeros(96,2,2,dtype=torch.float64);count=0
 def pre8(module,args):ctx['modes']=(args[0].double()@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  i=ctx['i'];n=args[0].shape[1];gamma=routing(args[0],p,1)
  if ctx['arm']==0:
   donors[i]=ctx['modes'].sum(-1).clone();modes[i,:n]=ctx['modes'][0].cpu();rho[i,:n]=ctx['rho'][0].cpu();gam[i,:n,:n]=gamma[0].cpu();return output
  delta=(donors[rows[i]['donor_id']]-ctx['modes'].sum(-1))*gain/ctx['rho'];s=(gamma@delta[...,None])[...,0];ds[i,:n]=s[0].cpu()
  return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].mlp.register_forward_pre_hook(pre8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for arm in range(2):
   for i,row in enumerate(rows):
    ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,arm=arm);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==192
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 ref=torch.load(P/'MLP8_VALUE_FRESH_V1_ARTIFACT.pt',weights_only=True)['regional'][:,[0,2]];replay=dict(old=rel(reg[:24],ref[:24]),fresh=rel(reg[24:],ref[24:]));checks=[];contrib=torch.zeros(96,nmax,4,dtype=torch.float64);cityonly=[];fullfinal=[];cuepos=[]
 for i,r in enumerate(rows):
  n=len(r['ids']);j=r['donor_id'];cue=[k for k,(a,b) in enumerate(zip(r['ids'],rows[j]['ids'])) if a!=b];assert len(cue)==1;cuepos.append(cue[0]);dm=(modes[j,:n]-modes[i,:n])*float(gain)/rho[i,:n,None];prediction=torch.einsum('ts,sm->tm',gam[i,:n,:n],dm).sum(-1);checks.append(rel(prediction,ds[i,:n]));contrib[i,:n]=gam[i,n-1,:n,None]*dm;cityonly.append(contrib[i,cue[0]].sum());fullfinal.append(ds[i,n-1])
 cityonly=torch.stack(cityonly);fullfinal=torch.stack(fullfinal);records=[]
 for group in range(4):
  ix=list(range(group*24,(group+1)*24));records.append(dict(group=group,city_token_final_error=rel(cityonly[ix],fullfinal[ix]),full_final_norm=float(fullfinal[ix].norm())))
 A=max(replay.values())<=1e-4;B=A and max(checks)<=1e-10;C=B and all(r['city_token_final_error']<=.2 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay':replay,'max_mode_source_sum_error':max(checks),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Native mode/routing provenance and exact donor replay; cue-token-only computational hypothesis. No causal single-source deletion or newtext confirmation.'}
 torch.save(dict(regional=reg,modes=modes,rho9=rho,gamma=gam,donor_scalar=ds,final_source_mode_contributions=contrib,cue_positions=torch.tensor(cuepos)),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
