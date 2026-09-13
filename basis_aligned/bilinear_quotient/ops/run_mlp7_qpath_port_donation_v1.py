#!/usr/bin/env python3
# BQGATE:768bodyforwards;96prefixes<=26tokens;180seconds;frozen rawparent input ports.
"""pred_a native/fullpostQpath replay<=1e-4relative.
pred_b nearpostS-only donor error<=.25relative fullpath andnegative>=20/24.
pred_c A andnearpostQ-only error>=.5 andS-onlybeatsQ-only.
Null: partnerdominance in cachedfinalwrite failsphysicalinput-port intervention.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP7_QPATH_PORT_DONATION_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('768bodyforwards96rows8arms; rawparent one-port swaps');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gain=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)['lambdas'][0].cuda();cache=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);nmax=cache['phi_native'].shape[1];parents=cache['parents'].cuda();r8=cache['rho8_squared'].cuda();eig=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)['eigenvalues'][:4].cuda();values=torch.zeros(96,nmax,7,dtype=torch.float64,device='cuda');ctx={};reg=torch.zeros(96,8,2,dtype=torch.float64);fields=torch.zeros(96,7,nmax,dtype=torch.float64);count=0
 # Inputs are Q,B,H,R; swap masks full, Q, B, H, B+H, R, Q+B+H.
 masks=[15,1,2,4,6,8,7]
 for i,row in enumerate(rows):
  j=row['donor_id'];n=len(row['ids']);own=parents[i,:n];don=parents[j,:n];q0=own[:,1];s0=own[:,0]+own[:,2];base=((q0.square()+2*q0*s0)*eig).sum(-1)/r8[i,:n]
  for k,mask in enumerate(masks):
   q=don[:,1] if mask&1 else own[:,1];b=don[:,0] if mask&2 else own[:,0];h=don[:,2] if mask&4 else own[:,2];rr=r8[j,:n] if mask&8 else r8[i,:n];values[i,:n,k]=((q.square()+2*q*(b+h))*eig).sum(-1)/rr-base

 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  arm=ctx['arm'];i=ctx['i'];n=args[0].shape[1]
  if arm==0:return output
  pos=int(prov['cue_positions'][i]);dv=values[i,:n,arm-1][None,:]*gain/ctx['rho'];mask=torch.zeros_like(dv);mask[:,pos+1:]=1
  s=(routing(args[0],p,1)@(dv*mask)[...,None])[...,0];fields[i,arm-1,:n]=s[0].cpu();return output[0]+(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]

 handles=[model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i
   for arm in range(8):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==768
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'MLP7_PHI_PATH_DONATION_V1_ARTIFACT.pt',weights_only=True)['regional'];replay=rel(reg[:,:2],ref[:,[0,5]]);sumerr=rel(fields[:,2]+fields[:,3],fields[:,4]);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];c=z[::2,0,0]-z[1::2,0,0];arms=[];e=[]
  for arm in range(1,8):
   delta=z[:,arm,0]-z[:,0,0];d=delta.clone();d[::2]*=-1;e.append(d);arms.append(dict(arm=arm,transfer=float(d.mean()/c.mean()),positive=int((d>0).sum()),negative=int((d<0).sum()),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/delta.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,arms=arms,relative_errors_to_full=[rel(x,e[0]) for x in e],B_H_composition_error=rel(e[2]+e[3],e[4])))
 A=replay<=1e-4 and sumerr<=1e-10;near=records[2];B=A and near['relative_errors_to_full'][4]<=.25 and near['arms'][4]['negative']>=20;C=A and near['relative_errors_to_full'][1]>=.5 and near['relative_errors_to_full'][4]<near['relative_errors_to_full'][1]
 result={'pred_a':A,'pred_b':B,'pred_c':C,'replay_error':replay,'B_H_scalar_sum_error':sumerr,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','fullpostQpath','Q_only','B_only','H_only','B_plus_H','R_only','Q_plus_B_plus_H'],'scope':'Rawinput port donation into Q-containing generatedvalue; onlypostcitysources, recipientotherports andhead9QK/rho. No globalparentmodule replacement; reused96prefixes.'}
 torch.save(dict(regional=reg,donor_fields=fields),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
