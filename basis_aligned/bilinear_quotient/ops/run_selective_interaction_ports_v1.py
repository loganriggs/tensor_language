#!/usr/bin/env python3
# BQGATE:144bodyforwards;72newcueprefixes;native/selectedhead8removal;120seconds.
"""pred_a baseline<=1e-4; routing/scalar and symmetric port replay<=1e-10.
pred_b value-only baseline-routing prediction<=10% fullscalar response eachfamily.
pred_c frozenfourmode valueportprediction<=5% eachfamily; nativebaseline9normheld.
Null: value path alone insufficient or four modes fail actualselectiveinteraction.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='SELECTIVE_INTERACTION_PORTS_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==72
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('144bodyforwards; actualselective head8removal; 72newcue rows');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('ports_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};context={};length=max(len(r['ids']) for r in rows);z8=torch.zeros(2,72,length,1152);raw9=torch.zeros_like(z8);value=torch.zeros(2,72,length,dtype=torch.float64);gamma=torch.zeros(2,72,length,length,dtype=torch.float64);scalar=torch.zeros_like(value);amp=torch.zeros_like(value);margins=torch.zeros(72,2,2,dtype=torch.float64);checks=[];count=0
 def before8(module,args):context['r8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def head8(module,args,output):
  state,i,n=context['state'],context['row'],args[0].shape[1];s=runtime.scalar(args[0],context['tokens'],p,0);amp[state,i,:n]=s[0].cpu();edited=output[0] if state==0 else output[0]-(s[...,None]*p['writers'][0]).to(output[0].dtype);z8[state,i,:n]=(context['r8']+edited)[0].cpu();return edited,output[1]
 def before9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];raw9[context['state'],context['row'],:raw.shape[1]]=raw[0].cpu()
 def head9(module,args,output):
  state,i,n=context['state'],context['row'],args[0].shape[1];g=routing(args[0],p,1);v=args[0].double()@p['current_value_reader'];s=runtime.scalar(args[0],context['tokens'],p,1);check=(g@v[...,None])[...,0];checks.append(float((check-s).norm()/s.norm().clamp_min(1e-30)));gamma[state,i,:n,:n]=g[0].cpu();value[state,i,:n]=v[0].cpu();scalar[state,i,:n]=s[0].cpu();return output
 handles=[model.transformer.h[8].register_forward_pre_hook(before8),model.transformer.h[8].attn.register_forward_hook(head8),model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for state in [0,1]:
   for i,row in enumerate(rows):
    ids=torch.tensor([row['ids']],device='cuda');context.update(state=state,row=i,tokens=ids);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;margins[i,state,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();margins[i,state,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for handle in handles:handle.remove()
 assert count==144
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));old=torch.load(P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ARTIFACT.pt',weights_only=True)['regional'][:,0];baseline=rel(margins[:,0],old);delta=scalar[1]-scalar[0];vg=((gamma[1]+gamma[0])/2@(value[1]-value[0])[...,None])[...,0];gg=((gamma[1]-gamma[0])@((value[1]+value[0])/2)[...,None])[...,0];only=(gamma[0]@(value[1]-value[0])[...,None])[...,0]
 gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);zz=z8.double();rho=zz.square().mean(-1)+torch.finfo(torch.float32).eps;proj=zz@gen['eigenvectors'][:,:4];num=gen['lambdas'][0]*((zz@gen['reader'])+(proj.square()*gen['eigenvalues'][:4]).sum(-1)/rho);den0=(raw9[0].square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double();dvhat=(num[1]-num[0])/den0;pred=((gamma[1]+gamma[0])/2@dvhat[...,None])[...,0];records=[];split=[]
 for family in range(3):
  ix=[i for i,r in enumerate(rows) if r['family']==family];ts=[len(rows[i]['ids'])-1 for i in ix];d=delta[ix,ts];v=vg[ix,ts];g=gg[ix,ts];e=only[ix,ts];vp=pred[ix,ts];split.append(rel(v+g,d));records.append(dict(family=family,name=rows[ix[0]]['family_name'],response_norm=float(d.norm()),valueport_norm=float(v.norm()),value_aligned_fraction=float((v@d)/d.square().sum()),routing_aligned_fraction=float((g@d)/d.square().sum()),value_norm_ratio=float(v.norm()/d.norm()),routing_norm_ratio=float(g.norm()/d.norm()),value_only_error=rel(e,d),rank4_valueport_error=rel(vp,v),value_only_pass=float(d.norm())>1e-8 and rel(e,d)<=.1,rank4_valueport_pass=float(v.norm())>1e-8 and rel(vp,v)<=.05))
 A=baseline<=1e-4 and max(checks)<=1e-10 and max(split)<=1e-10;result={'pred_a':A,'pred_b':A and all(r['value_only_pass'] for r in records),'pred_c':A and all(r['rank4_valueport_pass'] for r in records),'baseline_relative_error':baseline,'max_scalar_replay_error':max(checks),'max_symmetric_split_error':max(split),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Newcuepanel, actualrank64head8removal; head9observed. Conditionalvalue/routingport decomposition; rank4predictor holds baseline9norm but uses nativez8states andaveragerouting. Not completehead/logit or independentinputgenerator prediction.'}
 torch.save(dict(z8=z8,raw9=raw9,value=value,gamma=gamma,scalar=scalar,amplitude8=amp,margins=margins),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
