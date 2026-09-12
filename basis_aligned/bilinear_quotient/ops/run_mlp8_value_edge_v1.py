#!/usr/bin/env python3
# BQGATE:432bodyforwards;72prefixes;6arms;120seconds;4frozenweightmodes.
"""pred_a scalarpathsum<=1e-5 and nativecapability .2mean,10/12positive.
pred_b quad4/fullquad paired-effect error<=5%nativecontrastnorm eachfamily.
pred_c B pluscityquad4coverage>=.1,10/12positives,control<=.5 and absnationality<=.5city.
Null: value-contrast reconstruction fails to yield the predicted native path effect.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP8_VALUE_EDGE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'];validate(rows);assert len(rows)==72
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('432bodyforwards72rows6valueedgearms120seconds');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('value_edge_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};gen={k:v.cuda() for k,v in torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True).items()};r=gen['reader'];gain=gen['lambdas'][0];context={};checks=[];reg=torch.zeros(72,6,2,dtype=torch.float64);count=0
 def pre8(module,args):context['quad4']=((args[0].double()@gen['eigenvectors'][:,:4]).square()*gen['eigenvalues'][:4]).sum(-1)
 def post8(module,args,output):context['quadfull']=(output-module.Down_bias).double()@r
 def pre9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];context['rho9']=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def head9(module,args,output):
  gamma=routing(args[0],p,1);value=args[0].double()@r;fullq=gain*context['quadfull']/context['rho9'];q4=gain*context['quad4']/context['rho9'];direct=value-fullq;rem=fullq-q4;parts=[value,direct,fullq,q4,rem];ss=[(gamma@v[...,None])[...,0] for v in parts]
  if context['arm']==0:
   native=runtime.scalar(args[0],context['tokens'],p,1);checks.append(float((ss[1]+ss[3]+ss[4]-native).norm()/native.norm().clamp_min(1e-30)));return output
  s=ss[context['arm']-1];return output[0]-(s[...,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].mlp.register_forward_pre_hook(pre8),model.transformer.h[8].mlp.register_forward_hook(post8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(head9)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');context['tokens']=ids
   for arm in range(6):
    context['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==432;records=[]
 for family in range(3):
  ix=[i for i,rw in enumerate(rows) if rw['family']==family];z=reg[ix];c=z[::2,0,0]-z[1::2,0,0];arms=[];reds=[]
  for arm in range(1,6):
   e=z[:,arm,0]-z[:,0,0];red=e[1::2]-e[::2];reds.append(red);arms.append(dict(arm=arm,coverage=float(red.mean()/c.mean()),positive_pairs=int((red>0).sum()),unrelated_ratio=float((z[:,arm,1]-z[:,0,1]).abs().mean()/e.abs().mean().clamp_min(1e-30))))
  records.append(dict(family=family,capable=float(c.mean())>=.2 and int((c>0).sum())>=10,quad4_fullquad_error_over_native=float((reds[3]-reds[2]).norm()/c.norm()),arms=arms))
 A=max(checks)<=1e-5 and all(v['capable'] for v in records);B=A and all(v['quad4_fullquad_error_over_native']<=.05 for v in records);city=records[0]['arms'][3];nat=records[1]['arms'][3];C=B and city['coverage']>=.1 and city['positive_pairs']>=10 and city['unrelated_ratio']<=.5 and abs(nat['coverage'])<=.5*city['coverage'];result={'pred_a':A,'pred_b':B,'pred_c':C,'max_path_sum_error':max(checks),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'arm_names':['native','selected9value','direct_reentry_bias','full_MLP8_quadratic','fourmode_quadratic','quadratic_remainder'],'scope':'Physicalhead9value-edge removals; nativeQKroute/background retained. Fourmodesfrozenfromweights. Newendpointpanelreused, no newline/autonomouspredictionclaim.'};torch.save(dict(regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
