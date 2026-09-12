#!/usr/bin/env python3
# BQGATE:672bodyforwards;80prefixes;7keycuts plus native/mean control;120seconds.
"""pred_a prior native/64/fullsector/mean replay<=1e-4 and endpoint0/256<=1e-5.
pred_b BOTH56/72 preserve original regionalcoverage/sign/specificity and NLbars.
pred_c B plus48/80 also pass and all four neighbor effects<=10% relative to64.
Null: chosen64boundary is fragile; no threshold change or fitted factors.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='SCALAR_KEY_BOUNDARY_V1';CUTS=[0,48,56,64,72,80,256];FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 regional=json.loads((P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json').read_text())['rows'];natural=[r for r in json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text())['rows'] if r['pool']=='fineweb'];validate(regional);assert len(regional)==48 and len(natural)==32
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('672bodyforwards;80rows;frozen nested keycuts;120seconds');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('boundary_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'].cuda();basis=bands.permute(0,2,1,3).reshape(2,1152,256);programs=[]
 for cut in CUTS:z=dict(p);z['key_basis']=basis[:,:,:cut];programs.append(z)
 prior=torch.load(P/'SCALAR_JOINT_KEY_GRADES_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');mean=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True,map_location='cpu')['mean_head'].cuda();context={};count=0
 def pre(module,args):context['preov8']=args[0]
 def hook(index,module,args,output):
  arm=context['arm']
  if arm==0:return output
  if arm==8:
   if index==1:return output
   O=module.c_proj.weight[:,256:384];native=context['preov8'][...,256:384];return output[0]-F.linear(native,O)+F.linear(mean,O)[None,None],output[1]
  scalar=runtime.scalar(args[0],context['tokens'],programs[arm-1],index);return output[0]-(scalar[...,None]*p['writers'][index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[8].attn.c_proj.register_forward_pre_hook(pre)]
 for i,l in enumerate([8,9]):handles.append(model.transformer.h[l].attn.register_forward_hook(lambda m,a,o,i=i:hook(i,m,a,o)))
 def forward(row,arm):
  nonlocal count
  ids=torch.tensor([row['ids']],device='cuda');context.update(arm=arm,tokens=ids);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 reg=torch.zeros(48,8,2,dtype=torch.float64);ce=torch.zeros(32,9,dtype=torch.float64);margin=torch.zeros(32,dtype=torch.float64)
 try:
  for i,row in enumerate(regional):
   for arm in range(8):
    logits=forward(row,arm);reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
  for i,row in enumerate(natural):
   for arm in range(9):
    logits=forward(row,arm);ce[i,arm]=-logits.log_softmax(-1)[198].cpu()
    if arm==0:margin[i]=(logits[198]-logits[11]).cpu()
 finally:
  for handle in handles:handle.remove()
 assert count==672
 def relative(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 oldr=prior['regional'];oldn=prior['newline_ce'];checks=[relative(reg[:,0],oldr[:,0]),relative(ce[:,0],oldn[:,0]),relative(reg[:,4]-reg[:,0],oldr[:,45]-oldr[:,0]),relative(ce[:,4]-ce[:,0],oldn[:,45]-oldn[:,0]),relative(reg[:,1]-reg[:,0],oldr[:,63]-oldr[:,0]),relative(ce[:,1]-ce[:,0],oldn[:,63]-oldn[:,0]),relative(ce[:,8]-ce[:,0],oldn[:,65]-oldn[:,0])];endpoints=max(relative(reg[:,7]-reg[:,0],reg[:,1]-reg[:,0]),relative(ce[:,7]-ce[:,0],ce[:,1]-ce[:,0]));records=[];cap=[]
 for family in [0,1]:
  ri=[i for i,r in enumerate(regional) if r['family']==family];ni=[i for i,r in enumerate(natural) if r['family']==family];contrast=reg[ri,0,0][::2]-reg[ri,0,0][1::2];ctrl=ce[ni,8]-ce[ni,0];cap.append(bool(contrast.mean()>=.2 and (contrast>0).sum()>=10 and margin[ni].mean()>=.2 and (margin[ni]>0).sum()>=12 and ce[ni,0].mean()<=5 and ctrl.mean()>=.02 and (ctrl>0).sum()>=8))
 for arm,cut in enumerate(CUTS,1):
  rc=[];nc=[]
  for family in [0,1]:
   ri=[i for i,r in enumerate(regional) if r['family']==family];ni=[i for i,r in enumerate(natural) if r['family']==family];contrast=reg[ri,0,0][::2]-reg[ri,0,0][1::2];effect=reg[ri,arm,0]-reg[ri,0,0];reduction=effect[1::2]-effect[::2];coverage=float(reduction.mean()/contrast.mean());positive=int((reduction>0).sum());ratio=float((reg[ri,arm,1]-reg[ri,0,1]).abs().mean()/effect.abs().mean().clamp_min(1e-30));delta=ce[ni,arm]-ce[ni,0];ref=reg[ri,4,0]-reg[ri,0,0]
   rc.append(dict(family=family,coverage=coverage,positive_pairs=positive,unrelated_ratio=ratio,effect_error_vs64=relative(effect,ref),passed=coverage>=.5 and positive>=10 and ratio<=.5));nc.append(dict(family=family,meanabs=float(delta.abs().mean()),maxabs=float(delta.abs().max()),passed=float(delta.abs().mean())<=.02 and float(delta.abs().max())<=.1))
  records.append(dict(cut=cut,regional=rc,newline=nc,joint_pass=all(z['passed'] for z in rc+nc)))
 A=max(checks)<=1e-4 and endpoints<=1e-5 and all(cap);B=A and all(r['joint_pass'] for r in records if r['cut'] in [56,72]);neighbors=[r for r in records if r['cut'] in [48,56,72,80]]
 result={'pred_a':A,'pred_b':B,'pred_c':B and all(r['joint_pass'] and max(z['effect_error_vs64'] for z in r['regional'])<=.1 for r in neighbors),'prior_replay_errors':checks,'endpoint_relative_error':endpoints,'capability':cap,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Frozen nested keyspace boundaries on existingregional/FineWeb panels including earlieroutlier; coupledQK maintained, originalnorms andsequentialnativeinputs retained. Boundaryrobustness screen, not newOOD or roleidentification; any alternatecut requires freshconfirmation.'}
 torch.save(dict(regional=reg,newline_ce=ce),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
