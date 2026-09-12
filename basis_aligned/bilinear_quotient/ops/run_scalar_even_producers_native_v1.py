#!/usr/bin/env python3
# BQGATE:208bodyforwards;104prefixes<=247tokens;120seconds;no fitting.
"""pred_a packagedscalar<=1e-5relative and nativeoutputs<=1e-4relative.
pred_b EACH3regionalfamily baseline-subtracted removal effect<=1e-4relative.
pred_c EACH2newlinehalf CEremoval effect<=1e-4relative.
Null: compact conditional package fails native intervention replay.
Price208bodyforwards104rows2arms,120seconds,no fitting.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from scalar_joint_key_paths_v1 import paths
STEM='SCALAR_EVEN_PRODUCERS_NATIVE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 data=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text());regional=data['regional'];natural=data['natural'];assert len(regional)==72 and len(natural)==32
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('208bodyforwards104rows native/removal conditionalpackage replay');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('even_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();program={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};old={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()};bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'].cuda();context={};field_error=[];count=0
 def hook(index,args,output):
  s=runtime.scalar(args[0],context['tokens'],program,index);reference=paths(args[0],context['tokens'],old,index,bands[index])[...,[0,4,5,6,7,8,9],1 if index==0 else 0].sum(-1);field_error.append(float((s-reference).norm()/reference.norm().clamp_min(1e-30)))
  if context['arm']==0:return output
  return output[0]-(s[...,None]*program['writers'][index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[l].attn.register_forward_hook(lambda m,a,o,i=i:hook(i,a,o)) for i,l in enumerate((8,9))]
 def forward(row,arm):
  nonlocal count
  tokens=torch.tensor([row['ids']],device='cuda');context.update(tokens=tokens,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
 reg=torch.zeros(72,2,2,dtype=torch.float64);ce=torch.zeros(32,2,dtype=torch.float64)
 try:
  for i,row in enumerate(regional):
   for arm in range(2):
    logits=forward(row,arm);reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
  for i,row in enumerate(natural):
   for arm in range(2):ce[i,arm]=-forward(row,arm).log_softmax(-1)[198].cpu()
 finally:
  for h in handles:h.remove()
 assert count==208
 prior=torch.load(P/'SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');rr=prior['regional'];nn=prior['newline_ce']
 def comparison(a,b):
  norm=float(b.norm());diff=a-b;informative=norm>1e-8;rel=float(diff.norm()/max(norm,1e-30));absolute=float(diff.abs().max());return dict(reference_norm=norm,informative=informative,relative_error=rel,maxabs_error=absolute,passed=rel<=1e-4 if informative else absolute<=1e-6)
 baseline=[comparison(reg[:,0],rr[:,0]),comparison(ce[:,0],nn[:,0])];rcells=[];ncells=[]
 for family in range(3):
  ix=[i for i,r in enumerate(regional) if r['family']==family];rcells.append(dict(family=family,**comparison(reg[ix,1]-reg[ix,0],rr[ix,3]-rr[ix,0])))
 for family in range(2):
  ix=[i for i,r in enumerate(natural) if r['family']==family];ncells.append(dict(family=family,**comparison(ce[ix,1]-ce[ix,0],nn[ix,1]-nn[ix,0])))
 A=max(field_error)<=1e-5 and all(c['passed'] for c in baseline)
 result={'pred_a':A,'pred_b':A and all(c['passed'] for c in rcells),'pred_c':A and all(c['passed'] for c in ncells),'max_scalar_relative_error':max(field_error),'baseline':baseline,'regional':rcells,'newline':ncells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Packaged conditionalreflection-even scalar producer native/removal replay. No new independentbehavioralconfirmation or fullmodelinputclosure.','source_shas':binding}
 torch.save(dict(regional=reg,newline_ce=ce),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
