#!/usr/bin/env python3
# BQGATE:96bodyforwards;48prefixes<=22tokens;120seconds;15MBcache;no fitting.
"""pred_a native and physical8 final margin replay<=1e-4relative.
pred_b pristine scalar8/9 replay<=1e-5relative.
pred_c raw-state/component cache is finite and has96bodyforwards.
Null: invalid capture blocks subsequent native bridge inference.
Price96bodyforwards48rows2arms,120seconds,~15MBcache,no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
STEM='SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];assert len(rows)==48 and max(len(r['ids']) for r in rows)==22
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('96bodyforwards48rows; preMLP8/rawpreRMS9/native scalar cache');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();producer={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'][0].cuda()
 z8=torch.zeros(48,22,1152);r9=torch.zeros(2,48,22,1152);a8=torch.zeros(48,22,dtype=torch.float64);a9=torch.zeros(2,48,22,dtype=torch.float64);margins=torch.zeros(48,2,2,dtype=torch.float64);context={}
 def before8(module,args):context['r8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def after8(module,args,out):
  scalar=head_scalar(args[0],context['tokens'],producer,0);i=context['row'];n=args[0].shape[1]
  if context['arm']==0:z8[i,:n]=(context['r8']+out[0])[0].cpu();a8[i,:n]=scalar[0].cpu();return out
  return out[0]-(scalar[...,None]*d).to(out[0].dtype),out[1]
 def before9(module,args):
  raw=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];r9[context['arm'],context['row'],:raw.shape[1]]=raw[0].cpu()
 def after9(module,args,out):a9[context['arm'],context['row'],:args[0].shape[1]]=head_scalar(args[0],context['tokens'],producer,1)[0].cpu()
 handles=[model.transformer.h[8].register_forward_pre_hook(before8),model.transformer.h[8].attn.register_forward_hook(after8),model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].attn.register_forward_hook(after9)]
 count=0
 try:
  for i,row in enumerate(rows):
   tokens=torch.tensor([row['ids']],device='cuda')
   for arm in range(2):
    context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    count+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    margins[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();margins[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 old=torch.load(P/'SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['regional'][:,:2]
 original=torch.load(P/'SCALAR_PRODUCERS_SERIAL_NATIVE_SCALARS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['scalars']
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 checks=dict(margin_replay=rel(margins,old),scalar8_replay=rel(a8,original[0]),scalar9_replay=rel(a9[0],original[1]))
 data=dict(z8=z8,r9=r9,a8=a8,a9=a9,margins=margins)
 result={'pred_a':checks['margin_replay']<=1e-4,'pred_b':max(checks['scalar8_replay'],checks['scalar9_replay'])<=1e-5,'pred_c':count==96 and all(bool(v.isfinite().all()) for v in data.values()),'checks':checks,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Native bridge validation cache, reused regional rows. No factor optimization or causal term sufficiency conclusion.','source_shas':binding}
 torch.save(data,art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
