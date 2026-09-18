#!/usr/bin/env python3
# BQGATE:40bodyforwards;40prefixes;120seconds;no fitting.
"""pred_a native replay abs/rel<=1e-5, source sum rel<=1e-6;
40bodyforwards, four native city sources; no causal claim or random null.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_SOURCE7_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('40bodyforwards;40prefixes;four city-state sources');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(120)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();state={};fixtures=[]
 def pre7(module,args):state['mixed7']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def attn7(module,args,out):state['attention7']=out[0].clone()
 def mlp7(module,args,out):state['mlp7']=out.clone()
 def pre8(module,args):
  c=state['city'];l=module.lambdas
  state['sources']=torch.stack([l[0]*state[k][:,c] for k in ['mixed7','attention7','mlp7']]+[l[1]*args[2][:,c]])
  state['mixed8']=(l[0]*args[0]+l[1]*args[2])[:,c]
 def attention8(module,args):
  fixtures.append({'sources':state['sources'].cpu(),'mixed8_city':state['mixed8'].cpu(),'current8':args[0].cpu()})
 handles=[model.transformer.h[7].register_forward_pre_hook(pre7),model.transformer.h[7].attn.register_forward_hook(attn7),model.transformer.h[7].mlp.register_forward_hook(mlp7),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_pre_hook(attention8)]
 values=torch.zeros(1,40,10,dtype=torch.float64);count=0
 try:
  for i,row in enumerate(groups):
   state['city']=row['city_position'];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0)
   scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
   for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[0,i,j]=(scores[left]-scores[right]).cpu()
   count+=1
 finally:
  for h in handles:h.remove()
 v=expand(values,mapping,6);old=torch.load(P/'CITY_FULL_PILE_V3_ARTIFACT.pt',weights_only=True)['values'][0]
 diff=v[0]-old;absolute=float(diff.abs().max());relative=float(diff.norm()/old.norm())
 errors=[float((f['sources'].double().sum(0)-f['mixed8_city'].double()).norm()/f['mixed8_city'].double().norm()) for f in fixtures]
 r={'pred_a':absolute<=1e-5 and relative<=1e-5 and max(errors)<=1e-6 and count==40 and bool(torch.isfinite(v).all()),'source_sum_max_relative':max(errors),'replay_max_abs':absolute,'replay_relative':relative,'body_forwards':count,'seconds':time.perf_counter()-start,'scope':'Opened native source capture; no source-causal or port-closure claim','source_shas':binding}
 torch.save({'fixtures':fixtures,'values':v,'source_labels':['mixed7','attention7','mlp7','initial8']},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'}));signal.alarm(0)
if __name__=='__main__':main()
