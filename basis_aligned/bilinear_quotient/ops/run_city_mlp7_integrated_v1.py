#!/usr/bin/env python3
# BQGATE:80bodyforwards;40prefixes;120seconds;no fitting.
"""pred_a native replay abs/rel<=1e-5/localwrite<=1e-4;
pred_b installed abs<=1e-4/rel<=1e-5 and effects<=1e-3 each;
pred_c direct reader certificate and literal price.80forwards.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_MLP7_INTEGRATED_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('80bodyforwards;40prefixes;native/folded-city-removal');return
 certificate=json.loads((P/'CITY_MLP7_READERS_V1_RESULT.json').read_text())
 assert all(certificate[k] for k in ['pred_a','pred_b','pred_c']), 'Native reader prerequisite failed'
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(120)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval()
 fixtures=torch.load(P/(STEM+'_CPU_ARTIFACT.pt'),weights_only=True)['fixtures'];writes=[f['delta'].cuda() for f in fixtures]
 local=max(float((f['delta']-f['expected_native_delta']).norm()/f['expected_native_delta'].norm()) for f in fixtures)
 state={}
 def post8(module,args,out):
  return (out[0]+writes[state['i']].to(out[0].dtype),out[1]) if state['arm'] else out
 handle=model.transformer.h[8].attn.register_forward_hook(post8)
 values=torch.zeros(2,40,10,dtype=torch.float64);count=0
 try:
  for arm in range(2):
   state['arm']=arm
   for i,row in enumerate(groups):
    state['i']=i;ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    for block in model.transformer.h:x,first=block(x,first,x0)
    scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=(scores[left]-scores[right]).cpu()
    count+=1
 finally:handle.remove()
 v=expand(values,mapping,6);old=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['values'][:2]
 absolute=[float((v[i]-old[i]).abs().max()) for i in range(2)];relative=[float((v[i]-old[i]).norm()/old[i].norm()) for i in range(2)]
 actual=old[1]-old[0];got=v[1]-v[0];errors=((got-actual).square().sum(0).sqrt()/actual.square().sum(0).sqrt()).tolist()
 r={'pred_a':absolute[0]<=1e-5 and relative[0]<=1e-5 and local<=1e-4 and bool(torch.isfinite(v).all()) and count==80,
    'pred_b':absolute[1]<=1e-4 and relative[1]<=1e-5 and max(errors)<=1e-3,
    'pred_c':all(certificate[k] for k in ['pred_a','pred_b','pred_c']) and certificate['floating_scalars']==12386688 and certificate['fp32_bytes']==49546752,
    'max_abs_by_arm':absolute,'relative_by_arm':relative,'effect_errors_by_reader':errors,'local_write_error':local,'body_forwards':count,'seconds':time.perf_counter()-start,
    'scope':'Opened installed folded-reader write; reconstructed MLP7 input and supplied native query/context/RMS; not full port closure or composition confirmation','reader_certificate_sha256':hashlib.sha256((P/'CITY_MLP7_READERS_V1_RESULT.json').read_bytes()).hexdigest(),'source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'}));signal.alarm(0)
if __name__=='__main__':main()
