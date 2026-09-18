#!/usr/bin/env python3
# BQGATE:40bodyforwards;40prefixes;120seconds;no fitting.
"""pred_a native replay abs/rel<=1e-5; pred_b folded readers rel<=1e-4;
pred_c finite40fixtures/forwards and literal storage reduced. Native-input certificate.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from city_mlp7_readers_v1 import execute
STEM='CITY_MLP7_READERS_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_NO_MODEL') or os.environ.get('BQLIB_DRYRUN'):
  print('40bodyforwards;40prefixes;three folded MLP7 readers');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(120)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();program={k:v.cuda() for k,v in torch.load(P/(STEM+'_PROGRAM.pt'),weights_only=True).items()}
 head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8'];W=torch.cat([head[k] for k in ['k1','k2','current_value']]).cuda()
 fixtures=[];state={};errors=[]
 def post7(module,args,result):
  z=args[0][:,state['city']];native=F.linear(result,W)[:,state['city']];folded=execute(program,z)
  errors.append(float((folded.double()-native.double()).norm()/native.double().norm()))
  fixtures.append({'normalized_city':z.cpu(),'native_readers':native.cpu(),'folded_readers':folded.cpu()})
 handle=model.transformer.h[7].mlp.register_forward_hook(post7)
 values=torch.zeros(1,40,10,dtype=torch.float64);count=0
 try:
  for i,row in enumerate(groups):
   state['city']=row['city_position'];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0)
   scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
   for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[0,i,j]=(scores[left]-scores[right]).cpu()
   count+=1
 finally:handle.remove()
 v=expand(values,mapping,6);old=torch.load(P/'CITY_FULL_PILE_V3_ARTIFACT.pt',weights_only=True)['values'][0];diff=v[0]-old
 absolute=float(diff.abs().max());relative=float(diff.norm()/old.norm())
 got=torch.cat([f['folded_readers'] for f in fixtures]).double();expected=torch.cat([f['native_readers'] for f in fixtures]).double()
 per_reader=[float((g-e).norm()/e.norm()) for g,e in zip(got.split(128,-1),expected.split(128,-1))]
 price=json.loads((P/(STEM+'_CPU_RESULT.json')).read_text())
 r={'pred_a':absolute<=1e-5 and relative<=1e-5,'pred_b':max(errors)<=1e-4 and max(per_reader)<=1e-4,
    'pred_c':count==40 and len(fixtures)==40 and bool(torch.isfinite(v).all()) and bool(torch.isfinite(got).all()) and bool(torch.isfinite(expected).all()) and price['floating_scalars']<price['unfolded_floating_scalars'] and price['fp32_bytes']<price['unfolded_fp32_bytes'],
    'reader_relative_errors':per_reader,'max_sequence_relative_error':max(errors),'replay_max_abs':absolute,'replay_relative':relative,'body_forwards':count,'seconds':time.perf_counter()-start,'floating_scalars':price['floating_scalars'],'fp32_bytes':price['fp32_bytes'],'scope':'Opened local reader certificate at normalized MLP7 city input; no downstream substitution, OOD or closed-input claim','source_shas':binding}
 torch.save({'fixtures':fixtures,'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='source_shas'}));signal.alarm(0)
if __name__=='__main__':main()
