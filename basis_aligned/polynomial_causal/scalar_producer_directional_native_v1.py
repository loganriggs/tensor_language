"""Native fidelity of explicit conditional directional MLP runtime."""
import json,time,torch
import torch.nn.functional as F
from pathlib import Path
from directional_mlp_bridge_v1 import execute
from compiled_scalar_producers_v1 import head_scalar
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_PRODUCER_DIRECTIONAL_NATIVE_V1_RESULT.json';assert not out.exists()
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows']
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');program=torch.load(P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 native_weights=[state['transformer.h.8.mlp.'+k+'.weight'] for k in ('Left','Right','Down')];scale=float(state['transformer.h.9.lambdas'][0]);eps=torch.finfo(torch.float32).eps
 predictions=torch.zeros(2,48,22,dtype=torch.float64);baseline=torch.zeros(48,22,dtype=torch.float64);nums=torch.zeros(2,2,dtype=torch.float64);dens=torch.zeros(2,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);z0=cache['z8'][i,:n];r0=cache['r9'][0,i,:n];actual=cache['r9'][1,i,:n].double()-r0.double();tokens=torch.tensor([row['ids']]);family=row['family'];dens[family]+=actual.square().sum()
  baseline[i,:n]=head_scalar(F.rms_norm(r0,(1152,),eps=eps)[None],tokens,p,1)[0]
  for arm,dtype in enumerate((torch.float64,torch.float32)):
   z=z0.to(dtype);L,R,D=[w.to(dtype) for w in native_weights];x=F.rms_norm(z,(1152,),eps=eps);u=((x@L.T)*(x@R.T))@D.T
   delta=scale*execute(z,u,cache['a8'][i,:n,None].to(dtype),program);nums[arm,family]+=(delta.double()-actual).square().sum()
   current=F.rms_norm((r0.double()+delta.double()).float(),(1152,),eps=eps)[None];predictions[arm,i,:n]=head_scalar(current,tokens,p,1)[0]
 cells=[]
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];truth=cache['a9'][1,ix]-cache['a9'][0,ix]
  cells.append(dict(family=family,residual_change_errors=(nums[:,family]/dens[family]).sqrt().tolist(),scalar_change_errors=[rel(predictions[a,ix]-baseline[ix],truth) for a in range(2)],baseline_scalar_replay=rel(baseline[ix],cache['a9'][0,ix])))
 result={'pred_a':all(c['baseline_scalar_replay']<=1e-5 for c in cells),'pred_b':all(c['residual_change_errors'][0]<=1e-4 and c['scalar_change_errors'][0]<=1e-4 for c in cells),'pred_c':all(c['residual_change_errors'][1]<=1e-4 and c['scalar_change_errors'][1]<=1e-4 for c in cells),'cells':cells,'arms':['fp64','fp32'],'seconds':time.perf_counter()-tic,'scope':'Exact fixed-direction runtime with supplied native biasfree MLP background; native-state numerical fidelity, not independent input closure or task adoption.'}
 out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(predictions=predictions),P/'SCALAR_PRODUCER_DIRECTIONAL_NATIVE_V1_ARTIFACT.pt');print(json.dumps(result))
if __name__=='__main__':main()
