"""Frozen finite-amplitude MLP8 bridge on actual native circuit states."""
import json,time,torch
import torch.nn.functional as F
from pathlib import Path
from compiled_scalar_producers_v1 import head_scalar
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter()
 out=P/'SCALAR_PRODUCER_MLP_BRIDGE_NATIVE_V1_RESULT.json';assert not out.exists()
 receipt=json.loads((P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_RESULT.json').read_text());assert all(receipt[k] for k in ('pred_a','pred_b','pred_c'))
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows']
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files']
 state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'][0].double()
 L,R,D=[state['transformer.h.8.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
 scale=float(state['transformer.h.9.lambdas'][0]);eps=torch.finfo(torch.float32).eps
 ld=L@d;rd=R@d;quadratic=(ld*rd)@D.T
 predictions=torch.zeros(6,48,22,dtype=torch.float64);residual_error=[0.,0.];residual_norm=[0.,0.];term_norms=torch.zeros(2,4,dtype=torch.float64);perturb_ratios=[]
 for i,row in enumerate(rows):
  n=len(row['ids']);z=cache['z8'][i,:n].double();a=cache['a8'][i,:n,None];base9=cache['r9'][0,i,:n].double();changed9=cache['r9'][1,i,:n].double();zm=z-a*d
  rho=z.square().mean(-1,keepdim=True)+eps;rhom=zm.square().mean(-1,keepdim=True)+eps;zl=z@L.T;zr=z@R.T
  background=(zl*zr)@D.T;cross=(zl*rd+zr*ld)@D.T
  terms=scale*torch.stack([-a*d,(rhom.reciprocal()-rho.reciprocal())*background,-a/rhom*cross,a.square()/rhom*quadratic])
  full=terms.sum(0);actual=changed9-base9;family=row['family']
  residual_error[family]+=float((full-actual).square().sum());residual_norm[family]+=float(actual.square().sum());term_norms[family]+=terms.square().sum((1,2))
  perturb_ratios.extend((a[:,0].abs()*d.norm()/z.norm(dim=-1)).tolist())
  tokens=torch.tensor([row['ids']])
  deltas=[torch.zeros_like(full),full,terms[0]+terms[2],full-terms[1],full-terms[3],terms[0]]
  for arm,delta in enumerate(deltas):
   current=F.rms_norm((base9+delta).float(),(1152,),eps=eps)[None]
   predictions[arm,i,:n]=head_scalar(current,tokens,p,1)[0]
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];truth=cache['a9'][1,ix]-cache['a9'][0,ix]
  effects=predictions[:,ix]-predictions[0,ix][None];errors=[rel(effects[arm],truth) for arm in range(1,6)]
  cells.append(dict(family=family,exact_residual_change_error=(residual_error[family]/residual_norm[family])**.5,baseline_scalar_replay=rel(predictions[0,ix],cache['a9'][0,ix]),scalar_change_errors=errors,actual_scalar_change_norm=float(truth.norm()),actual_residual_change_norm=residual_norm[family]**.5,raw_term_to_full_delta_norm=(term_norms[family]/residual_norm[family]).sqrt().tolist()))
 A=all(c['exact_residual_change_error']<=1e-4 and c['baseline_scalar_replay']<=1e-5 and c['scalar_change_errors'][0]<=1e-4 for c in cells)
 result={'pred_a':A,'pred_b':A and all(c['scalar_change_errors'][1]<=.1 for c in cells),'pred_c':A and all(c['scalar_change_errors'][4]>=.25 for c in cells),'cells':cells,'arms':['native','exact','direct_plus_mixed','omit_norm_background','omit_quadratic','direct_only'],'term_order':['direct','norm_background','mixed','quadratic'],'perturbation_ratio_min_median_max':[min(perturb_ratios),float(torch.tensor(perturb_ratios).median()),max(perturb_ratios)],'seconds':time.perf_counter()-tic,'scope':'Conditional prediction of actual head9 scalar change through MLP8 on reused native rows. No fitting, end-task adoption or new OOD. Norms and native z8/a8/r9 inputs remain explicit.'}
 torch.save(dict(predictions=predictions),P/'SCALAR_PRODUCER_MLP_BRIDGE_NATIVE_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
