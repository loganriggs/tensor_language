import json,itertools,time
from pathlib import Path
import torch
from core import coefficients_from_dense,metric,inner
from sweep import fit

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'NATIVE_QUARTIC_PILOT_V1.json';assert not out.exists();a=p.parent.parent/'bilinear_quotient/circuits/followups'
 toys=json.loads((p/'TOY_SWEEP_V1.json').read_text());assert all(min(v['best'] for v in x.values())<1e-3 for x in toys['summary'].values())
 cases=torch.load(a/'native_two_mlp_quartic_ht_v1r1.pt',weights_only=True,map_location='cpu');case=cases[0];target=coefficients_from_dense(case['hessian_not_applicable_quartic'][0].double());normalizer=inner(target,target,metric(5,4)).sqrt();target=target/normalizer
 records=[];saved=[];start=time.perf_counter()
 for kind,width,opt,seed,objective in itertools.product(['tree','dag'],[1,2,4,8],['adam','muon'],[0,1,2],['gaussian','frobenius']):
  row,state=fit(target,5,4,kind,width,opt,.05,seed,1200,metric_kind=objective);records.append(row);saved.append(state)
  print(json.dumps({k:row[k] for k in ['kind','width','optimizer','seed','metric','relative_error','symmetric_frobenius_error']}),flush=True)
  out.write_text(json.dumps(dict(context={k:case[k] for k in ['panel','role','template']},row=0,teacher_normalizer=float(normalizer),records=records,seconds=time.perf_counter()-start,scope='Exact native two-MLP homogeneous branch, fouroutputs/fiveinputs; conditional frames/readers and normalization excluded, no fullmodel claim'),indent=2)+'\n')
 torch.save(dict(target=target,normalizer=normalizer,students=saved),out.with_suffix('.pt'))
if __name__=='__main__':main()
