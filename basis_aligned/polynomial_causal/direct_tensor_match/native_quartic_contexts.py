"""Freeze pilot-selected families; fit each declared weight target independently."""
import json,time,itertools
from pathlib import Path
import torch
from core import coefficients_from_dense,metric,inner
from sweep import fit

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'NATIVE_QUARTIC_CONTEXTS_V1.json';assert not out.exists();a=p.parent.parent/'bilinear_quotient/circuits/followups'
 cases=torch.load(a/'native_two_mlp_quartic_ht_v1r1.pt',weights_only=True,map_location='cpu');rows=[];start=time.perf_counter()
 for case_index,case in enumerate(cases):
  target=coefficients_from_dense(case['hessian_not_applicable_quartic'][0].double());target/=inner(target,target,metric(5,4)).sqrt()
  for kind,width,objective,seed in itertools.product(['tree','dag'],[2,4],['gaussian','frobenius'],[0,1,2]):
   row,_=fit(target,5,4,kind,width,'muon',.05,seed,1200,metric_kind=objective);row.update(case_index=case_index,context={k:case[k] for k in ['panel','role','template']},row=0);rows.append(row)
  print(json.dumps(dict(case=case_index,context=rows[-1]['context'],best_frobenius=min(x['symmetric_frobenius_error'] for x in rows if x['case_index']==case_index),seconds=time.perf_counter()-start)),flush=True)
  out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='16 fixed native weight contexts, first row each,384 independent fits. Families and optimizer selected from toy/pilot; per-context parameters independently optimized, not shared-context extraction.'),indent=2)+'\n')
if __name__=='__main__':main()
