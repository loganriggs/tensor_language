from pathlib import Path
import json,time,math,torch,sys
from check_overlap_varpro import fixture
from shared_subspace_loss import SharedSubspaceLoss
P=Path(__file__).parent;torch.set_num_threads(2)
version=sys.argv[1] if len(sys.argv)>1 else 'V1'
plan=json.loads((P/f'SHARED_SUBSPACE_TOY_PLAN_{version}.json').read_text());rows=[];t0=time.monotonic()
for case in range(plan['cases']):
 T,_,truth=fixture(case,mixed=True);metric=SharedSubspaceLoss(T)
 for seed in plan['seeds']:
  rng=torch.Generator().manual_seed(28000+seed)
  params=[torch.nn.Parameter(torch.randn(truth[i].shape,dtype=T.dtype,generator=rng)/len(T[0])**.5) for i in (0,2,4,6)]
  opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf')
  for step in range(plan['steps']+1):
   opt.zero_grad();loss=metric.loss(params);v=float(loss.detach());assert math.isfinite(v)
   if v<best:best=v;saved=[p.detach().clone() for p in params]
   if step==plan['steps']:break
   opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
  with torch.no_grad():
   explicit=float(metric.loss(saved,explicit=True));assert abs(explicit-best)<1e-8
  row=dict(case=case,seed=seed,error=explicit**.5);rows.append(row);print(json.dumps(row),flush=True)
errors=[min(r['error'] for r in rows if r['case']==i) for i in range(plan['cases'])];n=sum(e<=plan['threshold'] for e in errors)
out=dict(plan=plan,records=rows,best_restart_errors=errors,recovered_cases=n,recovery_pass=n>=plan['required_cases'],seconds=time.monotonic()-t0,scope='Random shared/private subspace learning with dense quadratic cores eliminated exactly. No product topology, native fit, sparsity or circuit claim.')
(P/f'SHARED_SUBSPACE_TOY_{version}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
