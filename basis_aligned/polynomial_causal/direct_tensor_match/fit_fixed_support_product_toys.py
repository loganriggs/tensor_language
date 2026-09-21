from pathlib import Path
import json,time,math,torch,sys
from fixed_support_products import FixedSupportProducts
from pairwise_product_toy_fixture import fixture
P=Path(__file__).parent;torch.set_num_threads(2)
version=sys.argv[1] if len(sys.argv)>1 else 'V1'
plan=json.loads((P/f'FIXED_SUPPORT_PRODUCT_TOY_PLAN_{version}.json').read_text());rows=[];t0=time.monotonic()
for case in range(5):
 T,bases,private,maps,templates=fixture(case,mixed=plan.get('mixed',False));metric=FixedSupportProducts(T,bases,private,templates)
 shapes=[]
 for j in range(3):shapes.extend([maps[j].shape,(metric.spans[j].shape[1],private[j].shape[1])])
 for seed in plan['seeds']:
  rng=torch.Generator().manual_seed(37000+seed);params=[torch.nn.Parameter(torch.randn(shape,dtype=T.dtype,generator=rng)/shape[0]**.5) for shape in shapes];opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf')
  for step in range(plan['steps']+1):
   opt.zero_grad();loss=metric.loss(params)[0];v=float(loss.detach());assert math.isfinite(v)
   if v<best:best=v;saved=[a.detach().clone() for a in params]
   if step==plan['steps']:break
   opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
  explicit=float(metric.loss(saved,dense=True)[0]);assert abs(explicit-best)<1e-8
  row=dict(case=case,seed=seed,regularized_root_error=max(explicit,0)**.5);rows.append(row);print(json.dumps(row),flush=True)
errors=[min(r['regularized_root_error'] for r in rows if r['case']==i) for i in range(5)];count=sum(e<=plan['threshold'] for e in errors)
out=dict(plan=plan,records=rows,best_restart_errors=errors,recovered_cases=count,recovery_pass=count>=plan['required_cases'],seconds=time.monotonic()-t0,scope='Product coefficients randomly initialized within known fixed shared/pair supports. Establishes stage2recovery with oracle support guidance, not discovery of supports or native success.')
(P/f'FIXED_SUPPORT_PRODUCT_TOY_{version}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
