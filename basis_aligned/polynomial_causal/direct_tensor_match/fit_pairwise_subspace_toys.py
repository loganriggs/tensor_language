"""Recover planted pairwise-shared input supports before native use."""
from pathlib import Path
import json,torch,time,math
from pairwise_subspace_loss import PairwiseSubspaceLoss
P=Path(__file__).parent;torch.set_num_threads(2)
plan=json.loads((P/'PAIRWISE_SUBSPACE_TOY_PLAN_V1.json').read_text());rows=[];t0=time.monotonic()
for case in range(5):
 rng=torch.Generator().manual_seed(32000+case);r=1+case%3;p=2+case%2;d=2*r+p+3
 rand=lambda *s:torch.randn(*s,dtype=torch.float64,generator=rng)
 truth=[rand(d,r) for _ in range(3)]+[rand(d,p) for _ in range(3)]
 helper=PairwiseSubspaceLoss(torch.stack([torch.eye(d,dtype=torch.float64)]*6));targets=[]
 for U in helper.spans(truth):
  K=rand(2,2*r+p,2*r+p);K=(K+K.transpose(-1,-2))/2;targets.append(U@K@U.T)
 T=torch.cat(targets);metric=PairwiseSubspaceLoss(T);oracle=float(metric.loss(truth,True));assert oracle<1e-20
 for seed in plan['seeds']:
  init=torch.Generator().manual_seed(33000+seed);params=[torch.nn.Parameter(torch.randn(a.shape,dtype=a.dtype,generator=init)/d**.5) for a in truth];opt=torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf')
  for step in range(plan['steps']+1):
   opt.zero_grad();loss=metric.loss(params);v=float(loss.detach());assert math.isfinite(v)
   if v<best:best=v;saved=[a.detach().clone() for a in params]
   if step==plan['steps']:break
   opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
  with torch.no_grad():explicit=float(metric.loss(saved,True));assert abs(explicit-best)<1e-8
  row=dict(case=case,seed=seed,pair_shared_width=r,private_width=p,input_width=d,error=explicit**.5,oracle_squared_error=oracle);rows.append(row);print(json.dumps(row),flush=True)
errors=[min(r['error'] for r in rows if r['case']==case) for case in range(5)];count=sum(e<=plan['threshold'] for e in errors)
out=dict(plan=plan,records=rows,best_restart_errors=errors,recovered_cases=count,recovery_pass=count>=plan['required_cases'],seconds=time.monotonic()-t0,scope='Five generic dense-core planted pairwise-sharing structures; random starts. Establishes optimizer behavior only, not sparse circuit extraction or native success.')
(P/'PAIRWISE_SUBSPACE_TOY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
