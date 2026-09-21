from pathlib import Path
import json,time,math,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from joint_orthogonal_private import JointOrthogonalPrivateMetric
from pairwise_exact_supports import supports
P=Path(__file__).parent;torch.set_num_threads(2);plan=json.loads((P/'JOINT_SUPPORTS_TOY_PLAN_V1.json').read_text());rows=[];start=time.monotonic()
for case in range(5):
 T,bases,private,_,_=fixture(case);shared=[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS];metric=JointOrthogonalPrivateMetric(T,shared);U=supports(T);oracle=[u.T@v for u,v in zip(U,list(bases)+list(private))];expand=lambda params:[u@v for u,v in zip(U,params)]
 for optimizer in plan['optimizers']:
  for seed in plan['seeds']:
   rng=torch.Generator().manual_seed(42000+seed);params=[torch.nn.Parameter(torch.randn(p.shape,dtype=p.dtype,generator=rng)/p.shape[0]**.5) for p in oracle];opt=torch.optim.Adam(params,lr=plan['rate']) if optimizer=='adam' else torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf')
   for step in range(plan['steps']+1):
    opt.zero_grad();loss=metric.loss(expand(params))[0];v=float(loss.detach());assert math.isfinite(v)
    if v<best:best=v;saved=[p.detach().clone() for p in params]
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
   with torch.no_grad():explicit=float(metric.loss(expand(saved),dense=True)[0]);assert abs(explicit-best)<1e-8
   row=dict(case=case,optimizer=optimizer,seed=seed,error=explicit**.5,basis_norms=[float(v.norm()) for v in saved]);rows.append(row);print(json.dumps(row),flush=True)
summaries=[]
for optimizer in plan['optimizers']:
 group=[r for r in rows if r['optimizer']==optimizer];errors=[min(r['error'] for r in group if r['case']==case) for case in range(5)];summaries.append(dict(optimizer=optimizer,best_restart_errors=errors,recovered_cases=sum(e<=plan['threshold'] for e in errors),geometric_mean_error=math.exp(sum(math.log(max(r['error'],1e-8)) for r in group)/len(group))))
winner=min(summaries,key=lambda r:r['geometric_mean_error']);out=dict(plan=plan,records=rows,summaries=summaries,winner=winner,recovery_pass=winner['recovered_cases']>=plan['required_cases'],seconds=time.monotonic()-start,scope='Five unchanged whole-block producttargets; random coefficients within target-derived exact supports; five instances of this family, symmetriccoreseliminated. Optimizercomparison specifictothisparameterization.')
(P/'JOINT_SUPPORTS_TOY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
