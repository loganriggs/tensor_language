"""Registered optimizer/rate comparison on five planted overlapping-reader graphs."""
from pathlib import Path
import json,time,math,torch
from check_overlap_varpro import fixture
from overlap_varpro import OverlapMetric
P=Path(__file__).parent;torch.set_num_threads(2)
def main():
 plan=json.loads((P/'TOY_OVERLAP_OPTIMIZER_PLAN_V1.json').read_text());rows=[];start=time.perf_counter()
 for case in range(5):
  targets,templates,truth=fixture(case);metric=OverlapMetric(targets,templates)
  for optimizer in plan['optimizers']:
   for rate in plan['rates']:
    for seed in plan['seeds']:
     g=torch.Generator().manual_seed(28000+seed);params=[torch.nn.Parameter(torch.randn(p.shape,dtype=p.dtype,generator=g)/p.shape[0]**.5) for p in truth]
     opt=torch.optim.Adam(params,lr=rate) if optimizer=='adam' else torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
     best=float('inf');initial=None
     for step in range(plan['steps']+1):
      opt.zero_grad();loss,_,_=metric.loss(params);value=float(loss.detach());assert math.isfinite(value)
      if initial is None:initial=value
      if value<best:best=value;saved=[p.detach().clone() for p in params];beststep=step
      if step==plan['steps']:break
      opt.param_groups[0]['lr']=rate*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
     with torch.no_grad():
      explicit=float(metric.loss(saved,dense=True)[0]);assert abs(explicit-best)<1e-8
     row=dict(case=case,optimizer=optimizer,rate=rate,seed=seed,initial_objective=initial,best_objective=best,regularized_root_error=max(0,best)**.5,best_step=beststep);rows.append(row);print(json.dumps(row),flush=True)
 scores=[]
 for optimizer in plan['optimizers']:
  for rate in plan['rates']:
   group=[r for r in rows if r['optimizer']==optimizer and r['rate']==rate];score=sum(math.log(max(r['regularized_root_error'],1e-8)) for r in group)/len(group);best_cases=[min(r['regularized_root_error'] for r in group if r['case']==case) for case in range(5)]
   scores.append(dict(optimizer=optimizer,rate=rate,geometric_mean_error=math.exp(score),best_restart_errors=best_cases,recovered_cases=sum(e<=.01 for e in best_cases)))
 winner=min(scores,key=lambda x:x['geometric_mean_error']);out=dict(plan=plan,records=rows,summaries=scores,winner=winner,seconds=time.perf_counter()-start,predictions=dict(instrument=True,optimization_recovery=winner['recovered_cases']>=4),scope='Selection uses five planted graph fixtures and two fully random starts per setting. This is optimizer evidence for this topology, not a universal Adam/Muon ranking or native success guarantee.')
 (P/'TOY_OVERLAP_OPTIMIZER_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summaries']),flush=True)
if __name__=='__main__':main()
