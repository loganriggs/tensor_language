"""Bounded curvature diagnostic on all saved mixed-slot Muon fits, no native run."""
from pathlib import Path
import json,time,torch
from check_overlap_varpro import fixture
from overlap_varpro import OverlapMetric
P=Path(__file__).parent;torch.set_num_threads(2)
def main():
 plan=json.loads((P/'OVERLAP_TOY_POLISH_PLAN_V1.json').read_text());parents=torch.load(P/'TOY_OVERLAP_OPTIMIZER_STATES_CROSS_V2.pt',weights_only=True);rows=[];states={};start=time.perf_counter()
 for case in range(5):
  targets,templates,_=fixture(case,mixed=True);metric=OverlapMetric(targets,templates)
  for seed in [0,1]:
   key=f'{case}_muon_0.03_{seed}';params=[p.clone().requires_grad_() for p in parents[key]];initial=float(metric.loss(params)[0].detach());best=initial;saved=[p.detach().clone() for p in params];calls=0;history=[]
   opt=torch.optim.LBFGS(params,lr=1.,max_iter=200,max_eval=300,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
   def closure():
    nonlocal best,saved,calls
    opt.zero_grad();loss,_,_=metric.loss(params);assert torch.isfinite(loss);loss.backward();value=float(loss.detach());calls+=1
    if value<best:best=value;saved=[p.detach().clone() for p in params]
    if calls==1 or calls%50==0:history.append(dict(evaluation=calls,objective=value))
    return loss
   opt.step(closure)
   with torch.no_grad():explicit=float(metric.loss(saved,dense=True)[0]);assert abs(explicit-best)<1e-8
   states[key]=saved;row=dict(case=case,seed=seed,initial_error=max(0,initial)**.5,polished_error=max(0,best)**.5,evaluations=calls,history=history);rows.append(row);print(json.dumps(row),flush=True)
 errors=[min(r['polished_error'] for r in rows if r['case']==case) for case in range(5)];out=dict(plan=plan,records=rows,best_restart_errors=errors,recovered_cases=sum(x<=.01 for x in errors),recovery_pass=sum(x<=.01 for x in errors)>=4,seconds=time.perf_counter()-start,scope='All five fixtures and both fixed Muon starts, same L-BFGS budget. This tests whether curvature-aware refinement repairs the mixed-wiring recovery failure. Does not retroactively pass the pure-Muon experiment.')
 torch.save(states,P/'OVERLAP_TOY_POLISHED_STATES_V1.pt');(P/'OVERLAP_TOY_POLISH_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['best_restart_errors'],out['recovery_pass'],flush=True)
if __name__=='__main__':main()
