"""Weights-derived exact support constraints for planted mixed-slot recovery.
Supports depend only on target tensors, never on planted parameters.
"""
from pathlib import Path
import json,time,math,torch
from check_overlap_varpro import fixture
from overlap_varpro import OverlapMetric
P=Path(__file__).parent;torch.set_num_threads(2)
def supports(target):
 bases=[]
 for j in range(3):
  T=target[2*j:2*j+2];e,U=torch.linalg.eigh((T@T).sum(0));bases.append(U[:,e>1e-10*e[-1]])
 e,U=torch.linalg.eigh(sum(B@B.T for B in bases));common=U[:,e>3-1e-8]
 return common,bases

def main():
 plan=json.loads((P/'SUPPORT_GUIDED_TOY_PLAN_V1.json').read_text());rows=[];controls=[];states={};start=time.perf_counter()
 for case in range(5):
  target,templates,truth=fixture(case,mixed=True);common,bases=supports(target);metric=OverlapMetric(target,templates);r=truth[0].shape[1];assert common.shape[1]>=r
  def expand(params):
   actual=[common@params[0]]
   for j in range(3):actual.extend([params[1+2*j],bases[j]@params[2+2*j]])
   return actual
  known=[common.T@truth[0]]
  for j in range(3):known.extend([truth[1+2*j],bases[j].T@truth[2+2*j]])
  oracle=expand(known);replay=max(float((a-b).norm()/b.norm()) for a,b in zip(oracle,truth));assert replay<1e-8
  controls.append(dict(case=case,pair_support_dimensions=[B.shape[1] for B in bases],intersection_dimension=common.shape[1],planted_shared_width=r,oracle_parameter_replay=replay))
  for seed in [0,1]:
   g=torch.Generator().manual_seed(28000+seed);params=[torch.nn.Parameter(torch.randn(p.shape,dtype=p.dtype,generator=g)/p.shape[0]**.5) for p in known];opt=torch.optim.Muon(params,lr=.03,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');initial=None
   for step in range(plan['steps']+1):
    opt.zero_grad();loss,_,_=metric.loss(expand(params));value=float(loss.detach());assert math.isfinite(value)
    if initial is None:initial=value
    if value<best:best=value;saved=[p.detach().clone() for p in params];beststep=step
    if step==plan['steps']:break
    opt.param_groups[0]['lr']=.03*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
   with torch.no_grad():explicit=float(metric.loss(expand(saved),dense=True)[0]);assert abs(explicit-best)<1e-8
   states[f'{case}_{seed}']=dict(params=saved,common=common,bases=bases);row=dict(case=case,seed=seed,initial_error=max(0,initial)**.5,final_error=max(0,best)**.5,best_step=beststep);rows.append(row);print(json.dumps(row),flush=True)
 errors=[min(r['final_error'] for r in rows if r['case']==case) for case in range(5)];out=dict(plan=plan,controls=controls,records=rows,best_restart_errors=errors,recovered_cases=sum(e<=.01 for e in errors),recovery_pass=sum(e<=.01 for e in errors)>=4,seconds=time.perf_counter()-start,scope='Random coefficients within weight-derived exact pair supports and their common intersection. This changes the optimization parameterization, not the target or held recovery threshold. Does not retroactively pass unrestricted random initialization. Native full-rank forms need approximate learned supports; no native success asserted.')
 (P/'SUPPORT_GUIDED_TOY_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(states,P/'SUPPORT_GUIDED_TOY_STATES_V1.pt');print(errors,out['recovery_pass'],flush=True)
if __name__=='__main__':main()
