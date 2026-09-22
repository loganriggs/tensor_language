"""Line-search baseline on the identical wide quartic controls."""
import argparse,json,time
import torch
from toy_local_quartic_residual import planted
from toy_paired_residual_learning import fit_readout
from quartic_cp_profile import normalize_factors
from correlated_gaussian_cp import metric
from audit_conditional_residual_accounting import P
class BudgetStop(Exception):pass

def main(initial_scale=False):
 torch.set_num_threads(2);start=time.monotonic();rows=[]
 for family in [0,1,3,4,5]:
  _,_,_,rf,rc=planted(0 if family==5 else family);rf=[a.clone() for a in rf]
  if family==5:rf[0][:]=rf[0][0].clone();rf[1][:]=rf[1][0].clone()
  rf=[torch.nn.functional.pad(a,(0,1148)) for a in rf];mu=torch.zeros(1152,dtype=torch.float64);mu[:4]=torch.tensor([.1,-.2,.3,.1]);rb=[f@mu for f in rf];norms={name:((rc.T@rc)*metric(rf,rb,rf,rb,rho)).sum() for name,rho in [('value',None),('response',.5)]}
  for seed in [0,1]:
   g=torch.Generator().manual_seed(24100+seed);base=[torch.randn(4,4,dtype=torch.float64,generator=g) for _ in range(4)];extra=torch.Generator().manual_seed(40000+seed);params=[torch.cat([a,torch.randn(4,1148,dtype=torch.float64,generator=extra)],dim=1).contiguous().requires_grad_() for a in base]
   opt=torch.optim.LBFGS(params,lr=1,max_iter=250,max_eval=500,tolerance_grad=1e-12,tolerance_change=1e-14,history_size=50,line_search_fn='strong_wolfe');calls=0;best=None;early=None;fitstart=time.monotonic();budget_stop=False;normalizer=None
   def closure():
    nonlocal calls,best,early,normalizer
    if calls>=500:raise BudgetStop()
    calls+=1;opt.zero_grad();fs=normalize_factors(params);bs=[f@mu for f in fs];loss,C=fit_readout(metric(fs,bs,fs,bs),rc@metric(rf,rb,fs,bs));loss=loss/norms['value']
    if normalizer is None:normalizer=max(abs(float(loss.detach())),1e-10) if initial_scale else 1.
    loss=loss/normalizer;value=float(loss.detach());assert torch.isfinite(loss)
    if best is None or value<best[0]:best=(value,calls,[f.detach().clone() for f in fs],C.detach().clone())
    if calls<=251:early=best
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in params);return loss
   try:opt.step(closure)
   except BudgetStop:budget_stop=True
   reports={}
   with torch.no_grad():
    for name,checkpoint in [('first251',early),('final',best)]:
     value,selected,fs,C=checkpoint;bs=[f@mu for f in fs];errors={}
     for m,rho in [('value',None),('response',.5)]:
      G=metric(fs,bs,fs,bs,rho);X=rc@metric(rf,rb,fs,bs,rho);energy=norms[m]+((C.T@C)*G).sum()-2*(C*X).sum();assert energy>-1e-9*norms[m];errors[m+'_error']=float((energy.clamp_min(0)/norms[m]).sqrt())
     reports[name]=dict(selected_evaluation=selected,normalized_objective=value,**errors)
   row=dict(initial_scale=initial_scale,objective_normalizer=normalizer,family=family,seed=seed,evaluations=calls,iterations=int(opt.state[params[0]].get('n_iter',0)),hard_budget_stop=budget_stop,seconds=time.monotonic()-fitstart,**reports);rows.append(row);print(row,flush=True)
 out=dict(rows=rows,pred_no_immediate_stop=all(r['evaluations']>1 for r in rows),prediction=sum(r['final']['value_error']<.05 for r in rows)>=8,seconds=time.monotonic()-start,scope='Same planted wide axis-aligned quartics and random starts; adaptive line-search baseline, evaluation-label-free exact moment objective. 251-evaluation checkpoint plus at most500gradient evaluations; no native optimizer selection/adoption.')
 (P/('WIDE_QUARTIC_LBFGS_SCALE_V1.json' if initial_scale else 'WIDE_QUARTIC_LBFGS_V1.json')).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--initial-scale',action='store_true');args=parser.parse_args();main(args.initial_scale)
