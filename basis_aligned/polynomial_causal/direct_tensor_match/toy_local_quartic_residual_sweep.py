"""Longer/rate-varied followup to the short output-local residual toy pilot."""
import json,time
import torch
from toy_local_quartic_residual import planted
from local_quartic_residual import objective
from quartic_cp_profile import normalize_factors
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import project_shifted
from audit_conditional_residual_accounting import P

def main():
 torch.set_num_threads(2);start=time.monotonic();rows=[]
 for family in range(5):
  t,pf,pc,rf,rc=planted(family);S=torch.eye(4,dtype=torch.float64);mu=torch.tensor([.1,-.2,.3,.1],dtype=torch.float64);proj=project_shifted(t,mu);norm=((rc.T@rc)*gram_dynamic(rf,[a@mu for a in rf],rf,[a@mu for a in rf])).sum()
  for optim in ['adam','muon']:
   for rate in [.01,.1]:
    for seed in [0,1]:
     torch.manual_seed(24100+seed);params=[torch.randn(4,4,dtype=torch.float64,requires_grad=True) for _ in range(4)];opt=(torch.optim.Adam if optim=='adam' else torch.optim.Muon)(params,lr=rate);best=None
     for step in range(501):
      fs=normalize_factors(params);loss,C,info=objective(t,t,mu,proj,S,mu,pf,pc,fs,2,ridge=1e-6);value=float(loss.detach())
      if best is None or value<best[0]:best=(value,step,[f.detach().clone() for f in fs],C.detach().clone())
      if step==500:break
      opt.zero_grad();(loss/norm).backward();opt.step()
     value,step,fs,C=best;G=gram_dynamic(fs,[a@mu for a in fs],fs,[a@mu for a in fs]);cross=rc@gram_dynamic(rf,[a@mu for a in rf],fs,[a@mu for a in fs]);err=(norm+((C.T@C)*G).sum()-2*(C*cross).sum()).clamp_min(0)/norm
     row=dict(family=family,optimizer=optim,lr=rate,seed=seed,selected_step=step,relative_error=float(err.sqrt()),objective=value);rows.append(row);print(json.dumps(row),flush=True)
 out=dict(rows=rows,steps=500,rates=[.01,.1],restarts=2,seconds=time.monotonic()-start,scope='Followup40fits, samefiveplantednativequarticfamilies as100steppilot, choosecheckpointbyfittingobjective. ExactGaussianpopulationerror. Fixedtwoadditionalrates and500steps, not universaloptimizerbenchmark. Witnesses separatelyvalidatecapacity.')
 (P/'LOCAL_QUARTIC_RESIDUAL_TOY_SWEEP_V2.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
